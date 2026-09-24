"""Verify the integrated v2 engine, safe graph ownership and shared-model lanes.

Run under the exclusive experiment flock. This is a functional check, not a
performance benchmark. All timing reports from the isolated experiments stay
unchanged.
"""

import argparse
import gc
import hashlib
import json
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from experiments.latency.engine import V2Engine
from experiments.latency.fusion.benchmark import benchmark_parity
from experiments.latency.serving.service import StreamService
from experiments.native.common import (
    CACHE,
    CASES,
    compare_outputs,
    metadata,
    validation_requests,
)
from laya_blackwell.workloads import workload


def exact(actual, expected):
    return all(np.array_equal(actual[i], expected[i]) for i in (0, 1))


def rejected_after_close(calls):
    result = {}
    for name, call in calls.items():
        try:
            call()
        except RuntimeError as exc:
            result[name] = {"rejected": True, "message": str(exc)}
        else:
            result[name] = {"rejected": False}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("results/latency-optimizations/integration.json")
    )
    parser.add_argument("--lanes", type=int, default=4)
    args = parser.parse_args()
    torch.set_num_threads(4)
    report = {
        "metadata": metadata(),
        "status": "running",
        "lanes": args.lanes,
        "fixtures": [],
        "concurrent": [],
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [
                Path(__file__),
                Path(__file__).parent / "engine.py",
                Path(__file__).parent / "fusion/adapter.py",
                Path(__file__).parent / "fusion/kernels.py",
                Path(__file__).parent / "fusion/best.json",
                Path(__file__).parent / "serving/graph_adapter.py",
                Path(__file__).parent / "serving/service.py",
            ]
        },
    }
    report["metadata"]["method"] = (
        "Functional fused-vs-native comparisons, fixture-cache check, changed-input replay, >8192-row fallback, concurrent lanes, owned-output lifetime and cleanup. No latency measurement."
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    native = fused = service = None
    service_outputs = []
    service_snapshots = []
    try:
        reference = np.load(CACHE / "outputs.npz", allow_pickle=False)
        requests = validation_requests()
        native = V2Engine(optimization="native", max_graphs=4)
        fused = V2Engine(optimization="fused", max_graphs=4)
        report["configuration"] = fused.fusion_metadata
        for index, request in enumerate(requests):
            prepared = fused.prepare(**request)
            baseline_prepared = native.prepare(**request)
            if prepared.items != baseline_prepared.items:
                raise RuntimeError(f"Prepared inputs differ in fixture {index}")
            expected, actual = (
                native.run_prepared(baseline_prepared),
                fused.run_prepared(prepared),
            )
            row = {
                "index": index,
                **compare_outputs(actual[:2], expected[:2], prepared, fused.agent),
                "baseline_matches_reference": exact(
                    expected, (reference[f"logits_{index}"], reference[f"act_{index}"])
                ),
                "shape": actual[2]["shape"],
            }
            report["fixtures"].append(row)
            if index % 10 == 0:
                print(json.dumps({"fixtures_completed": index + 1}), flush=True)
                save()
        report["benchmark_probes"] = benchmark_parity(native, fused, CASES)
        save()
        # The 16,384-row case must take the original projection paths.
        request = workload(32, "long")
        prepared = fused.prepare(**request)
        actual, expected = (
            fused.run_prepared(prepared),
            native.run_prepared(native.prepare(**request)),
        )
        report["fallback_32_long"] = {
            **compare_outputs(actual[:2], expected[:2], prepared, fused.agent),
            "shape": actual[2]["shape"],
            "max_rows": {
                key: val["max_rows"]
                for key, val in fused.fusion_metadata.items()
                if key in {"wi", "qkv"}
            },
        }
        save()
        # Service owns lane buffers and streams; the caller retains model ownership.
        mixed = [
            requests[index] for index in [0, 1, 4, 8, 9, 10, 15, 23, 34, 48, 61, 65]
        ]
        service = StreamService(fused.base, lanes=args.lanes, max_graphs=4)
        prepareds = [service.prepare(**request) for request in mixed]
        expected_outputs = [
            native.run_prepared(native.prepare(**request)) for request in mixed
        ]
        service.warm(mixed)
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [
                pool.submit(service.run_prepared, prepared) for prepared in prepareds
            ]
            service_outputs = [future.result(timeout=120) for future in futures]
        service_snapshots = [(row[0].copy(), row[1].copy()) for row in service_outputs]
        for index, (actual, expected, prepared) in enumerate(
            zip(service_outputs, expected_outputs, prepareds)
        ):
            report["concurrent"].append(
                {
                    "index": index,
                    **compare_outputs(actual[:2], expected[:2], prepared, fused.agent),
                    "lane": actual[2]["lane"],
                }
            )
        # Subsequent work overwrites every lane's staging without changing owned results.
        for prepared in reversed(prepareds):
            service.run_prepared(prepared)
        report["outputs_owned_after_replay"] = all(
            exact(row, snapshot)
            for row, snapshot in zip(service_outputs, service_snapshots)
        )
        service.close()
        service.close()
        report["service_after_close"] = rejected_after_close(
            {
                "run_prepared": lambda: service.run_prepared(prepareds[0]),
                "predict": lambda: service.predict(**mixed[0]),
                "prepare": lambda: service.prepare(**mixed[0]),
            }
        )
        # Closing a service must leave its caller-owned model usable.
        actual = fused.run_prepared(fused.prepare(**mixed[0]))
        report["base_survives_service_close"] = exact(actual, expected_outputs[0])
        report["closed_engines"] = {}
        for name, engine in [("native", native), ("fused", fused)]:
            engine.close()
            engine.close()
            report["closed_engines"][name] = rejected_after_close(
                {
                    "prepare": lambda engine=engine: engine.prepare(**mixed[0]),
                    "predict": lambda engine=engine: engine.predict(**mixed[0]),
                    "run_prepared": lambda engine=engine: engine.run_prepared(
                        prepareds[0]
                    ),
                }
            )
        report["outputs_owned_after_close"] = all(
            exact(row, snapshot)
            for row, snapshot in zip(service_outputs, service_snapshots)
        )
        gc.collect()
        torch.cuda.empty_cache()
        report["cuda_bytes_after_close"] = torch.cuda.memory_allocated()
        # The 128-KiB process-level GELU table is intentionally cached globally.
        from experiments.native.kernels import candidates

        lut = candidates.LUT
        report["shared_gelu_lut_bytes"] = (
            0 if lut is None else lut.numel() * lut.element_size()
        )
        report["all_fixture_outputs_exact"] = all(
            row["exact_logits_and_actions"] and row["baseline_matches_reference"]
            for row in report["fixtures"]
        )
        report["all_benchmark_outputs_exact"] = all(
            row["exact_logits_and_actions"] for row in report["benchmark_probes"]
        )
        report["all_concurrent_outputs_exact"] = all(
            row["exact_logits_and_actions"] for row in report["concurrent"]
        )
        report["fixture_decisions"] = sum(
            row["decisions"] for row in report["fixtures"]
        )
        report["benchmark_probe_decisions"] = sum(
            row["decisions"] for row in report["benchmark_probes"]
        )
        report["concurrent_decisions"] = sum(
            row["decisions"] for row in report["concurrent"]
        )
        checks = [
            report["all_fixture_outputs_exact"],
            report["all_benchmark_outputs_exact"],
            report["all_concurrent_outputs_exact"],
            report["fallback_32_long"]["exact_logits_and_actions"],
            report["base_survives_service_close"],
            report["outputs_owned_after_replay"],
            report["outputs_owned_after_close"],
            all(row["rejected"] for row in report["service_after_close"].values()),
            all(
                row["rejected"]
                for engine in report["closed_engines"].values()
                for row in engine.values()
            ),
            report["cuda_bytes_after_close"] == report["shared_gelu_lut_bytes"],
        ]
        if not all(checks):
            raise RuntimeError("An integration assertion failed; see the saved report")
        report["status"] = "passed"
        print(
            json.dumps(
                {
                    key: value
                    for key, value in report.items()
                    if key
                    in {
                        "status",
                        "fixture_decisions",
                        "benchmark_probe_decisions",
                        "concurrent_decisions",
                        "cuda_bytes_after_close",
                        "shared_gelu_lut_bytes",
                    }
                }
            ),
            flush=True,
        )
    except BaseException:
        report["status"] = "failed"
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        if service is not None:
            service.close()
        for engine in (fused, native):
            if engine is not None:
                engine.close()
        save()


if __name__ == "__main__":
    main()
