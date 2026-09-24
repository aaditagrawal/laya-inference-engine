"""Measure a completed request in a fresh native, compiled, or AOT process."""

import time

ENTRY_TIME = time.perf_counter()

import argparse
import json
import os
import statistics
import traceback
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=["aot", "aot-fast", "native", "compile"], required=True
    )
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cache-state", choices=["fresh", "reused"], required=True)
    parser.add_argument("--repeats", type=int, default=50)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "mode": args.mode,
        "cache_state": args.cache_state,
        "scope": "Fresh Python process through completed request. Includes imports, tokenizer/config/weights, CUDA Graph construction, tokenization, host/device transfers, inference, and formatting. Existing local weights and native extensions, warm OS file cache; excludes HTTP and one-time AOT artifact creation.",
        "status": "importing",
        "local_files_only": True,
        "cache_environment": {
            key: os.environ.get(key)
            for key in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR")
        },
    }
    report["cache_files_before"] = {
        key: sum(p.is_file() for p in Path(value).rglob("*"))
        if value and Path(value).exists()
        else 0
        for key, value in report["cache_environment"].items()
    }
    if args.cache_state == "fresh" and (
        not all(report["cache_environment"].values())
        or any(report["cache_files_before"].values())
    ):
        parser.error(
            "Fresh requires explicit empty or absent compiler cache directories"
        )
    engine = None
    try:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        import numpy as np
        import torch
        from huggingface_hub import snapshot_download

        from experiments.latency.serving.graph_adapter import replace_adapter
        from experiments.native.engine import ExperimentalEngine
        from laya_blackwell.engine import KEYS
        from laya_blackwell.workloads import workload

        from .engine import PackageEngine

        report["imports_ms"] = (time.perf_counter() - ENTRY_TIME) * 1000
        torch.set_num_threads(4)
        manifest = json.loads(args.package.with_suffix(".manifest.json").read_text())
        count, length = manifest["case"].split("-")
        request = workload(int(count), length)
        setup_start = time.perf_counter()
        reference_model = (
            snapshot_download(
                manifest["model_id"],
                revision=manifest["model_revision"],
                local_files_only=True,
                allow_patterns=[
                    "model.safetensors",
                    "rl_agent_config.json",
                    "encoder/*",
                    "tokenizer/*",
                ],
            )
            if not args.mode.startswith("aot")
            else None
        )
        engine = (
            PackageEngine(
                args.package,
                loader="checked-cpp" if args.mode == "aot-fast" else "standard",
            )
            if args.mode.startswith("aot")
            else ExperimentalEngine(
                mode="compiled" if args.mode == "compile" else "native-window",
                model=reference_model,
            )
        )
        if not args.mode.startswith("aot"):
            replace_adapter(engine)
        report["graph_adapter"] = "StableHostAdapter with uniquely owned capture stream"
        report["tokenizer_loader"] = getattr(
            engine.base, "tokenizer_loader", "upstream-auto-tokenizer"
        )
        report["model_and_adapter_setup_ms"] = (
            time.perf_counter() - setup_start
        ) * 1000
        report["hardware"] = engine.hardware
        report["status"] = "first_request"
        first_start = time.perf_counter()
        response = engine.predict(**request)
        report["first_predict_ms"] = (time.perf_counter() - first_start) * 1000
        report["entry_to_first_response_ms"] = (time.perf_counter() - ENTRY_TIME) * 1000
        report["response"] = {k: v for k, v in response.items() if k != "engine"}
        fixture = torch.load(args.package.with_suffix(".inputs.pt"), weights_only=True)
        prepared = engine.prepare(**request)
        logits, actions, _ = engine.run_prepared(prepared)
        report["exact_outputs"] = [
            bool(np.array_equal(actual, expected.numpy()))
            for actual, expected in zip((logits, actions), fixture["expected"])
        ]
        report["max_output_error"] = [
            float(np.max(np.abs(actual - expected.float().numpy())))
            for actual, expected in zip((logits, actions), fixture["expected"])
        ]
        slot = engine.adapter.graphs[engine.base._graph_key(prepared)]
        report["prepared_inputs_exact"] = all(
            torch.equal(slot.host[key], expected)
            for key, expected in zip(KEYS, fixture["inputs"])
        )
        for _ in range(10):
            engine.predict(**request)
        samples = []
        for _ in range(args.repeats):
            before = time.perf_counter()
            engine.predict(**request)
            samples.append((time.perf_counter() - before) * 1000)
        report["warm_predict_p50_ms"] = statistics.median(samples)
        report["warm_samples_ms"] = samples
        if args.mode.startswith("aot"):
            bad_requests = [
                {"state": "", "questions": request["questions"]},
                {
                    "state": request["state"],
                    "questions": {
                        "a": next(iter(request["questions"].values())),
                        "b": next(iter(request["questions"].values())),
                    },
                },
            ]
            rejected = []
            for bad in bad_requests:
                try:
                    engine.base._graph_key(engine.prepare(**bad))
                except ValueError:
                    rejected.append(True)
                else:
                    rejected.append(False)
            report["unsupported_mask_and_batch_rejected"] = rejected
        report["cache_files_after"] = {
            key: sum(p.is_file() for p in Path(value).rglob("*"))
            if value and Path(value).exists()
            else 0
            for key, value in report["cache_environment"].items()
        }
        report["status"] = "complete"
    except Exception as error:  # noqa: BLE001 - retain failures in raw measurements.
        report["failed_stage"] = report["status"]
        report["status"] = "failed"
        report["error"] = str(error)
        report["traceback"] = traceback.format_exc()
        print(report["traceback"], flush=True)
    finally:
        if engine is not None:
            engine.close()
        report["elapsed_seconds"] = time.perf_counter() - ENTRY_TIME
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                {
                    k: v
                    for k, v in report.items()
                    if k not in ("warm_samples_ms", "response")
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
