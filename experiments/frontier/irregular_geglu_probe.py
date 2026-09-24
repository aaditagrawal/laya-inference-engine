"""Exact real-activation checks followed by a bounded 28-weight comparison."""

import argparse
import fcntl
import json
import random
import statistics
from pathlib import Path

import torch

from experiments.native import common
from experiments.native.kernels.candidates import gelu_lut

from .cutlass_geglu import Operation as NativeOperation
from .cutlass_geglu_build import DIRECTORY as NATIVE_DIRECTORY
from .engine import FrontierEngine
from .holdout import requests
from .irregular_geglu import Operation, mapping
from .irregular_geglu_build import CUTLASS, DIRECTORY, ROOT, digest, key
from .mlp_geglu import COMPILED, pack, project
from .packed_bf16_probe import capture

CONFIG = [32, 64, 64, 4, 3, 2, False]


def sources():
    paths = list(Path(__file__).parent.glob("irregular_geglu*"))
    paths += [
        Path(__file__).parent / n
        for n in (
            "mlp_geglu.py",
            "packed_bf16_probe.py",
            "cutlass_geglu.py",
            "cutlass_geglu.cu",
        )
    ]
    return {str(p.relative_to(ROOT)): digest(p) for p in paths if p.is_file()}


def binaries(build):
    result = {
        r["key"]: digest(DIRECTORY / (r["key"] + ".so"))
        for r in build["rows"]
        if not r["returncode"]
    }
    result["original_native"] = digest(NATIVE_DIRECTORY / "cutlass_geglu.so")
    return result


def headers(build):
    values = {name: digest(CUTLASS / name) for name in build["cutlass_headers_sha256"]}
    assert values == build["cutlass_headers_sha256"]
    return values


def exact(actual, expected):
    return sum(
        int((a.view(torch.int16) != b.view(torch.int16)).sum())
        for a, b in zip(actual, expected)
    )


@torch.inference_mode()
def run(args):
    torch.set_num_threads(4)
    build = json.loads((DIRECTORY / "build.json").read_text())
    configs = [r["config"] for r in build["rows"] if not r["returncode"]]
    if not args.smoke:
        smoke = json.loads(
            (ROOT / "results/frontier/irregular_geglu-smoke.json").read_text()
        )
        accepted = {r["key"] for r in smoke["rows"] if r["mismatches"] == 0}
        configs = [c for c in configs if key(c) in accepted or c[4] == 2]
    report = {
        "metadata": common.metadata(),
        "build": build,
        "source_before": sources(),
        "binary_before": binaries(build),
        "headers_before": headers(build),
        "native_build": json.loads((NATIVE_DIRECTORY / "build.json").read_text()),
        "mode": "one-matrix legality smoke"
        if args.smoke
        else "28-distinct-weight bank",
        "retained_config": CONFIG,
        "rows": [],
        "order": [],
        "notes": "Fixed K64 mainloop and exact BF16 LUT epilogue. No shape padding. Direct stores use the pinned MMA accumulator mapping and are separately controlled.",
    }
    table = gelu_lut()
    with FrontierEngine(policy="bf16-exact", max_graphs=2) as engine:
        fixtures = [common.workload(1, "short"), *requests()[:2]]
        groups = [capture(engine, request) for request in fixtures]
        if args.smoke:
            groups = [records[:1] for records in groups]
        report["requests"] = fixtures
        report["matrix_count"] = len(groups[0])
        weights = [pack(w) for _, w in groups[0]]
        expected = [[project(x, w, CONFIG) for x, w in records] for records in groups]
        expected_raw = [
            [torch.nn.functional.linear(x, w) for x, w in records] for records in groups
        ]
        operations = {}
        for config in configs:
            name = key(config)
            row = {
                "key": name,
                "config": config,
                "mapping": mapping(config),
                "mismatches": 0,
                "raw_mismatches": 0,
                "samples_ms": [],
            }
            for idx, records in enumerate(groups):
                fused = [
                    Operation(x, w, table, config)
                    for (x, _), w in zip(records, weights)
                ]
                raw = [
                    Operation(x, w, table, config, False)
                    for (x, _), w in zip(records, weights)
                ]
                restored = [
                    op().view(1, 64, 2624, 2).transpose(2, 3).reshape(1, 64, 5248)
                    for op in raw
                ]
                row["raw_mismatches"] += exact(restored, expected_raw[idx])
                row["mismatches"] += exact([op() for op in fused], expected[idx])
                row["resources"] = fused[0].resources
                if idx == 0:
                    operations[name] = fused
            report["rows"].append(row)
            print(json.dumps(row), flush=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
        if not args.smoke:
            records = groups[0]
            native = [
                NativeOperation(x, w, table, 0, True)
                for (x, _), w in zip(records, weights)
            ]
            assert exact([op() for op in native], expected[0]) == 0
            calls = {
                "retained_triton": lambda: [project(x, w, CONFIG) for x, w in records],
                "native_32_64_3": lambda: [op() for op in native],
            }
            report["rows"] += [{"key": name, "samples_ms": []} for name in calls]
            for row in report["rows"]:
                if "config" in row and row["mismatches"] == row["raw_mismatches"] == 0:
                    ops = operations[row["key"]]
                    calls[row["key"]] = lambda ops=ops: [op() for op in ops]
            graphs = {}
            for name, call in calls.items():
                for _ in range(2):
                    call()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    outputs = call()
                graphs[name] = (graph, outputs)
            rows = {r["key"]: r for r in report["rows"]}
            rng = random.Random(984611)
            for _ in range(9):
                names = list(graphs)
                rng.shuffle(names)
                report["order"].append(names)
                for name in names:
                    graph = graphs[name][0]
                    for _ in range(3):
                        graph.replay()
                    start, end = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    start.record()
                    for _ in range(50):
                        graph.replay()
                    end.record()
                    end.synchronize()
                    rows[name]["samples_ms"].append(start.elapsed_time(end) / 50)
            for row in report["rows"]:
                if row["samples_ms"]:
                    row["median_ms"] = statistics.median(row["samples_ms"])
            report["native_resources"] = native[0].resources
        kernel = COMPILED[tuple(CONFIG)]
        report["triton_binary"] = {
            kind: __import__("hashlib")
            .sha256(value if isinstance(value, bytes) else value.encode())
            .hexdigest()
            for kind, value in kernel.asm.items()
            if kind in ("cubin", "ptx")
        }
        report["source_after"] = sources()
        report["binary_after"] = binaries(build)
        report["headers_after"] = headers(build)
        assert report["source_before"] == report["source_after"]
        assert report["binary_before"] == report["binary_after"]
        assert report["headers_before"] == report["headers_after"]
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "rows": report["rows"]}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.output is None:
        args.output = ROOT / (
            "results/frontier/irregular_geglu-smoke.json"
            if args.smoke
            else "results/frontier/irregular_geglu-bank.json"
        )
    with Path("/tmp/laya-gpu-experiments.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        run(args)


if __name__ == "__main__":
    main()
