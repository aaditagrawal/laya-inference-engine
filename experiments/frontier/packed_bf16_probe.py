"""Real-activation parity and fixed-geometry packed BF16 epilogue screen."""

import hashlib
import json
import random
import statistics
from pathlib import Path

import torch

from experiments.native import common

from .engine import FrontierEngine
from .holdout import requests
from .mlp_geglu import COMPILED as ORIGINAL_COMPILED
from .mlp_geglu import project as original
from .packed_bf16_kernel import COMPILED, binary_hashes, domain, inspect_sass, project


def source_hashes():
    root = Path(__file__).parent
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.glob("packed_bf16*.py")) + [root / "mlp_geglu.py"]
    }


@torch.inference_mode()
def capture(engine, request):
    prepared = engine.prepare(**request)
    key = engine.base._graph_key(prepared)
    assert key[:2] == (1, 64)
    host = engine.base._allocate(key[:3], host=True)
    engine.base._fill(host, prepared)
    inputs = {k: v.to(engine.base.device) for k, v in host.items()}
    inputs["global_attention_unmasked"] = key[-1]
    records, handles = [], []
    for layer in engine.base.model.net.encoder.layers:
        handles.append(
            layer.mlp.Wi.register_forward_pre_hook(
                lambda module, args: records.append((args[0], module.weight))
            )
        )
    try:
        engine.base._forward(inputs)
    finally:
        for handle in handles:
            handle.remove()
    assert len(records) == 28
    return records


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    before = source_hashes()
    path = Path("results/frontier/packed_bf16-screen.json")
    report = {
        "metadata": common.metadata(),
        "source_before": before,
        "domain": domain(),
        "config": [32, 64, 64, 4, 3, 2, False],
        "rows": {},
        "parity": [],
        "documentation": "https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#half-precision-floating-point-instructions-mul",
        "scope": "Only final BF16 GELU times BF16 gate product changed to explicit mul.rn.bf16x2. All GEMM/GELU arithmetic retained.",
    }
    if report["domain"]["bitwise_mismatches"]:
        path.write_text(json.dumps(report, indent=2) + "\n")
        return
    with FrontierEngine(policy="bf16-exact", max_graphs=2) as engine:
        fixtures = [common.workload(1, "short"), *requests()[:2]]
        groups = []
        for request in fixtures:
            records = capture(engine, request)
            groups.append(records)
            expected = [original(x, w, report["config"]) for x, w in records]
            actual = [project(x, w) for x, w in records]
            report["parity"].append(
                {
                    "request": request,
                    "layers": len(records),
                    "bitwise_mismatches": sum(
                        int((a.view(torch.int16) != b.view(torch.int16)).sum())
                        for a, b in zip(actual, expected)
                    ),
                }
            )
        report["sass"] = inspect_sass()
        if not report["sass"]["project"]["hmul2_bf16"]:
            raise RuntimeError("Native packed multiply not emitted")
        if any(r["bitwise_mismatches"] for r in report["parity"]):
            path.write_text(json.dumps(report, indent=2) + "\n")
            return
        records = groups[0]
        graphs = {}
        for name, call in {
            "retained": lambda: [original(x, w, report["config"]) for x, w in records],
            "packed": lambda: [project(x, w) for x, w in records],
        }.items():
            for _ in range(2):
                call()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                outputs = call()
            graphs[name] = (graph, outputs)
            report["rows"][name] = {"samples_ms": []}
        report["binary_before"] = binary_hashes()
        original_kernel = ORIGINAL_COMPILED[tuple(report["config"])]
        report["original_binary_sha256"] = {
            kind: hashlib.sha256(
                value if isinstance(value, bytes) else value.encode()
            ).hexdigest()
            for kind, value in original_kernel.asm.items()
            if kind in ("cubin", "ptx")
        }
        report["resources"] = {
            "retained": {
                "registers": original_kernel.n_regs,
                "shared": original_kernel.metadata.shared,
            },
            "packed": {
                "registers": COMPILED["project"].n_regs,
                "shared": COMPILED["project"].metadata.shared,
            },
        }
        rng = random.Random(73311)
        report["order"] = []
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
                for _ in range(100):
                    graph.replay()
                end.record()
                end.synchronize()
                report["rows"][name]["samples_ms"].append(start.elapsed_time(end) / 100)
        for row in report["rows"].values():
            row["median_ms"] = statistics.median(row["samples_ms"])
        a, b = report["rows"]["retained"], report["rows"]["packed"]
        report["speedup"] = a["median_ms"] / b["median_ms"]
        report["faster_rounds"] = sum(
            a > b for a, b in zip(a["samples_ms"], b["samples_ms"])
        )
        report["source_after"] = source_hashes()
        report["binary_after"] = binary_hashes()
        assert before == report["source_after"]
        assert report["binary_before"] == report["binary_after"]
        path.write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                {
                    k: report[k]
                    for k in ["rows", "speedup", "faster_rounds", "resources"]
                },
                indent=2,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
