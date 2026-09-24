"""Exact domain proof and 28-weight fixed-geometry compact GELU screen."""

import hashlib
import json
import random
import statistics
from pathlib import Path

import torch

from experiments.latency.engine import V2Engine
from experiments.native import common

from .compact_gelu_kernel import binary_hashes, derive, exhaustive, project
from .mlp_geglu import COMPILED as REFERENCE_COMPILED
from .mlp_geglu import project as retained_project


def hashes():
    root = Path(__file__).parent
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.glob("compact_gelu*.py")) + [root / "mlp_geglu.py"]
    }


def reference_hashes():
    return {
        str(key): {
            kind: hashlib.sha256(
                value if isinstance(value, bytes) else value.encode()
            ).hexdigest()
            for kind, value in kernel.asm.items()
            if kind in ("cubin", "ptx")
        }
        for key, kernel in sorted(REFERENCE_COMPILED.items())
    }


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    torch.manual_seed(271433)
    before = hashes()
    table, ranges, domain = derive()
    proof = exhaustive(table, ranges)
    report = {
        "metadata": common.metadata(),
        "source_before": before,
        "domain": domain,
        "proof": proof,
        "rows": {},
        "scope": "28 distinct checkpoint MLP input weights and seeded BF16 activations; retained fixed 32x64x64 MMA, 4 warps, 3 stages, unpacked GEGLU, no math reassociation",
        "method": "Exclusive experiment lock; interleaved randomized variants for 9 CUDA-event rounds of 50 graph replays",
    }
    path = Path("results/frontier/compact_gelu-screen.json")
    if proof["bitwise_mismatches"]:
        path.write_text(json.dumps(report, indent=2) + "\n")
        raise RuntimeError("Exhaustive BF16 proof failed")
    with V2Engine(optimization="native", max_graphs=1) as engine:
        weights = [
            layer.mlp.Wi.weight for layer in engine.base.model.net.encoder.layers
        ]
        inputs = [
            torch.randn((1, 64, 1024), device=w.device, dtype=w.dtype) for w in weights
        ]
        cases = {
            "retained-erf-corrections": lambda: [
                retained_project(x, w, (32, 64, 64, 4, 3, 2, False))
                for x, w in zip(inputs, weights)
            ],
            "full-lut": lambda: [
                retained_project(x, w, (32, 64, 64, 4, 3, 2, True))
                for x, w in zip(inputs, weights)
            ],
            "compact-lut": lambda: [
                project(x, w, table, ranges) for x, w in zip(inputs, weights)
            ],
        }
        references = cases["retained-erf-corrections"]()
        graphs = {}
        for name, call in cases.items():
            actual = call()
            report["rows"][name] = {
                "bitwise_mismatches": sum(
                    int((a.view(torch.int16) != b.view(torch.int16)).sum())
                    for a, b in zip(actual, references)
                ),
                "samples_ms": [],
            }
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                outputs = call()
            graphs[name] = (graph, outputs)
        report["binary_before"] = binary_hashes()
        report["reference_binary_before"] = reference_hashes()
        report["order"] = []
        rng = random.Random(17911)
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
                report["rows"][name]["samples_ms"].append(start.elapsed_time(end) / 50)
        for row in report["rows"].values():
            row["median_ms"] = statistics.median(row["samples_ms"])
        reference = report["rows"]["retained-erf-corrections"]
        for row in report["rows"].values():
            row["speedup_vs_retained"] = reference["median_ms"] / row["median_ms"]
            row["faster_rounds_vs_retained"] = sum(
                a < b for a, b in zip(row["samples_ms"], reference["samples_ms"])
            )
        report["source_after"] = hashes()
        report["binary_after"] = binary_hashes()
        report["reference_binary_after"] = reference_hashes()
        assert before == report["source_after"]
        assert report["binary_before"] == report["binary_after"]
        assert report["reference_binary_before"] == report["reference_binary_after"]
        path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report["rows"], indent=2), flush=True)


if __name__ == "__main__":
    main()
