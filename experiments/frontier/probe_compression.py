"""Compare identical weights in plain/compressible VMM across 28 layers."""

import json
import random
from pathlib import Path

import torch

from experiments.latency.engine import V2Engine
from experiments.native import common

from .compression import Allocation
from .matmul import matmul
from .tune import timing


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    torch.manual_seed(9024)
    configurations = json.loads(Path("results/frontier/matmul-bf16.json").read_text())
    report = {
        "metadata": common.metadata(),
        "method": "28 distinct weights, exact values; randomized rounds, GPU-populated VMM allocations",
        "rows": [],
    }
    path = Path("results/frontier/compression.json")
    with V2Engine(optimization="native", max_graphs=1) as engine:
        for field in ["attn.Wqkv", "attn.Wo", "mlp.Wi", "mlp.Wo"]:
            group, attr = field.split(".")
            originals = [
                getattr(getattr(layer, group), attr).weight
                for layer in engine.base.model.net.encoder.layers
            ]
            shape = originals[0].shape
            x = torch.randn((1, 64, shape[1]), device="cuda", dtype=torch.bfloat16)
            candidates = [
                r
                for r in configurations["rows"]
                if r["field"] == field and r.get("mismatches") == 0 and "config" in r
            ]
            if not candidates:
                # Compare storage formats using identical arithmetic even when
                # no screened Triton tile matches cuBLAS on this projection.
                candidates = [
                    r
                    for r in configurations["rows"]
                    if r["field"] == field and "config" in r and "ms" in r
                ]
            config = tuple(min(candidates, key=lambda r: r["ms"])["config"])
            owners, weights = {}, {"torch-bf16": originals}
            try:
                for label, dtype, compressed in [
                    ("vmm-bf16", torch.bfloat16, False),
                    ("compressed-bf16", torch.bfloat16, True),
                    ("vmm-fp32", torch.float32, False),
                    ("compressed-fp32", torch.float32, True),
                ]:
                    owner = Allocation((28, *shape), dtype, compressed)
                    owners[label] = owner
                    owner.copy(torch.stack(originals))
                    weights[label] = list(owner.tensor.unbind())
                expected = [matmul(x, w, config) for w in originals]
                rows = {}
                for label, bank in weights.items():
                    converted = [w.to(torch.bfloat16) for w in bank]
                    rows[label] = {
                        "field": field,
                        "variant": label,
                        "config": config,
                        "allocation": owners[label].report()
                        if label in owners
                        else None,
                        "weight_mismatches": sum(
                            int((a != b).sum()) for a, b in zip(converted, originals)
                        ),
                        "mismatches": 0,
                        "samples_ms": [],
                    }
                    actual = [
                        matmul(
                            x,
                            w,
                            config,
                            quant_mode=3 if w.dtype == torch.float32 else 0,
                        )
                        for w in bank
                    ]
                    rows[label]["mismatches"] = sum(
                        int((a != b).sum()) for a, b in zip(actual, expected)
                    )
                rng = random.Random(924)
                for _ in range(5):
                    labels = list(weights)
                    rng.shuffle(labels)
                    for label in labels:
                        bank = weights[label]

                        def run(bank=bank, x=x, config=config):
                            return [
                                matmul(
                                    x,
                                    w,
                                    config,
                                    quant_mode=3 if w.dtype == torch.float32 else 0,
                                )
                                for w in bank
                            ]

                        _, samples = timing(run, repeats=20, rounds=1)
                        rows[label]["samples_ms"].extend(samples)
                for row in rows.values():
                    row["ms"] = sorted(row["samples_ms"])[len(row["samples_ms"]) // 2]
                    row["speedup"] = (
                        sorted(rows["torch-bf16"]["samples_ms"])[2] / row["ms"]
                    )
                    report["rows"].append(row)
                    print(json.dumps(row), flush=True)
                path.write_text(json.dumps(report, indent=2) + "\n")
            finally:
                # timing() has destroyed its graphs; all work finishes before unmap.
                weights.clear()
                for owner in owners.values():
                    owner.close()


if __name__ == "__main__":
    main()
