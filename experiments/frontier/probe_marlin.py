"""Check Marlin packing against dequantized weights, then measure full banks."""

import json
from pathlib import Path

import torch

from experiments.latency.engine import V2Engine
from experiments.native import common

from .marlin import PackedWeight
from .tune import timing


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    torch.manual_seed(924)
    report = {
        "metadata": common.metadata(),
        "rows": [],
        "scope": "Weight-only group32 INT8, BF16 activation. CUDA Graph timing over 28 distinct weight matrices.",
    }
    path = Path("results/frontier/matmul-marlin.json")
    with V2Engine(optimization="native", max_graphs=1) as engine:
        for field in ["attn.Wqkv", "attn.Wo", "mlp.Wi", "mlp.Wo"]:
            group, name = field.split(".")
            weights = [
                getattr(getattr(layer, group), name).weight
                for layer in engine.base.model.net.encoder.layers
            ]
            inputs = [
                torch.randn(1, 64, w.shape[1], device="cuda", dtype=torch.bfloat16)
                for w in weights
            ]
            packed = [PackedWeight(w) for w in weights]
            row = {"field": field}
            try:

                def baseline(inputs=inputs, weights=weights):
                    return [
                        torch.nn.functional.linear(x, w)
                        for x, w in zip(inputs, weights)
                    ]

                def candidate(inputs=inputs, packed=packed):
                    return [w(x) for x, w in zip(inputs, packed)]

                expected = [
                    torch.nn.functional.linear(x, w.reference)
                    for x, w in zip(inputs, packed)
                ]
                actual = candidate()
                row["max_error_vs_dequantized"] = max(
                    float((a.float() - b.float()).abs().max())
                    for a, b in zip(actual, expected)
                )
                row["relative_l2_vs_dequantized"] = max(
                    float((a.float() - b.float()).norm() / b.float().norm())
                    for a, b in zip(actual, expected)
                )
                if row["relative_l2_vs_dequantized"] > 0.01:
                    raise RuntimeError("Marlin packing or numerical check failed")
                row["baseline_ms"], _ = timing(baseline)
                row["candidate_ms"], row["samples_ms"] = timing(candidate)
                row["speedup"] = row["baseline_ms"] / row["candidate_ms"]
            except Exception as error:  # noqa: BLE001 - retain failed experiment evidence
                row["error"] = str(error)
            report["rows"].append(row)
            path.write_text(json.dumps(report, indent=2) + "\n")
            print(row, flush=True)


if __name__ == "__main__":
    main()
