"""Measure explicit cuBLASLt algorithms with 28 distinct layer matrices."""

import json
import random
from pathlib import Path

import torch

from experiments.latency.engine import V2Engine
from experiments.native import common

from .lt import Plan
from .tune import timing


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    torch.manual_seed(924)
    path = Path("results/frontier/matmul-cublaslt.json")
    report = {
        "metadata": common.metadata(),
        "method": "Explicit cuBLASLt heuristics, 64 MB workspace, 28 distinct weights per CUDA Graph replay, three timing blocks per algorithm. Random BF16 input; full-request validation is required.",
        "rows": [],
    }
    with V2Engine(optimization="native", max_graphs=1) as engine:
        for field in ["attn.Wqkv", "attn.Wo", "mlp.Wi", "mlp.Wo"]:
            group, attr = field.split(".")
            weights = [
                getattr(getattr(layer, group), attr).weight
                for layer in engine.base.model.net.encoder.layers
            ]
            n, k = weights[0].shape
            inputs = [
                torch.randn(1, 64, k, device="cuda", dtype=torch.bfloat16)
                for _ in weights
            ]
            expected = [
                torch.nn.functional.linear(x, w) for x, w in zip(inputs, weights)
            ]

            def baseline(inputs=inputs, weights=weights):
                return [
                    torch.nn.functional.linear(x, w) for x, w in zip(inputs, weights)
                ]

            baseline_ms, samples = timing(baseline)
            report["rows"].append(
                {
                    "field": field,
                    "variant": "pytorch",
                    "ms": baseline_ms,
                    "samples_ms": samples,
                }
            )
            plan = Plan(64, n, k)
            indices = list(range(plan.native.count()))
            random.Random(924).shuffle(indices)
            for index in indices:
                row = {
                    "field": field,
                    "variant": "cublaslt",
                    "index": index,
                    "algorithm": dict(plan.native.info(index)),
                }
                try:

                    def candidate(
                        plan=plan, index=index, inputs=inputs, weights=weights
                    ):
                        return [plan(x, w, index) for x, w in zip(inputs, weights)]

                    actual = candidate()
                    row["mismatches"] = sum(
                        int((a != b).sum()) for a, b in zip(actual, expected)
                    )
                    row["max_abs_error"] = max(
                        float((a.float() - b.float()).abs().max())
                        for a, b in zip(actual, expected)
                    )
                    row["ms"], row["samples_ms"] = timing(candidate)
                    row["speedup"] = baseline_ms / row["ms"]
                except Exception as error:  # noqa: BLE001 - retain failed algorithms
                    row["error"] = str(error)
                report["rows"].append(row)
                path.write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps(row), flush=True)
            winners = sorted(
                (
                    row
                    for row in report["rows"]
                    if row["field"] == field and "ms" in row
                ),
                key=lambda row: row["ms"],
            )
            print("BEST", field, json.dumps(winners[:3]), flush=True)
            del plan


if __name__ == "__main__":
    main()
