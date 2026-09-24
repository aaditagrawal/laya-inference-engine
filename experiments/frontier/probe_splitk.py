"""Reproduce the installed cuBLAS MLP-output split-K rounding before tuning."""

import json
import random
from pathlib import Path

import torch

from experiments.latency.engine import V2Engine
from experiments.native import common

from .matmul import matmul
from .tune import timing


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    torch.manual_seed(924)
    path = Path("results/frontier/matmul-splitk.json")
    report = {"metadata": common.metadata(), "rows": []}
    with V2Engine(optimization="native", max_graphs=1) as engine:
        weights = [
            layer.mlp.Wo.weight for layer in engine.base.model.net.encoder.layers
        ]
        inputs = [
            torch.randn(1, 64, w.shape[1], device="cuda", dtype=torch.bfloat16)
            for w in weights
        ]
        expected = [torch.nn.functional.linear(x, w) for x, w in zip(inputs, weights)]
        base, samples = timing(
            lambda: [torch.nn.functional.linear(x, w) for x, w in zip(inputs, weights)]
        )
        report["rows"].append(
            {"field": "mlp.Wo", "variant": "cublas", "ms": base, "samples_ms": samples}
        )
        configs = [
            (bm, bn, bk, split, 4, stages)
            for bm, bn in [(32, 32), (32, 64), (64, 32), (64, 64)]
            for bk, split in [(32, 2), (32, 4), (32, 8), (64, 4)]
            for stages in [2, 3, 4]
        ]
        random.Random(924).shuffle(configs)
        for config in configs:
            row = {"field": "mlp.Wo", "variant": "bf16-partial", "config": config}
            try:

                def run(config=config):
                    return [
                        matmul(x, w, config, partial_bf16=True)
                        for x, w in zip(inputs, weights)
                    ]

                actual = run()
                row["mismatches"] = sum(
                    int((a != b).sum()) for a, b in zip(actual, expected)
                )
                row["max_abs_error"] = max(
                    float((a.float() - b.float()).abs().max())
                    for a, b in zip(actual, expected)
                )
                row["ms"], row["samples_ms"] = timing(run)
                row["speedup"] = base / row["ms"]
            except Exception as error:  # noqa: BLE001 - retain failed configurations
                row["error"] = str(error)
            report["rows"].append(row)
            path.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
