"""Screen metadata-free, escape-free BF16 weight compression."""

import argparse
import json
import random
from pathlib import Path

import torch

from experiments.latency.engine import V2Engine
from experiments.native import common

from .float13 import matmul, pack
from .tune import timing


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiled", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(924)
    report = {
        "metadata": common.metadata(),
        "storage_ratio": 13 / 16,
        "tiled": args.tiled,
        "rows": [],
    }
    path = Path(
        "results/frontier/matmul-float13-tiled.json"
        if args.tiled
        else "results/frontier/matmul-float13.json"
    )
    with V2Engine(optimization="native", max_graphs=1) as engine:
        for field in ["attn.Wqkv", "attn.Wo", "mlp.Wi", "mlp.Wo"]:
            group, attr = field.split(".")
            weights = [
                getattr(getattr(layer, group), attr).weight
                for layer in engine.base.model.net.encoder.layers
            ]
            packed = [pack(w) for w in weights]
            inputs = [
                torch.randn(1, 64, w.shape[1], device="cuda", dtype=torch.bfloat16)
                for w in weights
            ]
            expected = [
                torch.nn.functional.linear(x, w) for x, w in zip(inputs, weights)
            ]
            base, samples = timing(
                lambda inputs=inputs, weights=weights: [
                    torch.nn.functional.linear(x, w) for x, w in zip(inputs, weights)
                ]
            )
            report["rows"].append(
                {"field": field, "variant": "cublas", "ms": base, "samples_ms": samples}
            )
            configs = [
                (bm, bn, bk, 4 if field == "mlp.Wo" else 1, 4, stages)
                for bm in [32, 64]
                for bn in [32, 64]
                for bk in [64, 128]
                for stages in [2, 4]
            ]
            random.Random(924).shuffle(configs)
            for config in configs:
                row = {"field": field, "variant": "float13", "config": config}
                try:

                    def run(inputs=inputs, packed=packed, config=config):
                        return [
                            matmul(x, w, config, tiled=args.tiled)
                            for x, w in zip(inputs, packed)
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
                except Exception as error:  # noqa: BLE001 - preserve failed probes
                    row["error"] = str(error)
                report["rows"].append(row)
                path.write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
