"""Screen contiguous weight-tile reads against the retained exact kernels."""

import argparse
import itertools
import json
import random
from pathlib import Path

import torch

from experiments.native import common

from .engine import FrontierEngine
from .packed_tma import COMPILED, matmul, pack
from .tune import timing


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fields", nargs="+", default=["attn.Wqkv", "attn.Wo", "mlp.Wi", "mlp.Wo"]
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--access", type=int, choices=[0, 1, 2], default=2)
    args = parser.parse_args()
    if args.output is None:
        label = {0: "pointer", 1: "tma2", 2: "tma"}[args.access]
        args.output = Path(f"results/frontier/matmul-packed-{label}.json")
    torch.set_num_threads(4)
    torch.manual_seed(18283)
    report = {
        "metadata": common.metadata(),
        "method": "28 distinct BF16 layer matrices, exact blocked weights, baseline is retained tuned kernels",
        "rows": [],
        "access": args.access,
    }
    with FrontierEngine(policy="bf16-splitk-exact", max_graphs=1) as engine:
        for field in args.fields:
            group, attr = field.split(".")
            modules = [
                getattr(getattr(layer, group), attr)
                for layer in engine.base.model.net.encoder.layers
            ]
            x = [
                torch.randn(
                    (1, 64, module.weight.shape[1]), device="cuda", dtype=torch.bfloat16
                )
                for module in modules
            ]
            split = 4 if field == "mlp.Wo" else 1

            def baseline(x=x, modules=modules):
                return [module(a) for a, module in zip(x, modules)]

            reference = baseline()
            configs = list(
                itertools.product(
                    (16, 32, 64), (32, 64), (64,), (4,), (3, 4, 5), (False, True)
                )
            )
            random.Random(9471).shuffle(configs)
            packed = {}
            for bm, bn, bk, warps, stages, transposed in configs:
                row = {
                    "field": field,
                    "config": [bm, bn, bk, split, warps, stages, transposed],
                }
                try:
                    key = (bn, bk, transposed)
                    if key not in packed:
                        packed[key] = [
                            pack(module.weight, bn, bk, split, transposed)
                            for module in modules
                        ]
                    weights = packed[key]

                    def run(x=x, weights=weights, config=(bm, warps, stages)):
                        return [
                            matmul(a, w, config, access=args.access)
                            for a, w in zip(x, weights)
                        ]

                    actual = run()
                    row["mismatches"] = sum(
                        int((a != b).sum()) for a, b in zip(actual, reference)
                    )
                    row["ms"], row["samples_ms"] = timing(run, repeats=20, rounds=3)
                    row["baseline_ms"], row["baseline_samples_ms"] = timing(
                        baseline, repeats=20, rounds=3
                    )
                    row["speedup"] = row["baseline_ms"] / row["ms"]
                    sample = weights[0]
                    kernel = COMPILED[
                        (
                            sample.n,
                            sample.k,
                            bn,
                            bk,
                            split,
                            transposed,
                            bm,
                            warps,
                            stages,
                            args.access,
                        )
                    ]
                    row["shared_bytes"] = kernel.metadata.shared
                    row["tma_instruction"] = next(
                        (
                            line.strip()
                            for line in kernel.asm["ptx"].splitlines()
                            if "cp.async.bulk.tensor" in line
                        ),
                        None,
                    )
                except Exception as error:  # noqa: BLE001 - keep compiler/resource failures in the screening report.
                    row["error"] = str(error)[:2000]
                report["rows"].append(row)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
