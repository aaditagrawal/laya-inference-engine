"""Compare fused QKV/RoPE against the retained exact two-kernel path."""

import argparse
import itertools
import json
import random
from pathlib import Path

import torch

from experiments.native import common
from laya_blackwell.kernels import rope_qkv

from .engine import FrontierEngine
from .qkv_rope import project
from .tune import timing


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interleaved", action="store_true")
    parser.add_argument("--extended", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(14593)
    report = {
        "metadata": common.metadata(),
        "method": "28 distinct QKV matrices plus the model's per-layer RoPE",
        "interleaved": args.interleaved,
        "rows": [],
    }
    label = "-interleaved" if args.interleaved else "-extended" if args.extended else ""
    path = Path(f"results/frontier/qkv-rope{label}.json")
    with FrontierEngine(policy="bf16-splitk-exact", max_graphs=1) as engine:
        model = engine.base.model
        x = torch.randn((1, 64, 1024), device="cuda", dtype=torch.bfloat16)
        bank = [
            (
                layer.attn.Wqkv,
                model.global_cos if g else model.local_cos,
                model.global_sin if g else model.local_sin,
            )
            for layer, g in zip(model.net.encoder.layers, model.global_layers)
        ]

        def original():
            return [
                rope_qkv(module(x).view(1, 64, 3, 16, 64), c, s, fp32=True).view(
                    1, 64, 3072
                )
                for module, c, s in bank
            ]

        reference = original()
        weights = [
            module.weight.reshape(48, 2, 32, 1024)
            .transpose(1, 2)
            .contiguous()
            .reshape(3072, 1024)
            if args.interleaved
            else module.weight
            for module, _, _ in bank
        ]
        configs = list(
            itertools.product(
                (16, 32, 64), (64, 128), (32, 64), (4,), (2, 3, 4), (False,)
            )
        )
        if args.extended or args.interleaved:
            configs = list(
                itertools.product(
                    (16, 32, 64), (64, 128), (64,), (2, 4), (3, 4, 5, 6), (False,)
                )
            )
        random.Random(3814).shuffle(configs)
        for config in configs:
            row = {"config": config}
            try:

                def run(config=config):
                    return [
                        project(x, weight, c, s, config, args.interleaved)
                        for weight, (_, c, s) in zip(weights, bank)
                    ]

                actual = run()
                row["mismatches"] = sum(
                    int((a != b).sum()) for a, b in zip(actual, reference)
                )
                row["ms"], row["samples_ms"] = timing(run, repeats=20, rounds=3)
                row["baseline_ms"], row["baseline_samples_ms"] = timing(
                    original, repeats=20, rounds=3
                )
                row["speedup"] = row["baseline_ms"] / row["ms"]
            except Exception as error:  # noqa: BLE001 - record compiler/resource failures for each screened configuration.
                row["error"] = str(error)[:1500]
            report["rows"].append(row)
            path.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
