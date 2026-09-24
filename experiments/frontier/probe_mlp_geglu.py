"""Screen fused MLP/GEGLU across all 28 real encoder weight matrices."""

import argparse
import itertools
import json
import random
from pathlib import Path

import torch

from experiments.native import common
from experiments.native.kernels.candidates import triton_geglu_corrected

from .engine import FrontierEngine
from .mlp_geglu import COMPILED, pack, project
from .tune import timing


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--unpacked", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path("results/frontier/mlp-geglu.json")
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(74451)
    report = {
        "metadata": common.metadata(),
        "method": "28 distinct MLP input matrices and independent random inputs, graph replay. Retained tuned BF16 projection plus corrected GEGLU baseline. Isolated timing only.",
        "rows": [],
    }
    with FrontierEngine(policy="bf16-splitk-exact", max_graphs=1) as engine:
        modules = [layer.mlp.Wi for layer in engine.base.model.net.encoder.layers]
        inputs = [
            torch.randn(1, 64, 1024, device="cuda", dtype=torch.bfloat16)
            for _ in modules
        ]
        packed = [
            module.weight if args.unpacked else pack(module.weight)
            for module in modules
        ]

        def baseline():
            return [
                triton_geglu_corrected(module(x)) for x, module in zip(inputs, modules)
            ]

        reference = baseline()
        configs = list(
            itertools.product(
                (16, 32, 64),
                (32, 64, 128),
                (64,),
                (4,),
                (3, 4, 5),
                (0, 1),
                (False, True),
            )
        )
        if args.unpacked:
            configs = list(
                itertools.product(
                    (32,), (32, 64), (64,), (4,), (3, 4, 5), (2,), (False, True)
                )
            )
        random.Random(8931).shuffle(configs)
        for config in configs:
            row = {"config": config}
            try:

                def run(config=config):
                    return [project(x, w, config) for x, w in zip(inputs, packed)]

                actual = run()
                row["mismatches"] = sum(
                    int((a != b).sum()) for a, b in zip(actual, reference)
                )
                row["max_abs_error"] = max(
                    float((a.float() - b.float()).abs().max())
                    for a, b in zip(actual, reference)
                )
                row["ms"], row["samples_ms"] = timing(run, repeats=20, rounds=3)
                row["baseline_ms"], row["baseline_samples_ms"] = timing(
                    baseline, repeats=20, rounds=3
                )
                row["speedup"] = row["baseline_ms"] / row["ms"]
                kernel = COMPILED[config]
                row["shared_bytes"] = kernel.metadata.shared
                row["registers"] = kernel.n_regs
            except Exception as error:  # noqa: BLE001 - preserve resource/compiler failures.
                row["error"] = str(error)[:2000]
            report["rows"].append(row)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
