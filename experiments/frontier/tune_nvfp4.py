"""Validate native FP4 packing/arithmetic, then measure real Laya weight banks."""

import argparse
import json
import random
from pathlib import Path

import torch

from experiments.latency.engine import V2Engine
from experiments.native import common

from .nvfp4 import COMPILED, dequantize, matmul, quantize
from .tune import timing


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tma", action="store_true")
    args = parser.parse_args()
    mm = matmul
    compiled = COMPILED
    if args.tma:
        from .nvfp4_tma import COMPILED as tma_compiled
        from .nvfp4_tma import matmul as tma_mm

        mm, compiled = tma_mm, tma_compiled
    torch.set_num_threads(4)
    torch.manual_seed(924)
    path = Path(
        "results/frontier/matmul-nvfp4-tma.json"
        if args.tma
        else "results/frontier/matmul-nvfp4.json"
    )
    report = {
        "metadata": common.metadata(),
        "format": "FP4 E2M1, E4M3 blocks of 16, FP32 per-row outer scales",
        "rows": [],
    }
    with V2Engine(optimization="native", max_graphs=1) as engine:
        for field in ["attn.Wqkv", "attn.Wo", "mlp.Wi", "mlp.Wo"]:
            group, attr = field.split(".")
            weights = [
                getattr(getattr(layer, group), attr).weight
                for layer in engine.base.model.net.encoder.layers
            ]
            inputs = [
                torch.randn(1, 64, w.shape[1], device="cuda", dtype=torch.bfloat16)
                for w in weights
            ]
            packed = [quantize(w) for w in weights]
            expected = [
                torch.nn.functional.linear(x, w) for x, w in zip(inputs, weights)
            ]
            # FP32 GEMM of separately decoded inputs checks nibble/scale ordering.
            decoded = [
                dequantize(*quantize(x)) @ dequantize(*w).T
                for x, w in zip(inputs, packed)
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
                (bm, bn, bk, split, 4, stages)
                for bm, bn, bk in [
                    (32, 32, 64),
                    (32, 64, 64),
                    (64, 32, 64),
                    (64, 64, 64),
                    (32, 64, 128),
                    (64, 64, 128),
                ]
                for split in [1, 2, 4]
                for stages in [2, 4]
            ]
            if args.tma:
                configs = [
                    (bm, bn, bk, split, 4, stages)
                    for bm in [32, 64]
                    for bn in [64, 128]
                    for bk in [128, 256]
                    for split in [1, 2]
                    for stages in [2, 4]
                ]
            random.Random(924).shuffle(configs)
            for config in configs:
                row = {
                    "field": field,
                    "variant": "nvfp4-tma" if args.tma else "nvfp4",
                    "config": config,
                }
                try:

                    def run(inputs=inputs, packed=packed, config=config):
                        return [mm(x, *w, config) for x, w in zip(inputs, packed)]

                    actual = run()
                    row["relative_l2_vs_decoded"] = max(
                        float((a.float().view_as(b) - b).norm() / b.norm())
                        for a, b in zip(actual, decoded)
                    )
                    if row["relative_l2_vs_decoded"] > 0.005:
                        raise RuntimeError("FP4 packing/MMA check failed")
                    row["relative_l2_vs_bf16"] = max(
                        float((a.float() - b.float()).norm() / b.float().norm())
                        for a, b in zip(actual, expected)
                    )
                    row["mismatches"] = sum(
                        int((a != b).sum()) for a, b in zip(actual, expected)
                    )
                    row["ms"], row["samples_ms"] = timing(run)
                    row["speedup"] = base / row["ms"]
                    kernel = compiled[(64, *weights[0].shape, *config)]
                    row["mma_instruction"] = next(
                        (
                            line.strip().split(" {")[0]
                            for line in kernel.asm["ptx"].splitlines()
                            if "block_scale" in line and "mma.sync" in line
                        ),
                        None,
                    )
                    if not row["mma_instruction"]:
                        raise RuntimeError("Expected native block-scaled MMA")
                    if args.tma:
                        row["tma_instruction"] = next(
                            (
                                line.strip()
                                for line in kernel.asm["ptx"].splitlines()
                                if "cp.async.bulk.tensor" in line
                            ),
                            None,
                        )
                except Exception as error:  # noqa: BLE001 - retain failed configurations
                    row["error"] = str(error)
                report["rows"].append(row)
                path.write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
