"""Compare the installed vendor FP4 GEMM with the native Triton experiment."""

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.latency.engine import V2Engine
from experiments.native import common

from .nvfp4 import dequantize, quantize
from .tune import timing


def matmul(x, weight, scale, outer):
    q, xscale, xouter = quantize(x, swizzle=True)
    result = F.scaled_mm(
        q.view(torch.float4_e2m1fn_x2),
        weight.view(torch.float4_e2m1fn_x2).t(),
        xscale.flatten(),
        F.ScalingType.BlockWise1x16,
        scale.flatten(),
        F.ScalingType.BlockWise1x16,
        swizzle_a=F.SwizzleType.SWIZZLE_32_4_4,
        swizzle_b=F.SwizzleType.SWIZZLE_32_4_4,
        output_dtype=torch.float32,
    )
    return (
        (result * (xouter[:, None] * outer[None, :]))
        .bfloat16()
        .view(*x.shape[:-1], weight.shape[0])
    )


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    torch.manual_seed(924)
    path = Path("results/frontier/matmul-nvfp4-vendor.json")
    report = {"metadata": common.metadata(), "rows": []}
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
            packed = [quantize(w, swizzle=True) for w in weights]
            row = {"field": field}
            try:
                expected = [
                    dequantize(*quantize(x)) @ dequantize(*quantize(w)).T
                    for x, w in zip(inputs, weights)
                ]

                def run(inputs=inputs, packed=packed):
                    return [matmul(x, *w) for x, w in zip(inputs, packed)]

                actual = run()
                row["relative_l2_vs_decoded"] = max(
                    float((a.float().view_as(b) - b).norm() / b.norm())
                    for a, b in zip(actual, expected)
                )
                if row["relative_l2_vs_decoded"] > 0.005:
                    raise RuntimeError("Vendor FP4 scale/packing check failed")
                row["baseline_ms"], _ = timing(
                    lambda inputs=inputs, weights=weights: [
                        F.linear(x, w) for x, w in zip(inputs, weights)
                    ]
                )
                row["ms"], row["samples_ms"] = timing(run)
                row["speedup"] = row["baseline_ms"] / row["ms"]
            except Exception as error:  # noqa: BLE001 - retain unsupported vendor paths
                row["error"] = str(error)
            report["rows"].append(row)
            path.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
