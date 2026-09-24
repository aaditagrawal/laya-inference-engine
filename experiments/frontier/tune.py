"""Screen matrix configurations across distinct weights larger than GPU L2."""

import argparse
import json
import random
from pathlib import Path

import torch

from experiments.latency.engine import V2Engine
from experiments.native import common

from .matmul import matmul, quantize_fp8_weight, quantize_weight


def timing(call, repeats=12, rounds=3):
    for _ in range(2):
        call()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = call()  # Keep graph output storage alive throughout replay.
    samples = []
    for _ in range(rounds):
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        for _ in range(repeats):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / repeats)
    graph.reset()
    del output
    return sorted(samples)[len(samples) // 2], samples


def configurations(pipeline=False):
    if pipeline:
        return [
            (bm, bn, bk, 1, warps, stages)
            for bm, bn, bk in [
                (16, 32, 64),
                (16, 64, 64),
                (32, 32, 64),
                (32, 64, 64),
                (64, 32, 64),
                (64, 64, 64),
            ]
            for warps in [2, 4]
            for stages in [1, 2, 3, 4, 5]
        ]
    return [
        (bm, bn, bk, split, warps, stages)
        for bm, bn, bk, warps, stages in [
            (32, 32, 64, 4, 3),
            (32, 64, 64, 4, 3),
            (64, 32, 64, 4, 3),
            (64, 64, 32, 4, 3),
            (64, 64, 64, 4, 3),
            (64, 64, 128, 4, 3),
            (64, 128, 64, 4, 3),
            (64, 128, 64, 8, 3),
            (64, 128, 128, 8, 3),
            (64, 256, 64, 8, 3),
        ]
        for split in [1, 2, 4, 8]
    ]


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quant", action="store_true")
    parser.add_argument("--lossless", action="store_true")
    parser.add_argument("--mxfp8", action="store_true")
    parser.add_argument("--weight-fp8", action="store_true")
    parser.add_argument("--pipeline", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path("results/frontier/matmul-bf16.json")
    )
    parser.add_argument(
        "--fields", nargs="+", default=["attn.Wqkv", "attn.Wo", "mlp.Wi", "mlp.Wo"]
    )
    args = parser.parse_args()
    if sum([args.quant, args.lossless, args.mxfp8, args.weight_fp8]) > 1:
        parser.error("Choose one weight representation")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(924)
    report = {
        "metadata": common.metadata(),
        "method": "CUDA Graph with 28 distinct layer weights per replay, then median of three blocks. Larger than L2. Isolated operations require full-request confirmation.",
        "quantized": args.quant,
        "rows": [],
    }
    with V2Engine(optimization="native", max_graphs=1) as engine:
        for field in args.fields:
            group, attr = field.split(".")
            weights = [
                getattr(getattr(layer, group), attr).weight
                for layer in engine.base.model.net.encoder.layers
            ]
            inputs = [
                torch.randn(1, 64, w.shape[1], device="cuda", dtype=torch.bfloat16)
                for w in weights
            ]
            expected = [
                torch.nn.functional.linear(x, w) for x, w in zip(inputs, weights)
            ]

            def baseline(inputs=inputs, weights=weights):
                return [
                    torch.nn.functional.linear(x, w) for x, w in zip(inputs, weights)
                ]

            base_ms, base_samples = timing(baseline)
            report["rows"].append(
                {
                    "field": field,
                    "variant": "cublas",
                    "ms": base_ms,
                    "samples_ms": base_samples,
                }
            )
            quantized = (
                [quantize_weight(w) for w in weights]
                if args.quant
                else [(w, None) for w in weights]
            )
            if args.weight_fp8:
                quantized = [quantize_fp8_weight(w) for w in weights]
            packed, mx_weights = [], []
            if args.lossless:
                from .lossless import matmul as packed_matmul
                from .lossless import pack

                packed = [pack(w) for w in weights]
                report.setdefault("packing", {})[field] = [w.report for w in packed]
            if args.mxfp8:
                from .mxfp8 import COMPILED, quantize
                from .mxfp8 import matmul as mx_matmul

                mx_weights = [quantize(w) for w in weights]
            configs = configurations(args.pipeline)
            random.Random(924).shuffle(configs)
            for config in configs:
                row = {
                    "field": field,
                    "config": config,
                    "variant": "fp8-weight"
                    if args.weight_fp8
                    else "mxfp8"
                    if args.mxfp8
                    else "lossless"
                    if args.lossless
                    else "int8-weight"
                    if args.quant
                    else "bf16",
                }
                try:

                    def candidate(
                        inputs=inputs,
                        config=config,
                        packed=packed,
                        mx_weights=mx_weights,
                        quantized=quantized,
                    ):
                        if args.lossless:
                            return [
                                packed_matmul(x, w, config)
                                for x, w in zip(inputs, packed)
                            ]
                        if args.mxfp8:
                            return [
                                mx_matmul(x, w, scales, config)
                                for x, (w, scales) in zip(inputs, mx_weights)
                            ]
                        return [
                            matmul(
                                x,
                                w,
                                config,
                                scale,
                                quant_mode=2 if args.weight_fp8 else None,
                            )
                            for x, (w, scale) in zip(inputs, quantized)
                        ]

                    actual = candidate()
                    row["mismatches"] = sum(
                        int((a != b).sum()) for a, b in zip(actual, expected)
                    )
                    row["max_abs_error"] = max(
                        float((a.float() - b.float()).abs().max())
                        for a, b in zip(actual, expected)
                    )
                    row["ms"], row["samples_ms"] = timing(candidate)
                    row["speedup"] = base_ms / row["ms"]
                    if args.mxfp8:
                        kernel = COMPILED[(64, *weights[0].shape, *config)]
                        row["native_block_scaled_mma"] = (
                            "kind::mxf8f6f4.block_scale" in kernel.asm["ptx"]
                        )
                        row["mma_instruction"] = next(
                            (
                                line.strip().split(" {")[0]
                                for line in kernel.asm["ptx"].splitlines()
                                if "kind::mxf8f6f4.block_scale" in line
                            ),
                            None,
                        )
                except Exception as error:  # noqa: BLE001 - retain failed configurations
                    row["error"] = str(error)
                report["rows"].append(row)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps(row), flush=True)
            winners = sorted(
                [
                    row
                    for row in report["rows"]
                    if row["field"] == field and "ms" in row
                ],
                key=lambda row: row["ms"],
            )
            print("BEST", field, json.dumps(winners[:3]), flush=True)


if __name__ == "__main__":
    main()
