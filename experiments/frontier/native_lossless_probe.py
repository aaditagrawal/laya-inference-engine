"""Compare native fragment decoding with retained exact BF16 matrix kernels."""

import argparse
import json
import random
from pathlib import Path

import torch

from experiments.native import common

from .engine import FrontierEngine
from .native_lossless import load, matmul, pack
from .tune import timing


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fields", nargs="+", default=["attn.Wqkv", "attn.Wo", "mlp.Wi", "mlp.Wo"]
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/frontier/native-lossless.json")
    )
    parser.add_argument("--pipeline", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(20260924)
    load()
    report = {
        "metadata": common.metadata(),
        "seed": 20260924,
        "method": "28 distinct checkpoint matrices and independent random BF16 activations per field; graph replay; BF16 MMA register layout; split-4 BF16 partial rounding for MLPWo",
        "compressed_storage_ratio": 13 / 16,
        "rows": [],
    }
    with FrontierEngine(policy="bf16-splitk-exact", max_graphs=1) as engine:
        report["baseline_selection"] = engine.selection
        for field in args.fields:
            group, attr = field.split(".")
            modules = [
                getattr(getattr(layer, group), attr)
                for layer in engine.base.model.net.encoder.layers
            ]
            inputs = [
                torch.randn(
                    1, 64, m.weight.shape[1], device="cuda", dtype=torch.bfloat16
                )
                for m in modules
            ]
            expected = [module(x) for module, x in zip(modules, inputs)]

            def baseline(modules=modules, inputs=inputs):
                return [module(x) for module, x in zip(modules, inputs)]

            base, samples = timing(baseline, repeats=30, rounds=5)
            report["rows"].append(
                {
                    "field": field,
                    "variant": "retained-bf16",
                    "ms": base,
                    "samples_ms": samples,
                }
            )
            split = 4 if field == "mlp.Wo" else 1
            n = modules[0].weight.shape[0]
            for compressed in [True, False]:
                storage = [pack(m.weight, packed=compressed) for m in modules]
                configs = [
                    (tile, unroll)
                    for tile in range(4)
                    for unroll in ([0] if args.pipeline else [1, 4])
                ]
                random.Random(924).shuffle(configs)
                for tile, unroll in configs:
                    row = {
                        "field": field,
                        "variant": "compressed" if compressed else "raw-fragments",
                        "tile": tile,
                        "unroll": unroll,
                        "decode_exact_all_28": True,
                    }
                    try:

                        def run(
                            inputs=inputs,
                            storage=storage,
                            n=n,
                            split=split,
                            tile=tile,
                            unroll=unroll,
                            compressed=compressed,
                        ):
                            return [
                                matmul(x, w, n, split, tile, unroll, compressed)
                                for x, w in zip(inputs, storage)
                            ]

                        actual = run()
                        row["mismatches"] = sum(
                            int((a.view(torch.int16) != b.view(torch.int16)).sum())
                            for a, b in zip(actual, expected)
                        )
                        row["max_abs_error"] = max(
                            float((a.float() - b.float()).abs().max())
                            for a, b in zip(actual, expected)
                        )
                        row["ms"], row["samples_ms"] = timing(run, repeats=30, rounds=5)
                        row["speedup_retained"] = base / row["ms"]
                    except Exception as error:  # noqa: BLE001 - report failed probes
                        row["error"] = str(error)
                    report["rows"].append(row)
                    args.output.write_text(json.dumps(report, indent=2) + "\n")
                    print(json.dumps(row), flush=True)
                del storage
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
