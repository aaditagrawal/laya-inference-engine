"""Benchmark exact Turbo-Lossless TMA adaptations on all 28 actual layer weights."""

import argparse
import hashlib
import itertools
import json
import random
from pathlib import Path

import torch

from experiments.native import common

from .engine import FrontierEngine
from .tune import timing
from .turbo_build import DIRECTORY, REVISION, SOURCES
from .turbo_lossless import Operation, load, pack


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fields", nargs="+", default=["attn.Wqkv", "attn.Wo", "mlp.Wi", "mlp.Wo"]
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/frontier/turbo-lossless.json")
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(20260925)
    load()
    # Exercise every possible BF16 bit pattern, including signed zero,
    # subnormals, infinities and NaN payloads, through CPU and GPU decoders.
    exhaustive = (
        torch.arange(65536, dtype=torch.int32, device="cuda")
        .to(torch.int16)
        .view(torch.bfloat16)
        .reshape(64, 1024)
    )
    packed_exhaustive = pack(exhaustive)
    report = {
        "metadata": common.metadata(),
        "seed": 20260925,
        "upstream_revision": REVISION,
        "upstream_url": "https://github.com/cenconq25/Turbo-Lossless",
        "source_sha256": {
            s.name: hashlib.sha256(s.read_bytes()).hexdigest() for s in SOURCES
        },
        "native_build": json.loads((DIRECTORY / "build.json").read_text()),
        "all_65536_bf16_patterns_decoded_exactly": True,
        "exhaustive_patterns_escape_count": packed_exhaustive.patches,
        "method": "28 distinct real matrices and independent random BF16 activations. Exact sparse escape replacement before MMA. CUDA graph replay. MLPWo preserves split-4 BF16 partial rounding. Unsafe skip-escape controls are measured separately and cannot be retained.",
        "rows": [],
    }
    del exhaustive, packed_exhaustive
    with FrontierEngine(policy="bf16-splitk-exact", max_graphs=1) as engine:
        report["baseline_selection"] = engine.selection
        for field in args.fields:
            group, attr = field.split(".")
            modules = [
                getattr(getattr(layer, group), attr)
                for layer in engine.base.model.net.encoder.layers
            ]
            if args.smoke:
                modules = modules[:1]
            inputs = [
                torch.randn(
                    1, 64, m.weight.shape[1], device="cuda", dtype=torch.bfloat16
                )
                for m in modules
            ]
            expected = [m(x) for m, x in zip(modules, inputs)]

            def baseline(modules=modules, inputs=inputs):
                return [m(x) for m, x in zip(modules, inputs)]

            base, samples = timing(baseline, repeats=30, rounds=5)
            report["rows"].append(
                {
                    "field": field,
                    "variant": "retained-bf16",
                    "ms": base,
                    "samples_ms": samples,
                }
            )
            storage = [pack(m.weight) for m in modules]
            report.setdefault("packing", {})[field] = {
                "matrices": len(modules),
                "all_decoded_exactly": True,
                "escape_count": sum(p.patches for p in storage),
                "storage_bytes": sum(p.storage_bytes for p in storage),
                "original_bytes": sum(m.weight.numel() * 2 for m in modules),
                "base_exponents": [p.base for p in storage],
            }
            configs = list(itertools.product([16, 32, 64], [32, 64], [True, False]))
            random.Random(20260925).shuffle(configs)
            for tn, tm, exact in configs:
                row = {
                    "field": field,
                    "variant": "exact-escapes" if exact else "unsafe-no-escapes",
                    "token_tile": tn,
                    "weight_tile": tm,
                }
                operations = []
                try:
                    operations = [
                        Operation(x, p, tn, tm, 4 if field == "mlp.Wo" else 1, exact)
                        for x, p in zip(inputs, storage)
                    ]

                    def run(operations=operations):
                        return [op() for op in operations]

                    actual = run()
                    row["mismatches"] = sum(
                        int((a.view(torch.int16) != e.view(torch.int16)).sum())
                        for a, e in zip(actual, expected)
                    )
                    row["max_abs_error"] = max(
                        float((a.float() - e.float()).abs().max())
                        for a, e in zip(actual, expected)
                    )
                    row["ms"], row["samples_ms"] = timing(run, repeats=30, rounds=5)
                    row["speedup_retained"] = base / row["ms"]
                except Exception as error:  # noqa: BLE001 - preserve failed experiments
                    row["error"] = str(error)
                finally:
                    for op in operations:
                        op.close()
                report["rows"].append(row)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps(row), flush=True)
            del storage
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
