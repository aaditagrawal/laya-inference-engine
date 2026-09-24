"""Screen true producer/consumer lossless GEMMs with matched native controls."""

import argparse
import hashlib
import itertools
import json
import random
from pathlib import Path

import torch

from experiments.native import common
from experiments.native.kernels.candidates import triton_geglu_corrected

from .engine import FrontierEngine
from .mlp_geglu import project
from .tune import timing
from .ws_lossless import TILES, Operation, load, pack
from .ws_lossless_build import DIRECTORY, ROOT, SOURCE


def exact(actual, expected):
    mismatches = [
        int((a.view(torch.int16) != b.view(torch.int16)).sum())
        for a, b in zip(actual, expected)
    ]
    return {
        "mismatches": sum(mismatches),
        "per_matrix_mismatches": mismatches,
        "max_abs_error": max(
            float((a.float() - b.float()).abs().max()) for a, b in zip(actual, expected)
        ),
    }


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--shared-copy", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path("results/frontier/ws-lossless.json")
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(20260926)
    load()
    report = {
        "metadata": common.metadata(),
        "seed": 20260926,
        "native_build": json.loads((DIRECTORY / "build.json").read_text()),
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [
                SOURCE,
                Path(__file__),
                ROOT / "experiments/frontier/ws_lossless.py",
                ROOT / "experiments/frontier/ws_lossless_build.py",
                ROOT / "experiments/frontier/native_lossless.py",
            ]
        },
        "method": "One/two dedicated producer warps reconstruct fixed 13-bit BF16 weights into 2/3 buffered shared tiles; four consumer warps perform sequential BF16 MMA without decoding. Per-slot release/acquire full/empty mbarriers. Matched unpacked native controls use identical scheduling and consumer code. Timing is 28 distinct real matrices, graph replay, not full requests.",
        "storage_ratio": 13 / 16,
        "rows": [],
    }
    # Exhaustive accepted finite normal encoding range, including both zeros.
    codes = torch.arange(8192, device="cuda", dtype=torch.int32)
    exponent, mantissa, sign = (codes >> 7) & 31, codes & 127, (codes >> 12) << 15
    bits = (
        sign
        | torch.where(exponent == 0, 0, (exponent + 102) << 7)
        | torch.where(exponent == 0, 0, mantissa)
    )
    domain = bits.to(torch.int16).view(torch.bfloat16).reshape(8, 1024)
    pack(domain, True)
    pack(domain, False)
    report["accepted_domain_native_decode_exact"] = True
    report["accepted_unique_patterns"] = 7938
    fused_report = json.loads(
        (ROOT / "results/frontier/mlp-geglu-unpacked.json").read_text()
    )
    fused_config = max(
        [
            r
            for r in fused_report["rows"]
            if r.get("mismatches") == 0 and r.get("speedup", 0) > 1.03
        ],
        key=lambda r: r["speedup"],
    )["config"]
    report["fused_config"] = fused_config
    rounds, repeats = (3, 10) if args.smoke else (5, 30)
    with FrontierEngine(policy="bf16-splitk-exact", max_graphs=1) as engine:
        modules = [layer.mlp.Wi for layer in engine.base.model.net.encoder.layers]
        if args.smoke:
            modules = modules[:1]
        report["matrix_count"] = len(modules)
        report["retained_selection"] = engine.selection
        inputs = [
            torch.randn(1, 64, 1024, device="cuda", dtype=torch.bfloat16)
            for _ in modules
        ]

        def raw_baseline():
            return [module(x) for x, module in zip(inputs, modules)]

        def fused_baseline():
            return [
                project(x, module.weight, fused_config)
                for x, module in zip(inputs, modules)
            ]

        expected = raw_baseline()
        expected_geglu = [triton_geglu_corrected(y) for y in expected]
        report["fused_reference_parity"] = exact(fused_baseline(), expected_geglu)
        if report["fused_reference_parity"]["mismatches"]:
            raise RuntimeError("Retained fused baseline mismatch")
        base, samples = timing(raw_baseline, repeats=repeats, rounds=rounds)
        fused, fsamples = timing(fused_baseline, repeats=repeats, rounds=rounds)
        report["baseline_raw"] = {"ms": base, "samples_ms": samples}
        report["baseline_fused_geglu"] = {"ms": fused, "samples_ms": fsamples}
        storage = {
            packed: [pack(module.weight, packed) for module in modules]
            for packed in [False, True]
        }
        report["all_weights_native_decode_exact"] = True
        configs = list(itertools.product(TILES, [1, 2], [2, 3], [False, True]))
        random.Random(20260926).shuffle(configs)
        for tile, producers, stages, packed in configs:
            row = {
                "tile": tile,
                "bm": TILES[tile][0],
                "bn": TILES[tile][1],
                "producers": producers,
                "consumers": 4,
                "stages": stages,
                "packed": packed,
            }
            row["producer_shared_copy"] = args.shared_copy and packed
            operations = [
                Operation(
                    x, w, 5248, 1024, tile, producers, stages, packed, args.shared_copy
                )
                for x, w in zip(inputs, storage[packed])
            ]

            def run(operations=operations):
                return [op() for op in operations]

            def with_geglu(operations=operations):
                return [triton_geglu_corrected(op()) for op in operations]

            actual = run()
            row.update(exact(actual, expected))
            if row["mismatches"]:
                raise RuntimeError(f"Exact GEMM parity failed: {row}")
            row["geglu_parity"] = exact(with_geglu(), expected_geglu)
            if row["geglu_parity"]["mismatches"]:
                raise RuntimeError(f"Exact GEGLU parity failed: {row}")
            row["ms"], row["samples_ms"] = timing(run, repeats=repeats, rounds=rounds)
            row["with_geglu_ms"], row["with_geglu_samples_ms"] = timing(
                with_geglu, repeats=repeats, rounds=rounds
            )
            row["speedup_raw"] = base / row["ms"]
            row["speedup_fused_geglu"] = fused / row["with_geglu_ms"]
            report["rows"].append(row)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(row), flush=True)
        report["baseline_raw_after_ms"], _ = timing(
            raw_baseline, repeats=repeats, rounds=rounds
        )
        report["baseline_fused_after_ms"], _ = timing(
            fused_baseline, repeats=repeats, rounds=rounds
        )
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
