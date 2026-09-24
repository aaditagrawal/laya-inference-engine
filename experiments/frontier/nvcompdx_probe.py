"""Bounded ANS decode-to-shared throughput screen on the real MLPWi bank."""

import argparse
import json
import random
import statistics
import time
from pathlib import Path

import torch

from experiments.native import common

from .engine import FrontierEngine
from .nvcompdx_ans import Bank, load, transform
from .nvcompdx_build import DIRECTORY, ROOT, SOURCE, sha
from .tune import timing


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path("results/frontier/nvcompdx-ans.json")
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    load()
    report = {
        "metadata": common.metadata(),
        "native_build": json.loads((DIRECTORY / "probe-build.json").read_text()),
        "source_sha256": {
            str(p.relative_to(ROOT)): sha(p)
            for p in [
                SOURCE,
                Path(__file__),
                ROOT / "experiments/frontier/nvcompdx_ans.py",
                ROOT / "experiments/frontier/nvcompdx_build.py",
            ]
        },
        "method": "Whole checkpoint MLPWi bank, nvCOMPDx ANS SM decoder with device LTO. Exact entire-bank reconstruction gate before timing. Decompression writes directly into shared memory, followed by uint32-to-uint64 checksum reduction. Matched raw cp.async-to-shared/checksum uses the same chunk/block and shared scratch allocation. Compression, compaction, transforms and global restoration are excluded from steady-state timing. Transform reversal inside GEMM would be additional work, so transformed checksum throughput is optimistic.",
        "rows": [],
    }
    with FrontierEngine(policy="bf16-splitk-exact", max_graphs=1) as engine:
        modules = [layer.mlp.Wi for layer in engine.base.model.net.encoder.layers]
        if args.smoke:
            modules = modules[:1]
        bank = torch.cat(
            [module.weight.view(torch.uint8).reshape(-1) for module in modules]
        )
        report["matrix_count"] = len(modules)
        report["original_bytes"] = bank.numel()
        formats = [
            ("raw_bytes", 0, False),
            ("byte_planes", 1, False),
            ("sign_mantissa_exponent_planes", 2, False),
            ("fp16_bits", 0, True),
        ]
        for name, mode, half in formats:
            start = time.perf_counter()
            encoded = transform(bank, mode)
            torch.cuda.synchronize()
            transform_ms = (time.perf_counter() - start) * 1000
            if not torch.equal(transform(encoded, mode, True), bank):
                raise RuntimeError("Pre-compression reversible transform failed")
            configs = [
                (chunk, block) for chunk in [4096, 16384] for block in [128, 256]
            ]
            random.Random(926).shuffle(configs)
            for chunk, block in configs:
                row = {
                    "format": name,
                    "transform": mode,
                    "ans_datatype": "float16_bit_reinterpretation" if half else "uint8",
                    "chunk_bytes": chunk,
                    "block_threads": block,
                    "transform_setup_ms": transform_ms,
                }
                compressed = Bank(encoded, chunk, block, half)
                row["resource_info"] = compressed.info
                row["setup"] = compressed.setup
                row["correctness"] = compressed.validate(bank, mode)
                row["timing_pairs"] = []
                for repeat in range(3 if args.smoke else 5):
                    pair = {}
                    order = [0, 1] if repeat % 2 == 0 else [1, 0]
                    for variant in order:
                        label = (
                            "raw_shared_checksum"
                            if variant == 0
                            else "ans_shared_checksum"
                        )
                        ms, samples = timing(
                            lambda variant=variant, compressed=compressed: (
                                compressed.run(variant)
                            ),
                            repeats=10 if args.smoke else 20,
                            rounds=3,
                        )
                        pair[label] = {"ms": ms, "samples_ms": samples}
                    row["timing_pairs"].append(pair)
                for label in ["raw_shared_checksum", "ans_shared_checksum"]:
                    ms = statistics.median(p[label]["ms"] for p in row["timing_pairs"])
                    row[label + "_ms"] = ms
                    row[label + "_uncompressed_gbs"] = bank.numel() / ms / 1e6
                row["speedup_matched_raw"] = (
                    row["raw_shared_checksum_ms"] / row["ans_shared_checksum_ms"]
                )
                row["paired_faster_rounds"] = sum(
                    p["ans_shared_checksum"]["ms"] < p["raw_shared_checksum"]["ms"]
                    for p in row["timing_pairs"]
                )
                report["rows"].append(row)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print(
                    json.dumps(
                        {
                            k: v
                            for k, v in row.items()
                            if k not in ["timing_pairs", "setup"]
                        }
                    ),
                    flush=True,
                )
                del compressed
        del bank
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
