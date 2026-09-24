"""Positive and negative controls for hardware VMM memory compression."""

import json
import random
from pathlib import Path

import torch
import triton as tr
import triton.language as tl

from experiments.native import common

from .compression import Allocation
from .tune import timing


@tr.jit
def _fill(X, N: tl.constexpr, MODE: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    v = i.to(tl.uint32) + 0x9E3779B9
    v = (v ^ (v >> 16)) * 0x85EBCA6B
    v = (v ^ (v >> 13)) * 0xC2B2AE35
    v = v ^ (v >> 16)
    if MODE == 0:
        v = tl.full((BLOCK,), 0, tl.uint32)
    elif MODE == 1:
        v = tl.full((BLOCK,), 0x3DCCCCCD, tl.uint32)
    elif MODE == 3:
        v = v & 65535
    elif MODE == 4:
        v = v << 16
    elif MODE == 5:
        v = tl.where(i % 4 == 0, v, 0)
    elif MODE == 6:
        v = tl.where(i % 2 == 0, v, 0)
    tl.store(X + i, v, i < N)


@tr.jit
def _checksum(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    v = tl.load(X + i, i < N, 0)
    tl.store(Y + tl.program_id(0), tl.sum(v, 0))


def checksum(x):
    result = torch.empty(tr.cdiv(x.numel(), 4096), device=x.device, dtype=torch.uint32)
    _checksum[(result.numel(),)](x, result, x.numel(), 4096)
    return result


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    report = {
        "metadata": common.metadata(),
        "logical_bytes": 256 * 1024 * 1024,
        "rows": [],
    }
    owners = {
        name: Allocation((64 * 1024 * 1024,), torch.uint32, compressed=compressed)
        for name, compressed in [("plain", False), ("compressed", True)]
    }
    try:
        for mode, label in enumerate(
            [
                "zero",
                "constant",
                "random32",
                "random-low16",
                "random-high16",
                "75-percent-zero",
                "50-percent-zero",
            ]
        ):
            for owner in owners.values():
                _fill[(tr.cdiv(owner.tensor.numel(), 1024),)](
                    owner.tensor, owner.tensor.numel(), mode, 1024
                )
            outputs = [checksum(owner.tensor) for owner in owners.values()]
            if not torch.equal(*outputs):
                raise RuntimeError("Pattern checksums differ")
            samples = {name: [] for name in owners}
            rng = random.Random(924)
            for _ in range(5):
                names = list(owners)
                rng.shuffle(names)
                for name in names:
                    owner = owners[name]
                    _, block = timing(
                        lambda owner=owner: checksum(owner.tensor), repeats=20, rounds=1
                    )
                    samples[name].extend(block)
            row = {
                "pattern": label,
                "samples_ms": samples,
                "p50_ms": {n: sorted(v)[2] for n, v in samples.items()},
            }
            row["speedup"] = row["p50_ms"]["plain"] / row["p50_ms"]["compressed"]
            row["allocations"] = {n: o.report() for n, o in owners.items()}
            report["rows"].append(row)
            print(json.dumps(row), flush=True)
            Path("results/frontier/compression-patterns.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
    finally:
        for owner in owners.values():
            owner.close()


if __name__ == "__main__":
    main()
