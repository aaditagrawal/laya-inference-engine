"""Screen exact weight transforms for CUDA's transparent memory compression.

These are streaming-read probes, not GEMM or complete-inference measurements.
No physical compression ratio is inferred from logical allocation sizes.
"""

import hashlib
import json
import random
import statistics
from pathlib import Path

import torch
import triton as tr
import triton.language as tl

from experiments.latency.engine import V2Engine
from experiments.native import common

from .compression import Allocation
from .compression_patterns import checksum
from .tune import timing


@tr.jit
def _planes(X, Y, WORDS: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    lane = tl.arange(0, 32)
    bits = tl.load(X + i[:, None] * 32 + lane[None, :], i[:, None] < WORDS, 0).to(
        tl.uint32
    )
    plane = tl.program_id(1)
    word = tl.sum(((bits >> plane) & 1) << lane[None, :], 1)
    tl.store(Y + plane * WORDS + i, word, i < WORDS)


@tr.jit
def _unplanes(X, Y, TOTAL: tl.constexpr, WORDS: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    bits = tl.full((BLOCK,), 0, tl.uint32)
    for plane in tl.static_range(16):
        word = tl.load(X + plane * WORDS + i // 32, i < TOTAL, 0)
        bits |= ((word >> (i % 32)) & 1) << plane
    tl.store(Y + i, bits, i < TOTAL)


@tr.jit
def _random_codes(X, N: tl.constexpr, BITS: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    v = i.to(tl.uint32) + 0x9E3779B9
    v = (v ^ (v >> 16)) * 0x85EBCA6B
    v = (v ^ (v >> 13)) * 0xC2B2AE35
    v ^= v >> 16
    tl.store(X + i, v & ((1 << BITS) - 1), i < N)


def encoded_bank(bits, mode):
    if mode == "raw-bf16":
        return bits, bits
    exponent = (bits.to(torch.int32) >> 7) & 255
    delta = exponent - 119
    z = (delta << 1) ^ (delta >> 31)
    if not bool((z < 256).all()):
        raise ValueError("Exponent is outside the reversible byte transform range")
    code = (
        (bits.to(torch.int32) & 127) | ((bits.to(torch.int32) >> 15) << 7) | (z << 8)
    ).to(torch.uint16)
    if mode == "zigzag-u16":
        encoded, decoded = code, code
    elif mode == "zigzag-u32":
        encoded = code.to(torch.uint32)
        decoded = encoded.to(torch.uint16)
    elif mode in {"zigzag-bytes", "zigzag-nibbles"}:
        width = 8 if mode == "zigzag-bytes" else 4
        encoded = torch.stack(
            [
                ((code.to(torch.int32) >> shift) & ((1 << width) - 1)).to(torch.uint8)
                for shift in range(0, 16, width)
            ]
        )
        decoded = torch.zeros_like(code, dtype=torch.int32)
        for part, shift in enumerate(range(0, 16, width)):
            decoded |= encoded[part].to(torch.int32) << shift
        decoded = decoded.to(torch.uint16)
    elif mode == "zigzag-bitplanes":
        words = tr.cdiv(code.numel(), 32)
        if words * 32 != code.numel():
            raise ValueError("Expected a multiple of 32 weight elements")
        encoded = torch.empty((16, words), dtype=torch.uint32, device=bits.device)
        _planes[(tr.cdiv(words, 32), 16)](code, encoded, words, 32)
        decoded = torch.empty_like(code)
        _unplanes[(tr.cdiv(code.numel(), 256),)](
            encoded, decoded, code.numel(), words, 256
        )
    else:
        raise ValueError(mode)
    recovered = decoded.to(torch.int32)
    dz = recovered >> 8
    de = 119 + ((dz >> 1) ^ -(dz & 1))
    recovered = ((recovered & 127) | (((recovered >> 7) & 1) << 15) | (de << 7)).to(
        torch.uint16
    )
    return encoded.contiguous(), recovered


def measure(encoded, label, metadata):
    owners = {}
    try:
        for name, compressed in [("plain", False), ("compressed", True)]:
            owner = Allocation(encoded.shape, encoded.dtype, compressed=compressed)
            owners[name] = owner
            owner.copy(encoded)
            if not torch.equal(owner.tensor, encoded):
                raise RuntimeError("VMM copy changed transformed storage")
        expected = checksum(encoded.view(torch.uint32).flatten())
        for owner in owners.values():
            if not torch.equal(
                checksum(owner.tensor.view(torch.uint32).flatten()), expected
            ):
                raise RuntimeError("Streaming checksums differ")
        samples = {name: [] for name in owners}
        order = []
        rng = random.Random(97813)
        for _ in range(7):
            names = list(owners)
            rng.shuffle(names)
            order.append(names)
            for name in names:
                owner = owners[name]
                _, values = timing(
                    lambda owner=owner: checksum(
                        owner.tensor.view(torch.uint32).flatten()
                    ),
                    repeats=30,
                    rounds=1,
                )
                samples[name].extend(values)
        medians = {name: statistics.median(values) for name, values in samples.items()}
        return {
            "format": label,
            **metadata,
            "logical_bytes": encoded.nbytes,
            "storage_dtype": str(encoded.dtype),
            "allocations": {name: owner.report() for name, owner in owners.items()},
            "samples_ms": samples,
            "order": order,
            "median_ms": medians,
            "compression_read_speedup": medians["plain"] / medians["compressed"],
        }
    finally:
        for owner in owners.values():
            owner.close()


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    path = Path("results/frontier/compression-transforms.json")
    report = {
        "metadata": common.metadata(),
        "scope": "Streaming checksums over banks larger than L2, not matrix/inference timings; compression ratios are not measured",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "rows": [],
    }

    def record(row):
        report["rows"].append(row)
        path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(row), flush=True)

    # Controls determine whether merely reducing integer range can help.
    control = torch.empty(32 * 1024 * 1024, device="cuda", dtype=torch.uint32)
    for width in [0, 1, 4, 8, 12, 16, 32]:
        _random_codes[(tr.cdiv(control.numel(), 1024),)](
            control, control.numel(), width, 1024
        )
        record(
            measure(control, f"uniform-low-{width}-bits", {"kind": "synthetic-control"})
        )
    del control
    with V2Engine(optimization="native", max_graphs=1) as engine:
        bits = (
            torch.stack(
                [layer.mlp.Wi.weight for layer in engine.base.model.net.encoder.layers]
            )
            .view(torch.uint16)
            .flatten()
        )
        for mode in [
            "raw-bf16",
            "zigzag-u16",
            "zigzag-u32",
            "zigzag-bytes",
            "zigzag-nibbles",
            "zigzag-bitplanes",
        ]:
            encoded, restored = encoded_bank(bits, mode)
            mismatches = int((restored != bits).sum())
            if mismatches:
                raise RuntimeError(f"Weight transform failed: {mode}: {mismatches}")
            record(
                measure(
                    encoded,
                    mode,
                    {
                        "kind": "all-28-mlp-input-weights",
                        "weight_mismatches": mismatches,
                        "original_weight_bytes": bits.nbytes,
                    },
                )
            )


if __name__ == "__main__":
    main()
