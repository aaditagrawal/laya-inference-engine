"""Lossless BF16 tiles with one uniform header and an exact raw fallback.

The common tile stores each low byte plus four bits for sign and upper exponent.
Its unpack path has no per-weight metadata or sparse escape gather. A tile with
an exponent span wider than eight upper-byte values stores raw BF16 instead.
Slots retain the original allocation size; only bytes fetched are compressed.
"""

from dataclasses import dataclass

import torch
import triton as tr
import triton.language as tl

from .matmul import _reduce


@dataclass
class PackedWeight:
    data: torch.Tensor
    header: torch.Tensor
    shape: tuple
    tile: tuple
    report: dict


@torch.inference_mode()
def pack(weight, bn=32, bk=64):
    n, k = weight.shape
    if weight.dtype != torch.bfloat16 or n % bn or k % bk:
        raise ValueError(
            "This layout requires BF16 and dimensions divisible by the tile"
        )
    tiled = (
        weight.contiguous()
        .view(n // bn, bn, k // bk, bk)
        .permute(0, 2, 1, 3)
        .contiguous()
    )
    bits = tiled.view(torch.int16).to(torch.int32).reshape(-1, bn * bk) & 65535
    upper = (bits >> 8) & 127
    base, maximum = upper.amin(1), upper.amax(1)
    raw = maximum - base > 7
    low = (bits & 255).to(torch.uint8)
    nibble = (((upper - base[:, None]) & 7) | ((bits >> 12) & 8)).to(torch.uint8)
    high = nibble[:, 0::2] | (nibble[:, 1::2] << 4)
    data = torch.empty(
        (bits.shape[0], 2 * bn * bk), device=weight.device, dtype=torch.uint8
    )
    data[:, : bn * bk] = low
    data[:, bn * bk : 3 * bn * bk // 2] = high
    data[raw] = tiled.view(torch.uint8).reshape(-1, 2 * bn * bk)[raw]
    header = (base | (raw.to(torch.int32) << 8)).to(torch.int16)
    reconstructed = (
        low.to(torch.int32)
        | (((nibble.to(torch.int32) & 7) + base[:, None]) << 8)
        | ((nibble.to(torch.int32) & 8) << 12)
    )
    reconstructed[raw] = bits[raw]
    if not torch.equal(reconstructed, bits):
        raise AssertionError("Lossless tile packing changed a weight bit")
    normal_count = int((~raw).sum())
    raw_count = int(raw.sum())
    return PackedWeight(
        data,
        header,
        (n, k),
        (bn, bk),
        {
            "original_bytes": n * k * 2,
            "allocated_bytes": data.numel() + header.numel() * 2,
            "encoded_bytes_read_once": normal_count * bn * bk * 3 // 2
            + raw_count * bn * bk * 2
            + header.numel() * 2,
            "compressed_tiles": normal_count,
            "raw_tiles": raw_count,
            "exact_bits": True,
        },
    )


@tr.jit
def _matmul(
    X,
    W,
    HEADER,
    PARTIAL,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SPLIT: tl.constexpr,
):
    block_n = tl.program_id(1)
    part = tl.program_id(2)
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    nk = tl.arange(0, BN)
    n = block_n * BN + nk
    kk = tl.arange(0, BK)
    steps: tl.constexpr = ((K + BK - 1) // BK + SPLIT - 1) // SPLIT
    acc = tl.full((BM, BN), 0, tl.float32)
    for step in range(steps):
        block_k = part * steps + step
        k = block_k * BK + kk
        tile = block_n * (K // BK) + block_k
        header = tl.load(HEADER + tile, block_k < K // BK, 256).to(tl.uint32)
        base = W + tile * (2 * BN * BK)
        index = nk[None, :] * BK + kk[:, None]
        if header & 256:
            raw_pointer = base.to(tl.pointer_type(tl.bfloat16))
            w = tl.load(raw_pointer + index, block_k < K // BK, 0.0)
        else:
            low = tl.load(base + index).to(tl.uint32)
            high = tl.load(base + BN * BK + index // 2).to(tl.uint32)
            nibble = (high >> ((kk[:, None] & 1) * 4)) & 15
            bits = low | (((nibble & 7) + header) << 8) | ((nibble & 8) << 12)
            w = bits.to(tl.uint16).to(tl.bfloat16, bitcast=True)
        a = tl.load(
            X + m[:, None] * K + k[None, :], (m[:, None] < M) & (k[None, :] < K), 0
        )
        acc = tl.dot(a, w, acc)
    offset = m[:, None] * N + n[None, :]
    valid = (m[:, None] < M) & (n[None, :] < N)
    if SPLIT == 1:
        tl.store(Y + offset, acc, valid)
    else:
        tl.store(PARTIAL + part * M * N + offset, acc, valid)


def matmul(x, weight, config):
    bm, bn, bk, split, warps, stages = config
    n, k = weight.shape
    if (bn, bk) != weight.tile:
        raise ValueError("Packed tile and launch configuration differ")
    m = x.numel() // k
    y = torch.empty((*x.shape[:-1], n), device=x.device, dtype=torch.bfloat16)
    partial = (
        torch.empty((split, m, n), device=x.device, dtype=torch.float32)
        if split > 1
        else y
    )
    _matmul[(tr.cdiv(m, bm), n // bn, split)](
        x,
        weight.data,
        weight.header,
        partial,
        y,
        m,
        n,
        k,
        bm,
        bn,
        bk,
        split,
        num_warps=warps,
        num_stages=stages,
    )
    if split > 1:
        _reduce[(tr.cdiv(m * n, 512),)](partial, y, m * n, split, 512)
    return y
