"""Lossless blocked BF16 weights: sign/mantissa bytes + exponent nibbles.

Each 64-weight block has a minimum exponent. Blocks whose exponent span exceeds
15 use an exact BF16 escape block. Every bit is checked during construction.
This compresses storage without quantization; GEMM reduction order is separate.
"""

from dataclasses import dataclass

import torch
import triton as tr
import triton.language as tl

from .matmul import _reduce


@dataclass
class PackedWeight:
    low: torch.Tensor
    high: torch.Tensor
    metadata: torch.Tensor
    escapes: torch.Tensor
    shape: tuple
    report: dict


@torch.inference_mode()
def pack(weight):
    n, k = weight.shape
    if weight.dtype != torch.bfloat16 or k % 64:
        raise ValueError("Expected BF16 weights with a multiple of 64 columns")
    bits = weight.contiguous().view(torch.int16).to(torch.int32) & 65535
    grouped = bits.view(n, k // 64, 64)
    exp = (grouped >> 7) & 255
    minimum, maximum = exp.amin(-1), exp.amax(-1)
    escape = (maximum - minimum > 15) | (minimum == 255)
    escape_index = escape.flatten().to(torch.int32).cumsum(0).view_as(escape) - 1
    metadata = torch.where(escape, (escape_index << 8) | 255, minimum).to(torch.int32)
    escapes = weight.view(n, k // 64, 64)[escape].contiguous()
    if not escapes.numel():
        escapes = torch.zeros(64, dtype=weight.dtype, device=weight.device)
    low = ((bits & 127) | ((bits >> 8) & 128)).to(torch.uint8)
    nibble = ((exp - minimum[..., None]) & 15).view(n, k).to(torch.uint8)
    high = (nibble[:, 0::2] | (nibble[:, 1::2] << 4)).contiguous()
    recovered = ((low.to(torch.int32) & 128) << 8) | (low.to(torch.int32) & 127)
    recovered |= ((nibble.view_as(exp).to(torch.int32) + minimum[..., None]) << 7).view(
        n, k
    )
    recovered.view_as(grouped)[escape] = grouped[escape]
    if not torch.equal(recovered, bits):
        raise AssertionError("Lossless weight packing changed a BF16 bit pattern")
    byte_count = sum(
        t.numel() * t.element_size() for t in (low, high, metadata, escapes)
    )
    report = {
        "original_bytes": weight.numel() * weight.element_size(),
        "packed_bytes": byte_count,
        "escape_blocks": int(escape.sum()),
        "blocks": escape.numel(),
        "exact_bits": True,
    }
    return PackedWeight(low, high, metadata, escapes, (n, k), report)


@tr.jit
def _packed_matmul(
    X,
    LOW,
    HIGH,
    META,
    ESCAPE,
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
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    part = tl.program_id(2)
    kk = tl.arange(0, BK)
    steps: tl.constexpr = tr.cdiv(tr.cdiv(K, BK), SPLIT)
    acc = tl.full((BM, BN), 0, tl.float32)
    for i in range(steps):
        k = (part * steps + i) * BK + kk
        valid = (n[None, :] < N) & (k[:, None] < K)
        low = tl.load(LOW + n[None, :] * K + k[:, None], valid, 0).to(tl.uint32)
        high = tl.load(HIGH + n[None, :] * (K // 2) + k[:, None] // 2, valid, 0).to(
            tl.uint32
        )
        meta = tl.load(META + n[None, :] * (K // 64) + k[:, None] // 64, valid, 0).to(
            tl.uint32
        )
        exponent = (meta & 255) + ((high >> ((k[:, None] & 1) * 4)) & 15)
        bits = ((low & 128) << 8) | (exponent << 7) | (low & 127)
        decoded = bits.to(tl.uint16).to(tl.bfloat16, bitcast=True)
        escaped = tl.load(
            ESCAPE + (meta >> 8) * 64 + k[:, None] % 64,
            valid & ((meta & 255) == 255),
            0,
        )
        w = tl.where((meta & 255) == 255, escaped, decoded)
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
    m = x.numel() // k
    y = torch.empty((*x.shape[:-1], n), device=x.device, dtype=torch.bfloat16)
    partial = (
        torch.empty((split, m, n), device=x.device, dtype=torch.float32)
        if split > 1
        else y
    )
    _packed_matmul[(tr.cdiv(m, bm), tr.cdiv(n, bn), split)](
        x,
        weight.low,
        weight.high,
        weight.metadata,
        weight.escapes,
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
