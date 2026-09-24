"""Lossless fixed-exponent BF16 encoding with sparse original-weight escapes.

Common weights use 12 bits with upper-byte magnitude 55..62. Code zero is
reserved for escapes, including signed zero and out-of-range exponents. Dense
original BF16 storage is retained, but fetched only at escaped positions.
"""

from dataclasses import dataclass

import torch
import triton as tr
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from .matmul import _reduce


@dataclass
class PackedWeight:
    low: torch.Tensor
    high: torch.Tensor
    original: torch.Tensor
    escapes: int


@torch.inference_mode()
def pack(weight):
    if weight.dtype != torch.bfloat16 or weight.shape[1] % 64:
        raise ValueError("Expected BF16 matrix with K divisible by 64")
    bits = weight.contiguous().view(torch.int16).to(torch.int32) & 65535
    upper = (bits >> 8) & 127
    escape = (upper < 55) | (upper > 62) | (bits == 0x3700)
    low = torch.where(escape, 0, bits & 255)
    high = torch.where(escape, 0, (upper - 55) | ((bits >> 12) & 8))
    restored = low | (((high & 7) + 55) << 8) | ((high & 8) << 12)
    restored[escape] = bits[escape]
    if not torch.equal(restored, bits):
        raise RuntimeError("Lossless encoding changed a weight bit")
    low = (low[:, ::2] | (low[:, 1::2] << 8)).to(torch.uint16).contiguous()
    high = (high[:, ::2] | (high[:, 1::2] << 4)).to(torch.uint8).contiguous()
    return PackedWeight(low, high, weight, int(escape.sum()))


@tr.jit
def _decode(low, high):
    lo = tl.join(low & 255, low >> 8).reshape((low.shape[0], low.shape[1] * 2))
    hi = tl.join(high & 15, high >> 4).reshape((high.shape[0], high.shape[1] * 2))
    escaped = (lo | hi) == 0
    bits = lo | (((hi & 7) + 55) << 8) | ((hi & 8) << 12)
    return bits.to(tl.uint16).to(tl.bfloat16, bitcast=True), escaped


@tr.jit
def _unpack(LOW, HIGH, RAW, Y, TOTAL: tl.constexpr):
    pairs = tl.program_id(0) * 128 + tl.arange(0, 128)
    low = tl.load(LOW + pairs, pairs < TOTAL // 2, 0).to(tl.uint32)[None, :]
    high = tl.load(HIGH + pairs, pairs < TOTAL // 2, 0).to(tl.uint32)[None, :]
    value, escaped = _decode(low, high)
    i = tl.program_id(0) * 256 + tl.arange(0, 256)
    raw = tl.load(RAW + i, (i < TOTAL) & escaped.reshape((256,)), 0)
    tl.store(
        Y + i, tl.where(escaped.reshape((256,)), raw, value.reshape((256,))), i < TOTAL
    )


def check_decode(weight):
    actual = torch.empty_like(weight.original)
    _unpack[(tr.cdiv(actual.numel(), 256),)](
        weight.low, weight.high, weight.original, actual, actual.numel()
    )
    return int((actual.view(torch.int16) != weight.original.view(torch.int16)).sum())


@tr.jit
def _gemm(
    X,
    LOW,
    HIGH,
    RAW,
    P,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SPLIT: tl.constexpr,
    TMA: tl.constexpr,
):
    m0, n0, part = tl.program_id(0) * BM, tl.program_id(1) * BN, tl.program_id(2)
    m, n = m0 + tl.arange(0, BM), n0 + tl.arange(0, BN)
    kc, kp = tl.arange(0, BK), tl.arange(0, BK // 2)
    steps: tl.constexpr = tl.cdiv(tl.cdiv(K, BK), SPLIT)
    acc = tl.full((BM, BN), 0, tl.float32)
    for step in range(steps):
        k0 = (part * steps + step) * BK
        if TMA:
            a = X.load([m0, k0])
            low = LOW.load([n0, k0 // 2]).to(tl.uint32)
            high = HIGH.load([n0, k0 // 2]).to(tl.uint32)
        else:
            a = tl.load(
                X + m[:, None] * K + k0 + kc[None, :],
                (m[:, None] < M) & (k0 + kc[None, :] < K),
                0,
            )
            offset = n[:, None] * (K // 2) + k0 // 2 + kp[None, :]
            valid = (n[:, None] < N) & (k0 // 2 + kp[None, :] < K // 2)
            low = tl.load(LOW + offset, valid, 0).to(tl.uint32)
            high = tl.load(HIGH + offset, valid, 0).to(tl.uint32)
        decoded, escaped = _decode(low, high)
        raw = tl.load(
            RAW + n[:, None] * K + k0 + kc[None, :],
            escaped & (n[:, None] < N) & (k0 + kc[None, :] < K),
            0,
        )
        w = tl.where(escaped, raw, decoded)
        acc = tl.dot(a, w.T, acc)
    offset = m[:, None] * N + n[None, :]
    valid = (m[:, None] < M) & (n[None, :] < N)
    if SPLIT == 1:
        tl.store(Y + offset, acc, valid)
    else:
        tl.store(P + part * M * N + offset, acc, valid)


def matmul(x, weight, config):
    bm, bn, bk, split, warps, stages, tma = config
    n, k = weight.original.shape
    m = x.numel() // k
    if not x.is_contiguous() or x.shape[-1] != k:
        raise ValueError("Expected matching contiguous input")
    if tma:
        a = TensorDescriptor(x, [m, k], [k, 1], [bm, bk])
        low = TensorDescriptor(weight.low, [n, k // 2], [k // 2, 1], [bn, bk // 2])
        high = TensorDescriptor(weight.high, [n, k // 2], [k // 2, 1], [bn, bk // 2])
    else:
        a, low, high = x, weight.low, weight.high
    y = torch.empty((*x.shape[:-1], n), device=x.device, dtype=torch.bfloat16)
    partial = (
        torch.empty((split, m, n), device=x.device, dtype=torch.bfloat16)
        if split > 1
        else y
    )
    _gemm[(tr.cdiv(m, bm), tr.cdiv(n, bn), split)](
        a,
        low,
        high,
        weight.original,
        partial,
        y,
        m,
        n,
        k,
        bm,
        bn,
        bk,
        split,
        tma,
        num_warps=warps,
        num_stages=stages,
    )
    if split > 1:
        _reduce[(tr.cdiv(m * n, 512),)](partial, y, m * n, split, 512)
    return y
