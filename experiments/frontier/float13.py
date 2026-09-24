"""Lossless 13-bit storage for this checkpoint's finite BF16 weight range.

Sign and seven mantissa bits are preserved. Five exponent bits encode zero or
IEEE exponents 103..133. Values outside that range are rejected before packing.
This is an experiment-specific storage format, not arithmetic quantization.
"""

import torch
import triton as tr
import triton.language as tl

from .matmul import _reduce


@tr.jit
def _encode(bits):
    exponent = (bits >> 7) & 255
    return (
        (bits & 127)
        | (tl.where(exponent == 0, 0, exponent - 102) << 7)
        | ((bits >> 15) << 12)
    )


@tr.jit
def _pack(X, P, TOTAL: tl.constexpr, WORDS: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    first = (i * 32) // 13
    shift = (i * 32) % 13
    a = _encode(tl.load(X + first, first < TOTAL, 0).to(tl.uint32))
    b = _encode(tl.load(X + first + 1, first + 1 < TOTAL, 0).to(tl.uint32))
    c = _encode(tl.load(X + first + 2, first + 2 < TOTAL, 0).to(tl.uint32))
    d = _encode(tl.load(X + first + 3, first + 3 < TOTAL, 0).to(tl.uint32))
    word = (
        (a >> shift)
        | (b << (13 - shift))
        | (c << (26 - shift))
        | tl.where(shift >= 8, d << (39 - shift), 0)
    )
    tl.store(P + i, word, i < WORDS)


@tr.jit
def _decode(P, row, k, K: tl.constexpr):
    word = (k * 13) // 32
    shift = (k * 13) % 32
    lo = tl.load(P + row * (K * 13 // 32) + word, k < K, 0)
    hi = tl.load(
        P + row * (K * 13 // 32) + word + 1,
        (k < K) & (shift > 19) & (word + 1 < K * 13 // 32),
        0,
    )
    code = ((lo >> shift) | tl.where(shift > 19, hi << (32 - shift), 0)) & 8191
    e = (code >> 7) & 31
    bits = (code & 127) | (tl.where(e == 0, 0, e + 102) << 7) | ((code >> 12) << 15)
    return bits.to(tl.uint16).to(tl.bfloat16, bitcast=True)


@tr.jit
def _unpack(P, Y, N: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, k = i // K, i % K
    value = _decode(P, tl.minimum(row, N - 1), k, K)
    tl.store(Y + i, value, i < N * K)


def pack(weight):
    if (
        weight.dtype != torch.bfloat16
        or not weight.is_contiguous()
        or weight.shape[1] % 32
    ):
        raise ValueError("Expected BF16 matrix with K divisible by 32")
    bits = weight.view(torch.int16).to(torch.int32) & 65535
    e = (bits >> 7) & 255
    valid = ((e >= 103) & (e <= 133)) | ((e == 0) & ((bits & 127) == 0))
    if not bool(valid.all()):
        raise ValueError("Weight contains an unsupported exponent or subnormal")
    n, k = weight.shape
    packed = torch.empty((n, k * 13 // 32), device=weight.device, dtype=torch.uint32)
    _pack[(tr.cdiv(packed.numel(), 256),)](
        weight.view(torch.uint16), packed, weight.numel(), packed.numel(), 256
    )
    decoded = torch.empty_like(weight)
    _unpack[(tr.cdiv(weight.numel(), 256),)](packed, decoded, n, k, 256)
    if not torch.equal(decoded.view(torch.int16), weight.view(torch.int16)):
        raise RuntimeError("Lossless packing check failed")
    return packed


@tr.jit
def _decode_tile(
    P, n, start, K: tl.constexpr, N: tl.constexpr, BK: tl.constexpr, BN: tl.constexpr
):
    PW: tl.constexpr = 32 if BK == 64 else 64
    packed_col = tl.arange(0, PW)
    first = start * 13 // 32
    packed = tl.load(
        P + n[:, None] * (K * 13 // 32) + first + packed_col[None, :],
        (n[:, None] < N)
        & (packed_col[None, :] < BK * 13 // 32)
        & (first + packed_col[None, :] < K * 13 // 32),
        0,
    )
    k = tl.arange(0, BK)
    index = k * 13 // 32
    shift = k * 13 % 32
    lo = tl.gather(packed, tl.broadcast_to(index[None, :], (BN, BK)), 1)
    hi = tl.gather(packed, tl.broadcast_to((index + 1)[None, :], (BN, BK)), 1)
    code = (
        (lo >> shift[None, :])
        | tl.where(shift[None, :] > 19, hi << (32 - shift[None, :]), 0)
    ) & 8191
    e = (code >> 7) & 31
    bits = (code & 127) | (tl.where(e == 0, 0, e + 102) << 7) | ((code >> 12) << 15)
    return bits.to(tl.uint16).to(tl.bfloat16, bitcast=True).T


@tr.jit
def _matmul(
    X,
    W,
    PARTIAL,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SPLIT: tl.constexpr,
    TILED: tl.constexpr,
):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    steps: tl.constexpr = tl.cdiv(tl.cdiv(K, BK), SPLIT)
    acc = tl.full((BM, BN), 0, tl.float32)
    for i in range(steps):
        k = (tl.program_id(2) * steps + i) * BK + kk
        a = tl.load(
            X + m[:, None] * K + k[None, :], (m[:, None] < M) & (k[None, :] < K), 0
        )
        # N is an exact multiple of these tiles in this checkpoint.
        if TILED:
            w = _decode_tile(W, n, (tl.program_id(2) * steps + i) * BK, K, N, BK, BN)
        else:
            w = _decode(W, n[None, :], k[:, None], K)
        acc = tl.dot(a, w, acc)
    offset = m[:, None] * N + n[None, :]
    mask = (m[:, None] < M) & (n[None, :] < N)
    if SPLIT == 1:
        tl.store(Y + offset, acc, mask)
    else:
        tl.store(PARTIAL + tl.program_id(2) * M * N + offset, acc, mask)


def matmul(x, weight, config, tiled=False):
    bm, bn, bk, split, warps, stages = config
    n, words = weight.shape
    k = words * 32 // 13
    if n % bn:
        raise ValueError("This probe requires N to divide the tile")
    m = x.numel() // k
    y = torch.empty((*x.shape[:-1], n), device=x.device, dtype=torch.bfloat16)
    partial = (
        torch.empty((split, m, n), device=x.device, dtype=torch.bfloat16)
        if split > 1
        else y
    )
    _matmul[(tr.cdiv(m, bm), tr.cdiv(n, bn), split)](
        x,
        weight,
        partial,
        y,
        m,
        n,
        k,
        bm,
        bn,
        bk,
        split,
        tiled,
        num_warps=warps,
        num_stages=stages,
    )
    if split > 1:
        _reduce[(tr.cdiv(m * n, 512),)](partial, y, m * n, split, 512)
    return y
