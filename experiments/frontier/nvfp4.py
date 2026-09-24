"""Native SM120 FP4 with E4M3 block scales and FP32 per-row outer scales.

This changes model numerics and is an experiment, never a default. The outer
scale is per row rather than NVFP4's common per-tensor scale; the native MMA
still consumes E2M1 values and 16-element E4M3 block scales.
"""

import torch
import triton as tr
import triton.language as tl

from .matmul import _reduce

COMPILED = {}


@tr.jit
def _pack_pair(low, high):
    return tl.inline_asm_elementwise(
        "{ .reg .b8 pair; cvt.rn.satfinite.e2m1x2.f32 pair, $2, $1; mov.b16 $0, {pair, 0}; }",
        constraints="=h,f,f",
        args=[low, high],
        dtype=tl.uint16,
        is_pure=True,
        pack=1,
    ).to(tl.uint8)


@tr.jit
def _quantize(
    X,
    Q,
    S,
    OUTER,
    M: tl.constexpr,
    K: tl.constexpr,
    COLS: tl.constexpr,
    BLOCK: tl.constexpr,
    SWIZZLE: tl.constexpr,
):
    row = tl.program_id(0)
    offset = tl.arange(0, BLOCK)
    x = tl.load(X + row * K + offset, (row < M) & (offset < K), 0).to(tl.float32)
    maximum = tl.max(tl.abs(x), 0)
    outer = tl.maximum(maximum / (6.0 * 448.0), 1.0e-20)
    blocks = x.reshape(BLOCK // 16, 16)
    block_max = tl.max(tl.abs(blocks), 1)
    scales = tl.maximum(block_max / (6.0 * outer), 2.0**-9).to(tl.float8e4nv)
    normalized = blocks / (scales.to(tl.float32)[:, None] * outer)
    low, high = tl.split(normalized.reshape(BLOCK // 2, 2))
    packed = _pack_pair(low, high)
    p = tl.arange(0, BLOCK // 2)
    g = tl.arange(0, BLOCK // 16)
    tl.store(Q + row * (K // 2) + p, packed, (row < M) & (p < K // 2))
    if SWIZZLE:
        address = (
            ((row // 128) * (COLS // 4) + g // 4) * 512
            + (row % 32) * 16
            + ((row % 128) // 32) * 4
            + g % 4
        )
        tl.store(S + address, scales, g < COLS)
    else:
        tl.store(S + row * (K // 16) + g, scales, g < K // 16)
    tl.store(OUTER + row, outer, row < M)


def quantize(x, swizzle=False):
    if not x.is_contiguous() or x.shape[-1] % 16:
        raise ValueError("FP4 requires contiguous 16-element groups")
    k = x.shape[-1]
    rows = x.numel() // k
    q = torch.empty((rows, k // 2), device=x.device, dtype=torch.uint8)
    padded_rows = tr.cdiv(rows, 128) * 128 if swizzle else rows
    cols = tr.cdiv(k // 16, 4) * 4 if swizzle else k // 16
    scales = torch.empty(
        (padded_rows, cols), device=x.device, dtype=torch.float8_e4m3fn
    )
    outer = torch.empty(rows, device=x.device, dtype=torch.float32)
    _quantize[(padded_rows,)](
        x, q, scales, outer, rows, k, cols, tr.next_power_of_2(k), swizzle, num_warps=4
    )
    return q, scales, outer


def dequantize(q, scales, outer):
    table = torch.tensor(
        [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6],
        device=q.device,
    )
    codes = torch.stack((q & 15, q >> 4), -1).flatten(-2).long()
    return table[codes] * scales.float().repeat_interleave(16, -1) * outer[:, None]


@tr.jit
def _matmul(
    X,
    XS,
    XO,
    W,
    WS,
    WO,
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
    pair = tl.arange(0, BK // 2)
    group = tl.arange(0, BK // 16)
    steps: tl.constexpr = tl.cdiv(tl.cdiv(K, BK), SPLIT)
    acc = tl.full((BM, BN), 0, tl.float32)
    for i in range(steps):
        start = (tl.program_id(2) * steps + i) * BK
        kp, kg = start // 2 + pair, start // 16 + group
        a = tl.load(
            X + m[:, None] * (K // 2) + kp[None, :],
            (m[:, None] < M) & (kp[None, :] < K // 2),
            0,
        )
        w = tl.load(
            W + n[None, :] * (K // 2) + kp[:, None],
            (n[None, :] < N) & (kp[:, None] < K // 2),
            0,
        )
        sa = tl.load(
            XS + m[:, None] * (K // 16) + kg[None, :],
            (m[:, None] < M) & (kg[None, :] < K // 16),
            0.0,
        )
        sw = tl.load(
            WS + n[:, None] * (K // 16) + kg[None, :],
            (n[:, None] < N) & (kg[None, :] < K // 16),
            0.0,
        )
        acc = tl.dot_scaled(a, sa, "e2m1", w, sw, "e2m1", acc)
    alpha = tl.load(XO + m, m < M, 0)[:, None] * tl.load(WO + n, n < N, 0)[None, :]
    acc *= alpha
    index = m[:, None] * N + n[None, :]
    mask = (m[:, None] < M) & (n[None, :] < N)
    if SPLIT == 1:
        tl.store(Y + index, acc, mask)
    else:
        tl.store(PARTIAL + tl.program_id(2) * M * N + index, acc, mask)


def matmul(x, weight, weight_scales, weight_outer, config):
    bm, bn, bk, split, warps, stages = config
    n, kp = weight.shape
    k = kp * 2
    m = x.numel() // k
    q, scales, outer = quantize(x)
    y = torch.empty((*x.shape[:-1], n), device=x.device, dtype=torch.bfloat16)
    partial = (
        torch.empty((split, m, n), device=x.device, dtype=torch.float32)
        if split > 1
        else y
    )
    kernel = _matmul[(tr.cdiv(m, bm), tr.cdiv(n, bn), split)](
        q,
        scales,
        outer,
        weight,
        weight_scales,
        weight_outer,
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
    if not torch.compiler.is_compiling():
        COMPILED[(m, n, k, *config)] = kernel
    if split > 1:
        _reduce[(tr.cdiv(m * n, 512),)](partial, y, m * n, split, 512)
    return y
