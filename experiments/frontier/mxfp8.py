"""SM120 block-scaled FP8 experiment. Numerically changed, never a default."""

import torch
import triton as tr
import triton.language as tl

from .matmul import _reduce

COMPILED = {}


@tr.jit
def _quantize(X, Q, S, TOTAL: tl.constexpr, GROUPS: tl.constexpr, BLOCK: tl.constexpr):
    groups = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    columns = tl.arange(0, 32)
    index = groups[:, None] * 32 + columns[None, :]
    x = tl.load(X + index, index < TOTAL, 0).to(tl.float32)
    maximum = tl.max(tl.abs(x), 1)
    exponent = tl.ceil(tl.log2(tl.maximum(maximum / 448.0, 2.0**-126)))
    scale = tl.exp2(exponent)
    q = tl.minimum(tl.maximum(x / scale[:, None], -448.0), 448.0)
    tl.store(Q + index, q, index < TOTAL)
    tl.store(S + groups, (exponent + 127).to(tl.uint8), groups < GROUPS)


def quantize(x):
    if not x.is_contiguous() or x.shape[-1] % 32:
        raise ValueError("MXFP8 requires contiguous 32-element groups")
    q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scales = torch.empty(
        (*x.shape[:-1], x.shape[-1] // 32), device=x.device, dtype=torch.uint8
    )
    _quantize[(tr.cdiv(scales.numel(), 16),)](
        x, q, scales, x.numel(), scales.numel(), 16
    )
    return q, scales


@tr.jit
def _matmul(
    X,
    XS,
    W,
    WS,
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
    group = tl.arange(0, BK // 32)
    steps: tl.constexpr = tr.cdiv(tr.cdiv(K, BK), SPLIT)
    acc = tl.full((BM, BN), 0, tl.float32)
    for i in range(steps):
        k = (part * steps + i) * BK + kk
        kg = (part * steps + i) * (BK // 32) + group
        a = tl.load(
            X + m[:, None] * K + k[None, :], (m[:, None] < M) & (k[None, :] < K), 0.0
        )
        w = tl.load(
            W + n[None, :] * K + k[:, None], (n[None, :] < N) & (k[:, None] < K), 0.0
        )
        sa = tl.load(
            XS + m[:, None] * (K // 32) + kg[None, :],
            (m[:, None] < M) & (kg[None, :] < K // 32),
            127,
        )
        sw = tl.load(
            WS + n[:, None] * (K // 32) + kg[None, :],
            (n[:, None] < N) & (kg[None, :] < K // 32),
            127,
        )
        acc = tl.dot_scaled(a, sa, "e4m3", w, sw, "e4m3", acc)
    offset = m[:, None] * N + n[None, :]
    valid = (m[:, None] < M) & (n[None, :] < N)
    if SPLIT == 1:
        tl.store(Y + offset, acc, valid)
    else:
        tl.store(PARTIAL + part * M * N + offset, acc, valid)


def matmul(x, weight, weight_scales, config):
    bm, bn, bk, split, warps, stages = config
    n, k = weight.shape
    m = x.numel() // k
    q, scales = quantize(x)
    y = torch.empty((*x.shape[:-1], n), device=x.device, dtype=torch.bfloat16)
    partial = (
        torch.empty((split, m, n), device=x.device, dtype=torch.float32)
        if split > 1
        else y
    )
    kernel = _matmul[(tr.cdiv(m, bm), tr.cdiv(n, bn), split)](
        q,
        scales,
        weight,
        weight_scales,
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
    if kernel is not None:
        COMPILED[(m, n, k, *config)] = kernel
    if split > 1:
        _reduce[(tr.cdiv(m * n, 512),)](partial, y, m * n, split, 512)
    return y
