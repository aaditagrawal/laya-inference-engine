"""Optional exactness-screened small head GEMMs, with BF16 bias epilogues."""

import torch
import triton as tr
import triton.language as tl


@tr.jit
def _gemm(
    X,
    W,
    B,
    P,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SPLIT: tl.constexpr,
    RELU: tl.constexpr,
):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    steps: tl.constexpr = tl.cdiv(tl.cdiv(K, BK), SPLIT)
    k0 = tl.program_id(2) * steps * BK
    kk = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for step in range(steps):
        k = k0 + step * BK + kk
        a = tl.load(
            X + m[:, None] * K + k[None, :], (m[:, None] < M) & (k[None, :] < K), 0
        )
        w = tl.load(
            W + n[None, :] * K + k[:, None], (n[None, :] < N) & (k[:, None] < K), 0
        )
        acc = tl.dot(a, w, acc)
    offsets = m[:, None] * N + n[None, :]
    mask = (m[:, None] < M) & (n[None, :] < N)
    if SPLIT == 1:
        acc += tl.load(B + n, n < N, 0).to(tl.float32)[None, :]
        if RELU:
            acc = tl.maximum(acc, 0)
        tl.store(Y + offsets, acc, mask)
    else:
        tl.store(P + tl.program_id(2) * M * N + offsets, acc, mask)


@tr.jit
def _reduce(
    P,
    B,
    Y,
    TOTAL: tl.constexpr,
    N: tl.constexpr,
    SPLIT: tl.constexpr,
    RELU: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    acc = tl.full((BLOCK,), 0, tl.float32)
    for p in tl.static_range(SPLIT):
        acc += tl.load(P + p * TOTAL + i, i < TOTAL, 0)
    acc += tl.load(B + i % N).to(tl.float32)
    if RELU:
        acc = tl.maximum(acc, 0)
    tl.store(Y + i, acc, i < TOTAL)


def gemm(x, weight, bias, config, *, relu=False):
    bm, bn, bk, split, warps, stages, partial_bf16 = config
    n, k = weight.shape
    m = x.numel() // k
    if (
        not x.is_contiguous()
        or not weight.is_contiguous()
        or x.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or x.shape[-1] != k
        or bias.shape != (n,)
    ):
        raise ValueError("Expected contiguous BF16 matrices and matching bias")
    y = torch.empty((*x.shape[:-1], n), device=x.device, dtype=x.dtype)
    p = (
        torch.empty(
            (split, m, n),
            device=x.device,
            dtype=torch.bfloat16 if partial_bf16 else torch.float32,
        )
        if split > 1
        else y
    )
    _gemm[(tr.cdiv(m, bm), tr.cdiv(n, bn), split)](
        x,
        weight,
        bias,
        p,
        y,
        m,
        n,
        k,
        bm,
        bn,
        bk,
        split,
        relu,
        num_warps=warps,
        num_stages=stages,
    )
    if split > 1:
        _reduce[(tr.cdiv(m * n, 512),)](p, bias, y, m * n, n, split, relu, 512)
    return y
