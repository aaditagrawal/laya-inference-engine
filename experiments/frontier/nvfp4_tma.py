"""Native FP4 MMA fed by TMA descriptor loads; numerical experiment."""

import torch
import triton as tr
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from .matmul import _reduce
from .nvfp4 import quantize

COMPILED = {}


@tr.jit
def _tma(
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
    m0, n0 = tl.program_id(0) * BM, tl.program_id(1) * BN
    m, n = m0 + tl.arange(0, BM), n0 + tl.arange(0, BN)
    group = tl.arange(0, BK // 16)
    steps: tl.constexpr = tl.cdiv(tl.cdiv(K, BK), SPLIT)
    acc = tl.full((BM, BN), 0, tl.float32)
    for i in range(steps):
        start = (tl.program_id(2) * steps + i) * BK
        a = X.load([m0, start // 2])
        b = W.load([n0, start // 2])
        g = start // 16 + group
        sa = tl.load(
            XS + m[:, None] * (K // 16) + g[None, :],
            (m[:, None] < M) & (g[None, :] < K // 16),
            0.0,
        )
        sb = tl.load(
            WS + n[:, None] * (K // 16) + g[None, :],
            (n[:, None] < N) & (g[None, :] < K // 16),
            0.0,
        )
        acc = tl.dot_scaled(a, sa, "e2m1", b.T, sb, "e2m1", acc)
    acc *= tl.load(XO + m, m < M, 0)[:, None] * tl.load(WO + n, n < N, 0)[None, :]
    offset = m[:, None] * N + n[None, :]
    mask = (m[:, None] < M) & (n[None, :] < N)
    if SPLIT == 1:
        tl.store(Y + offset, acc, mask)
    else:
        tl.store(PARTIAL + tl.program_id(2) * M * N + offset, acc, mask)


def matmul(x, weight, weight_scales, weight_outer, config):
    bm, bn, bk, split, warps, stages = config
    n, kp = weight.shape
    k = 2 * kp
    m = x.numel() // k
    q, scales, outer = quantize(x)
    xd = TensorDescriptor(q, [m, kp], [kp, 1], [bm, bk // 2])
    wd = TensorDescriptor(weight, [n, kp], [kp, 1], [bn, bk // 2])
    y = torch.empty((*x.shape[:-1], n), device=x.device, dtype=torch.bfloat16)
    partial = (
        torch.empty((split, m, n), device=x.device, dtype=torch.float32)
        if split > 1
        else y
    )
    kernel = _tma[(tr.cdiv(m, bm), tr.cdiv(n, bn), split)](
        xd,
        scales,
        outer,
        wd,
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
    COMPILED[(m, n, k, *config)] = kernel
    if split > 1:
        _reduce[(tr.cdiv(m * n, 512),)](partial, y, m * n, split, 512)
    return y


@torch.library.custom_op(
    "laya_frontier_fp4_tma::linear", mutates_args=(), device_types="cuda"
)
def compiled_matmul(
    x: torch.Tensor,
    weight: torch.Tensor,
    scales: torch.Tensor,
    outer: torch.Tensor,
    config: list[int],
) -> torch.Tensor:
    return matmul(x, weight, scales, outer, tuple(config))


@compiled_matmul.register_fake
def _(x, weight, scales, outer, config):
    return torch.empty(
        (*x.shape[:-1], weight.shape[0]), device=x.device, dtype=torch.bfloat16
    )
