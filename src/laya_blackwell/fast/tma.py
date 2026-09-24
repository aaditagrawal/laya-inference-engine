"""Small-shape GEMM with hardware tensor-descriptor loads on SM120."""

import torch
import triton as tr
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from .matmul import _reduce

COMPILED = {}


@tr.jit
def _tma(
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
    WS: tl.constexpr,
):
    m0, n0 = tl.program_id(0) * BM, tl.program_id(1) * BN
    steps: tl.constexpr = tl.cdiv(tl.cdiv(K, BK), SPLIT)
    start = tl.program_id(2) * steps * BK
    acc = tl.full((BM, BN), 0, tl.float32)
    for i in tl.range(steps, warp_specialize=WS):
        kk = start + i * BK
        a = X.load([m0, kk])
        b = W.load([n0, kk])
        acc = tl.dot(a, b.T, acc)
    m, n = m0 + tl.arange(0, BM), n0 + tl.arange(0, BN)
    offset = m[:, None] * N + n[None, :]
    mask = (m[:, None] < M) & (n[None, :] < N)
    if SPLIT == 1:
        tl.store(Y + offset, acc, mask)
    else:
        tl.store(PARTIAL + tl.program_id(2) * M * N + offset, acc, mask)


def matmul(x, weight, config):
    bm, bn, bk, split, warps, stages, ws = config
    n, k = weight.shape
    m = x.numel() // k
    if not x.is_contiguous() or not weight.is_contiguous():
        raise ValueError("Expected contiguous matrices")
    xd = TensorDescriptor(x, [m, k], [k, 1], [bm, bk])
    wd = TensorDescriptor(weight, [n, k], [k, 1], [bn, bk])
    y = torch.empty((*x.shape[:-1], n), device=x.device, dtype=x.dtype)
    partial = (
        torch.empty((split, m, n), device=x.device, dtype=torch.float32)
        if split > 1
        else y
    )
    kernel = _tma[(tr.cdiv(m, bm), tr.cdiv(n, bn), split)](
        xd,
        wd,
        partial,
        y,
        m,
        n,
        k,
        bm,
        bn,
        bk,
        split,
        ws,
        num_warps=warps,
        num_stages=stages,
    )
    if not torch.compiler.is_compiling():
        COMPILED[(m, n, k, *config)] = kernel
    if split > 1:
        _reduce[(tr.cdiv(m * n, 512),)](partial, y, m * n, split, 512)
    return y


@torch.library.custom_op("laya_fast_tma::linear", mutates_args=(), device_types="cuda")
def compiled_matmul(
    x: torch.Tensor, weight: torch.Tensor, config: list[int]
) -> torch.Tensor:
    # Keep host descriptor construction outside Dynamo tracing. During request
    # serving the surrounding CUDA Graph replays the captured GPU operations.
    return matmul(x, weight, tuple(config))


@compiled_matmul.register_fake
def _(x, weight, config):
    return torch.empty((*x.shape[:-1], weight.shape[0]), device=x.device, dtype=x.dtype)
