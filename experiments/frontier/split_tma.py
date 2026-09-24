"""TMA loads with the retained MLP-output BF16 split boundaries and rounding."""

import torch
import triton as tr
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

COMPILED = {}


@tr.jit
def _partial(
    X,
    W,
    Y,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    BOTH: tl.constexpr,
):
    m0, n0, part = tl.program_id(0) * BM, tl.program_id(1) * BN, tl.program_id(2)
    m, n = m0 + tl.arange(0, BM), n0 + tl.arange(0, BN)
    lane = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    # The cuBLAS-compatible partition is 704,704,704,512. It must not
    # silently change when changing the K tile size.
    for step in range(tr.cdiv(704, BK)):
        start = part * 704 + step * BK
        k = start + lane
        if BOTH:
            a = X.load([m0, start])
            if 704 % BK:
                a = tl.where(k[None, :] < (part + 1) * 704, a, 0)
        else:
            a = tl.load(
                X + m[:, None] * 2624 + k[None, :],
                (m[:, None] < 64)
                & (k[None, :] < 2624)
                & (k[None, :] < (part + 1) * 704),
                0,
            )
        w = W.load([n0, start])
        acc = tl.dot(a, w.T, acc)
    tl.store(
        Y + part * 64 * 1024 + m[:, None] * 1024 + n[None, :],
        acc.to(tl.bfloat16),
        (m[:, None] < 64) & (n[None, :] < 1024),
    )


def partial(x, weight, config):
    bm, bn, bk, warps, stages, both = config
    if (
        x.shape[-1] != 2624
        or x.numel() != 64 * 2624
        or weight.shape != (1024, 2624)
        or not x.is_contiguous()
        or not weight.is_contiguous()
        or not x.is_cuda
        or x.device != weight.device
        or x.dtype != torch.bfloat16
        or weight.dtype != x.dtype
    ):
        raise ValueError("Expected contiguous CUDA BF16 64x2624 and 1024x2624")
    a = TensorDescriptor(x, [64, 2624], [2624, 1], [bm, bk]) if both else x
    w = TensorDescriptor(weight, [1024, 2624], [2624, 1], [bn, bk])
    output = torch.empty((4, *x.shape[:-1], 1024), device=x.device, dtype=x.dtype)
    kernel = _partial[(tr.cdiv(64, bm), tr.cdiv(1024, bn), 4)](
        a,
        w,
        output,
        bm,
        bn,
        bk,
        both,
        num_warps=warps,
        num_stages=stages,
    )
    COMPILED[tuple(config)] = kernel
    return output


@torch.library.custom_op(
    "laya_frontier_split_tma::partial", mutates_args=(), device_types="cuda"
)
def compiled_partial(
    x: torch.Tensor, weight: torch.Tensor, config: list[int]
) -> torch.Tensor:
    return partial(x, weight, config)


@compiled_partial.register_fake
def _(x, weight, config):
    return torch.empty((4, *x.shape[:-1], 1024), device=x.device, dtype=x.dtype)
