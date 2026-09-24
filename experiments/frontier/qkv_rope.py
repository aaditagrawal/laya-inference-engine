"""TMA QKV projection with exactly rounded rotary embedding in its epilogue."""

import torch
import triton as tr
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


@tr.jit
def _project(
    X,
    W,
    C,
    S,
    Y,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    WS: tl.constexpr,
    INTERLEAVED: tl.constexpr,
):
    m0, n0 = tl.program_id(0) * BM, tl.program_id(1) * BN
    acc = tl.full((BM, BN), 0, tl.float32)
    for kk in tl.range(0, 1024, BK, warp_specialize=WS):
        acc = tl.dot(X.load([m0, kk]), W.load([n0, kk]).T, acc)
    rounded = acc.to(tl.bfloat16).to(tl.float32)
    m, n = m0 + tl.arange(0, BM), n0 + tl.arange(0, BN)
    col = tl.arange(0, BN)
    # Each tile spans whole 64-value heads. Keep GEMM's BF16 rounding before
    # FP32 RoPE, and keep its two products separately rounded (no FMA).
    other = col ^ (1 if INTERLEAVED else 32)
    paired = tl.gather(rounded, tl.broadcast_to(other[None, :], (BM, BN)), 1)
    d = n % 64
    if INTERLEAVED:
        d = d // 2 + (d % 2) * 32
        n = (n // 64) * 64 + d
    paired = tl.where(d[None, :] < 32, -paired, paired)
    cosine = tl.load(C + m[:, None] * 64 + d[None, :], m[:, None] < 64, 0)
    sine = tl.load(S + m[:, None] * 64 + d[None, :], m[:, None] < 64, 0)
    output = tl.where(n[None, :] < 2048, rounded * cosine + paired * sine, rounded)
    tl.store(Y + m[:, None] * 3072 + n[None, :], output, m[:, None] < 64)


def project(x, weight, cosine, sine, config, interleaved=False):
    bm, bn, bk, warps, stages, ws = config
    if x.numel() != 64 * 1024 or weight.shape != (3072, 1024) or bn % 64:
        raise ValueError("Expected 64x1024 input and 3072x1024 QKV weight")
    xd = TensorDescriptor(x, [64, 1024], [1024, 1], [bm, bk])
    wd = TensorDescriptor(weight, [3072, 1024], [1024, 1], [bn, bk])
    output = torch.empty((*x.shape[:-1], 3072), device=x.device, dtype=x.dtype)
    _project[(tr.cdiv(64, bm), tr.cdiv(3072, bn))](
        xd,
        wd,
        cosine,
        sine,
        output,
        bm,
        bn,
        bk,
        ws,
        interleaved,
        num_warps=warps,
        num_stages=stages,
        enable_fp_fusion=False,
    )
    return output
