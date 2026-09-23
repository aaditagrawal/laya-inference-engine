"""Small CUDA kernels compiled by Triton for the actual device architecture.

The default model uses RoPE only. Fused norm and GEGLU are retained as tested
experiments; their rounding differed enough to exclude them from the default.
"""
import torch
import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice


@tr.jit
def _add_norm(X, R, W, B, Y, N, D: tl.constexpr, EPS: tl.constexpr,
              HAS_R: tl.constexpr, HAS_B: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * D + col, col < D, 0).to(tl.float32)
    if HAS_R:
        x += tl.load(R + row * D + col, col < D, 0).to(tl.float32)
    mean = tl.sum(x, 0) / D
    diff = tl.where(col < D, x - mean, 0.)
    var = tl.sum(diff * diff, 0) / D
    w = tl.load(W + col, col < D, 0).to(tl.float32)
    n = (x - mean) * tl.rsqrt(var + EPS) * w
    if HAS_B:
        n += tl.load(B + col, col < D, 0).to(tl.float32)
    tl.store(Y + row * D + col, x, col < D)
    tl.store(N + row * D + col, n, col < D)


def add_norm(x, residual, norm):
    """FP32 residual addition + LayerNorm, BF16 output for Tensor Core GEMM."""
    y = torch.empty_like(x, dtype=torch.float32)
    n = torch.empty_like(x, dtype=torch.bfloat16)
    d = x.shape[-1]
    _add_norm[(x.numel() // d,)](
        x, residual if residual is not None else x, norm.weight,
        norm.bias if norm.bias is not None else norm.weight, y, n, d, norm.eps,
        residual is not None, norm.bias is not None, tr.next_power_of_2(d),
    )
    return y, n


@tr.jit
def _geglu(X, Y, D: tl.constexpr, TOTAL: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row, col = i // D, i % D
    a = tl.load(X + row * (2 * D) + col, i < TOTAL, 0).to(tl.float32)
    b = tl.load(X + row * (2 * D) + col + D, i < TOTAL, 0).to(tl.float32)
    # Preserve the upstream BF16 GELU rounding before the gate multiplication.
    g = (0.5 * a * (1. + libdevice.erf(a * 0.7071067811865476))).to(tl.bfloat16).to(tl.float32)
    tl.store(Y + i, g * b, i < TOTAL)


def geglu(x):
    y = torch.empty((*x.shape[:-1], x.shape[-1] // 2), device=x.device, dtype=x.dtype)
    _geglu[(tr.cdiv(y.numel(), 512),)](x, y, y.shape[-1], y.numel(), 512,
                                   enable_fp_fusion=False)
    return y


@tr.jit
def _rope(X, C, S, Y, L: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
          TOTAL: tl.constexpr, BLOCK: tl.constexpr, FP32: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    d = i % D
    head = (i // D) % H
    qkv = (i // (D * H)) % 3
    pos = (i // (D * H * 3)) % L
    a = tl.load(X + i, i < TOTAL, 0).to(tl.float32)
    other = i + tl.where(d < D // 2, D // 2, -D // 2)
    b = tl.load(X + other, i < TOTAL, 0).to(tl.float32)
    b = tl.where(d < D // 2, -b, b)
    c = tl.load(C + pos * D + d, i < TOTAL, 0).to(tl.float32)
    s = tl.load(S + pos * D + d, i < TOTAL, 0).to(tl.float32)
    if FP32:
        ac = a * c
        bs = b * s
    else:
        ac = (a * c).to(tl.bfloat16).to(tl.float32)
        bs = (b * s).to(tl.bfloat16).to(tl.float32)
    y = tl.where(qkv < 2, ac + bs, a)
    tl.store(Y + i, y, i < TOTAL)


def rope_qkv(qkv, cos, sin, *, fp32=False):
    y = torch.empty_like(qkv)
    _, length, _, heads, dim = qkv.shape
    _rope[(tr.cdiv(qkv.numel(), 512),)](
        qkv, cos, sin, y, length, heads, dim, qkv.numel(), 512, fp32,
        enable_fp_fusion=False,
    )
    return y
