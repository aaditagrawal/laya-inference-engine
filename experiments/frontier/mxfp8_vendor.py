"""Probe the installed vendor MXFP8 path with directly swizzled scale output.

Scale addressing follows NVIDIA's cuBLAS 1D block-scaling-factor layout:
https://docs.nvidia.com/cuda/cublas/index.html#d-block-scaling-factors-layout
"""

import torch
import torch.nn.functional as F
import triton as tr
import triton.language as tl


@tr.jit
def _quantize(
    X, Q, S, M: tl.constexpr, K: tl.constexpr, COLS: tl.constexpr, TOTAL: tl.constexpr
):
    ids = tl.program_id(0) * 16 + tl.arange(0, 16)
    row, group = ids // COLS, ids % COLS
    k = group[:, None] * 32 + tl.arange(0, 32)[None, :]
    valid = (row[:, None] < M) & (k < K)
    x = tl.load(X + row[:, None] * K + k, valid, 0).to(tl.float32)
    maximum = tl.max(tl.abs(x), 1)
    exponent = tl.ceil(tl.log2(tl.maximum(maximum / 448.0, 2.0**-126)))
    q = tl.minimum(tl.maximum(x * tl.exp2(-exponent[:, None]), -448.0), 448.0)
    tl.store(Q + row[:, None] * K + k, q, valid)
    offset = (
        ((row // 128) * (COLS // 4) + group // 4) * 512
        + (row % 32) * 16
        + ((row % 128) // 32) * 4
        + group % 4
    )
    tl.store(S + offset, (exponent + 127).to(tl.uint8), ids < TOTAL)


def quantize(x):
    k, m = x.shape[-1], x.numel() // x.shape[-1]
    if k % 32 or not x.is_contiguous():
        raise ValueError("Expected contiguous 32-element groups")
    rows, cols = tr.cdiv(m, 128) * 128, tr.cdiv(k // 32, 4) * 4
    q = torch.empty((m, k), device=x.device, dtype=torch.float8_e4m3fn)
    scales = torch.empty(rows * cols, device=x.device, dtype=torch.uint8)
    _quantize[(tr.cdiv(rows * cols, 16),)](x, q, scales, m, k, cols, rows * cols)
    return q, scales.view(torch.float8_e8m0fnu)


def matmul(x, weight, scale):
    q, xscale = quantize(x)
    result = F.scaled_mm(
        q,
        weight.t(),
        xscale,
        F.ScalingType.BlockWise1x32,
        scale,
        F.ScalingType.BlockWise1x32,
        swizzle_a=F.SwizzleType.SWIZZLE_32_4_4,
        swizzle_b=F.SwizzleType.SWIZZLE_32_4_4,
        output_dtype=torch.bfloat16,
    )
    return result.view(*x.shape[:-1], weight.shape[0])
