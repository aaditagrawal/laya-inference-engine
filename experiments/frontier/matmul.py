"""Small-M matrix kernels. Weight-only INT8 is a separate nonexact experiment."""

import torch
import triton as tr
import triton.language as tl


@tr.jit
def _matmul(
    X,
    W,
    SCALE,
    PARTIAL,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SPLIT: tl.constexpr,
    QUANT: tl.constexpr,
    GROUP: tl.constexpr,
):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    part = tl.program_id(2)
    steps: tl.constexpr = ((K + BK - 1) // BK + SPLIT - 1) // SPLIT
    kk = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for i in range(steps):
        k = (part * steps + i) * BK + kk
        a = tl.load(
            X + m[:, None] * K + k[None, :], (m[:, None] < M) & (k[None, :] < K), 0
        )
        w = tl.load(
            W + n[None, :] * K + k[:, None], (n[None, :] < N) & (k[:, None] < K), 0.0
        )
        if QUANT == 1:
            scale = tl.load(
                SCALE + n[None, :] * ((K + GROUP - 1) // GROUP) + k[:, None] // GROUP,
                (n[None, :] < N) & (k[:, None] < K),
                0,
            )
            w = (w.to(tl.float32) * scale).to(tl.bfloat16)
        elif QUANT == 2:
            w = w.to(tl.bfloat16)
        elif QUANT == 3:
            # Exact BF16 values held in FP32 compressible memory. Only storage
            # changes; tensor-core inputs and accumulation remain BF16/FP32.
            w = w.to(tl.bfloat16)
        acc = tl.dot(a, w, acc)
    if QUANT == 2:
        scale = tl.load(SCALE + n, n < N, 0)
        acc *= scale[None, :]
    offsets = m[:, None] * N + n[None, :]
    mask = (m[:, None] < M) & (n[None, :] < N)
    if SPLIT == 1:
        tl.store(Y + offsets, acc, mask)
    else:
        tl.store(PARTIAL + part * M * N + offsets, acc, mask)


@tr.jit
def _reduce(PARTIAL, Y, TOTAL: tl.constexpr, SPLIT: tl.constexpr, BLOCK: tl.constexpr):
    offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    acc = tl.full((BLOCK,), 0, tl.float32)
    for part in tl.static_range(SPLIT):
        acc += tl.load(PARTIAL + part * TOTAL + offset, offset < TOTAL, 0)
    tl.store(Y + offset, acc, offset < TOTAL)


def quantize_weight(weight, group=64):
    n, k = weight.shape
    if k % group:
        raise ValueError("Weight columns must divide the quantization group")
    grouped = weight.float().view(n, k // group, group)
    scale = (grouped.abs().amax(-1) / 127).clamp_min(1e-20)
    quantized = (
        (grouped / scale[..., None]).round().clamp(-127, 127).to(torch.int8).view(n, k)
    )
    return quantized, scale


def quantize_fp8_weight(weight):
    # Power-of-two row scales permit scaling once after FP32 accumulation.
    amax = weight.float().abs().amax(dim=1).clamp_min(1e-20)
    scale = torch.exp2(torch.ceil(torch.log2(amax / 448)))
    return (weight.float() / scale[:, None]).to(torch.float8_e4m3fn), scale


def matmul(
    x,
    weight,
    config,
    scale=None,
    group=64,
    quant_mode=None,
    partial_bf16=False,
    return_partials=False,
):
    bm, bn, bk, split, warps, stages = config
    n, k = weight.shape
    if x.shape[-1] != k or not x.is_contiguous():
        raise ValueError("Expected contiguous input with the matching reduction size")
    m = x.numel() // k
    y = torch.empty((*x.shape[:-1], n), device=x.device, dtype=torch.bfloat16)
    partial = (
        torch.empty(
            (split, m, n),
            device=x.device,
            dtype=torch.bfloat16 if partial_bf16 else torch.float32,
        )
        if split > 1
        else y
    )
    _matmul[(tr.cdiv(m, bm), tr.cdiv(n, bn), split)](
        x,
        weight,
        scale if scale is not None else weight,
        partial,
        y,
        m,
        n,
        k,
        bm,
        bn,
        bk,
        split,
        int(scale is not None) if quant_mode is None else quant_mode,
        group,
        num_warps=warps,
        num_stages=stages,
    )
    if return_partials:
        if split != 4 or not partial_bf16:
            raise ValueError("Fused normalization requires four BF16 partials")
        return partial.view(split, *x.shape[:-1], n)
    if split > 1:
        _reduce[(tr.cdiv(m * n, 512),)](partial, y, m * n, split, 512)
    return y
