"""BF16 projection/epilogue fusions for the fixed Laya encoder shapes.

GEMM accumulation order may differ from cuBLAS. Every projected value is rounded
back to BF16 before GELU, gating or rotary arithmetic, matching model semantics.
"""

import torch
import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice

from experiments.native.kernels.candidates import gelu_corrections, gelu_lut

COMPILED = {}


@tr.jit
def _gelu(a, LUT, USE_LUT: tl.constexpr, IN_BITS: tl.constexpr, OUT_BITS: tl.constexpr):
    a = a.to(tl.bfloat16)
    bits = a.to(tl.uint16, bitcast=True)
    if USE_LUT:
        return tl.load(LUT + bits.to(tl.int32)).to(tl.float32)
    af = a.to(tl.float32)
    g = (0.5 * af * (1.0 + libdevice.erf(af * 0.7071067811865476))).to(tl.bfloat16)
    for j in tl.static_range(len(IN_BITS)):
        g = tl.where(
            bits == IN_BITS[j],
            tl.full((), OUT_BITS[j], tl.uint16).to(tl.bfloat16, bitcast=True),
            g,
        )
    return g.to(tl.float32)


@tr.jit
def _wi_dual(
    X,
    W,
    BIAS,
    LUT,
    Y,
    M: tl.constexpr,
    D: tl.constexpr,
    K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    USE_LUT: tl.constexpr,
    IN_BITS: tl.constexpr,
    OUT_BITS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    acc_a = tl.full((BM, BN), 0.0, tl.float32)
    acc_b = tl.full((BM, BN), 0.0, tl.float32)
    for start in range(tr.cdiv(K, BK)):
        kk = start * BK + k
        a = tl.load(
            X + m[:, None] * K + kk[None, :], (m[:, None] < M) & (kk[None, :] < K), 0
        )
        wa = tl.load(
            W + n[None, :] * K + kk[:, None], (n[None, :] < D) & (kk[:, None] < K), 0
        )
        wb = tl.load(
            W + (n[None, :] + D) * K + kk[:, None],
            (n[None, :] < D) & (kk[:, None] < K),
            0,
        )
        acc_a = tl.dot(a, wa, acc_a)
        acc_b = tl.dot(a, wb, acc_b)
    if HAS_BIAS:
        acc_a += tl.load(BIAS + n, n < D, 0)[None, :].to(tl.float32)
        acc_b += tl.load(BIAS + D + n, n < D, 0)[None, :].to(tl.float32)
    g = _gelu(acc_a, LUT, USE_LUT, IN_BITS, OUT_BITS)
    gate = acc_b.to(tl.bfloat16).to(tl.float32)
    tl.store(
        Y + m[:, None] * D + n[None, :], g * gate, (m[:, None] < M) & (n[None, :] < D)
    )


@tr.jit
def _wi_packed(
    X,
    W,
    BIAS,
    LUT,
    Y,
    M: tl.constexpr,
    D: tl.constexpr,
    K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    USE_LUT: tl.constexpr,
    IN_BITS: tl.constexpr,
    OUT_BITS: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    nn = tl.program_id(1) * (2 * BN) + tl.arange(0, 2 * BN)
    k = tl.arange(0, BK)
    acc = tl.full((BM, 2 * BN), 0.0, tl.float32)
    for start in range(tr.cdiv(K, BK)):
        kk = start * BK + k
        a = tl.load(
            X + m[:, None] * K + kk[None, :], (m[:, None] < M) & (kk[None, :] < K), 0
        )
        w = tl.load(
            W + nn[None, :] * K + kk[:, None],
            (nn[None, :] < 2 * D) & (kk[:, None] < K),
            0,
        )
        acc = tl.dot(a, w, acc)
    if HAS_BIAS:
        acc += tl.load(BIAS + nn, nn < 2 * D, 0)[None, :].to(tl.float32)
    acc_a, acc_b = tl.split(tl.reshape(acc, (BM, BN, 2)))
    g = _gelu(acc_a, LUT, USE_LUT, IN_BITS, OUT_BITS)
    gate = acc_b.to(tl.bfloat16).to(tl.float32)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    tl.store(
        Y + m[:, None] * D + n[None, :], g * gate, (m[:, None] < M) & (n[None, :] < D)
    )


@tr.jit
def _qkv_rope(
    X,
    W,
    BIAS,
    COS,
    SIN,
    Y,
    M: tl.constexpr,
    L: tl.constexpr,
    PAD: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    PACKED: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0.0, tl.float32)
    for start in range(tr.cdiv(K, BK)):
        kk = start * BK + k
        a = tl.load(
            X + m[:, None] * K + kk[None, :], (m[:, None] < M) & (kk[None, :] < K), 0
        )
        w = tl.load(
            W + n[None, :] * K + kk[:, None], (n[None, :] < N) & (kk[:, None] < K), 0
        )
        acc = tl.dot(a, w, acc)
    if HAS_BIAS:
        acc += tl.load(BIAS + n, n < N, 0)[None, :].to(tl.float32)
    projected = acc.to(tl.bfloat16).to(tl.float32)
    if PACKED:
        d = (n % 64) // 2 + (n % 2) * 32
        original_n = n // 64 * 64 + d
        other = tl.arange(0, BN) ^ 1
    else:
        d = n % 64
        original_n = n
        other = tl.arange(0, BN) ^ 32
    rotated = tl.gather(projected, tl.broadcast_to(other[None, :], (BM, BN)), 1)
    rotated = tl.where(d[None, :] < 32, -rotated, rotated)
    pos = m % L
    cos = tl.load(COS + pos[:, None] * 64 + d[None, :], m[:, None] < M, 0).to(
        tl.float32
    )
    sin = tl.load(SIN + pos[:, None] * 64 + d[None, :], m[:, None] < M, 0).to(
        tl.float32
    )
    value = tl.where(n[None, :] < 2048, projected * cos + rotated * sin, projected)
    output_row = m // L * (L + PAD) + pos
    tl.store(
        Y + output_row[:, None] * N + original_n[None, :],
        value,
        (m[:, None] < M) & (n[None, :] < N),
    )
    if PAD > 0:
        tail_row = m // L * (L + PAD) + L + pos
        tl.store(
            Y + tail_row[:, None] * N + original_n[None, :],
            0.0,
            (m[:, None] < M) & (pos[:, None] < PAD) & (n[None, :] < N),
        )


# BM, BN, BK, warps, stages. BN counts gated channels for Wi, projected channels for QKV.
WI_CONFIGS = [
    (32, 32, 32, 4, 3),
    (32, 64, 32, 4, 3),
    (32, 64, 64, 4, 3),
    (64, 32, 32, 4, 3),
    (64, 64, 32, 4, 3),
    (64, 64, 64, 4, 3),
    (64, 128, 32, 8, 3),
    (128, 64, 32, 8, 3),
]
QKV_CONFIGS = [
    (32, 64, 32, 4, 3),
    (32, 128, 32, 4, 3),
    (64, 64, 32, 4, 3),
    (64, 128, 32, 4, 3),
    (64, 128, 64, 4, 3),
    (128, 64, 32, 4, 3),
]


def pack_wi(weight, bias=None):
    d = weight.shape[0] // 2
    index = torch.stack(
        (
            torch.arange(d, device=weight.device),
            torch.arange(d, device=weight.device) + d,
        ),
        -1,
    ).flatten()
    return weight[index].contiguous(), bias[
        index
    ].contiguous() if bias is not None else None


def pack_qkv(weight, bias=None):
    n = torch.arange(weight.shape[0], device=weight.device)
    index = n // 64 * 64 + (n % 64) // 2 + (n % 2) * 32
    return weight[index].contiguous(), bias[
        index
    ].contiguous() if bias is not None else None


def _validate_projection(x, weight, bias, out_features):
    if (
        not x.is_cuda
        or x.dtype != torch.bfloat16
        or not x.is_contiguous()
        or x.shape[-1] != 1024
        or weight.shape != (out_features, 1024)
        or weight.dtype != x.dtype
        or weight.device != x.device
        or not weight.is_contiguous()
    ):
        raise ValueError(
            "Expected contiguous CUDA BF16 input and checkpoint projection weights"
        )
    if bias is not None and (
        bias.shape != (out_features,)
        or bias.dtype != x.dtype
        or bias.device != x.device
        or not bias.is_contiguous()
    ):
        raise ValueError("Projection bias must match the weight rows, dtype and device")


def wi_geglu(x, weight, bias, config, *, packed=True, lut=False):
    _validate_projection(x, weight, bias, 5248)
    m, k = x.numel() // x.shape[-1], x.shape[-1]
    d = weight.shape[0] // 2
    y = torch.empty((*x.shape[:-1], d), device=x.device, dtype=x.dtype)
    ins, outs = gelu_corrections()
    bm, bn, bk, warps, stages = config
    kernel = _wi_packed if packed else _wi_dual
    compiled = kernel[(tr.cdiv(m, bm), tr.cdiv(d, bn))](
        x,
        weight,
        bias if bias is not None else weight,
        gelu_lut() if lut else weight,
        y,
        m,
        d,
        k,
        bias is not None,
        lut,
        ins,
        outs,
        bm,
        bn,
        bk,
        num_warps=warps,
        num_stages=stages,
        enable_fp_fusion=False,
    )
    COMPILED[("wi", m, packed, lut, tuple(config))] = compiled
    return y


def qkv_rope(x, weight, bias, cos, sin, config, *, packed=True, padding=0):
    _validate_projection(x, weight, bias, 3072)
    batch, length, k = x.shape
    n = weight.shape[0]
    if padding < 0 or padding > length or config[1] % 64:
        raise ValueError(
            "RoPE requires 64-channel head-aligned tiles and padding <= length"
        )
    for rotary in (cos, sin):
        if (
            rotary.device != x.device
            or rotary.dtype != torch.float32
            or rotary.ndim != 2
            or rotary.shape[0] < length
            or rotary.shape[1] != 64
            or not rotary.is_contiguous()
        ):
            raise ValueError(
                "Expected contiguous FP32 RoPE tables covering the sequence"
            )
    y = torch.empty(
        (batch, length + padding, 3, 16, 64), device=x.device, dtype=x.dtype
    )
    bm, bn, bk, warps, stages = config
    compiled = _qkv_rope[(tr.cdiv(batch * length, bm), tr.cdiv(n, bn))](
        x,
        weight,
        bias if bias is not None else weight,
        cos,
        sin,
        y,
        batch * length,
        length,
        padding,
        n,
        k,
        bias is not None,
        packed,
        bm,
        bn,
        bk,
        num_warps=warps,
        num_stages=stages,
        enable_fp_fusion=False,
    )
    COMPILED[("qkv", batch * length, packed, padding, tuple(config))] = compiled
    q, key, value = y.permute(0, 3, 2, 1, 4).unbind(2)
    return q[:, :, :length], key, value
