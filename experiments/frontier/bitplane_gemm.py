"""Warp-shuffle reconstruction of exact tiled BF16 weight bitplanes."""

import torch
import triton as tr
import triton.language as tl
from triton.experimental import gluon as g
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.nvidia.ampere import mma_v2

COMPILED = {}


@tr.jit
def _pack(X, Y, K: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    groups: tl.constexpr = BN * BK // 32
    group = tl.arange(0, groups)
    lane = tl.arange(0, 32)
    nt, kt, plane = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    local = group[:, None] * 32 + lane[None, :]
    bits = tl.load(X + (nt * BN + local // BK) * K + kt * BK + local % BK).to(tl.uint32)
    exponent = ((bits >> 7) & 255).to(tl.int32)
    delta = exponent - 119
    z = (delta << 1) ^ (delta >> 31)
    code = (bits & 127) | ((bits >> 15) << 7) | (z.to(tl.uint32) << 8)
    word = tl.sum(((code >> plane) & 1) << lane[None, :], 1)
    tile = nt * (K // BK) + kt
    tl.store(Y + (tile * 16 + plane) * groups + group, word)


@g.jit
def _transpose_stage(word, lane, SHIFT: gl.constexpr):
    s: gl.constexpr = 1 << SHIFT
    mask: gl.constexpr = 0xFFFFFFFF // ((1 << s) + 1)
    high: gl.constexpr = mask ^ 0xFFFFFFFF
    peer = gl.inline_asm_elementwise(
        "shfl.sync.bfly.b32 $0, $1, $2, 31, -1;",
        constraints="=r,r,r",
        args=[word, s],
        dtype=gl.uint32,
        is_pure=True,
        pack=1,
    )
    return gl.where(
        (lane & s) == 0,
        (word & mask) | ((peer & mask) << s),
        ((peer & high) >> s) | (word & high),
    )


@g.jit
def _decode(W, tile, BN: gl.constexpr, BK: gl.constexpr):
    groups: gl.constexpr = BN * BK // 32
    load_layout: gl.constexpr = gl.BlockedLayout([1, 1], [32, 1], [1, 4], [0, 1])
    shuffle_layout: gl.constexpr = gl.BlockedLayout([1, 1], [1, 32], [4, 1], [0, 1])
    group = gl.arange(0, groups, layout=gl.SliceLayout(1, load_layout))
    plane = gl.arange(0, 32, layout=gl.SliceLayout(0, load_layout))
    word = gl.load(
        W + tile * 16 * groups + plane[None, :] * groups + group[:, None],
        plane[None, :] < 16,
        0,
    )
    word = gl.convert_layout(word, shuffle_layout)
    lane = gl.arange(0, 32, layout=gl.SliceLayout(0, shuffle_layout))[None, :]
    # Bit transpose: each warp starts with one bitplane per lane and ends
    # with one original weight code per lane. All lanes participate.
    for shift in gl.static_range(5):
        word = _transpose_stage(word, lane, shift)
    z = word >> 8
    exponent = 119 + ((z >> 1).to(gl.int32) ^ -(z & 1).to(gl.int32))
    bits = (word & 127) | (((word >> 7) & 1) << 15) | (exponent.to(gl.uint32) << 7)
    return gl.reshape(bits.to(gl.uint16).to(gl.bfloat16, bitcast=True), (BN, BK))


@g.jit
def _unpack(W, Y, N: gl.constexpr, K: gl.constexpr, BN: gl.constexpr, BK: gl.constexpr):
    nt, kt = gl.program_id(0), gl.program_id(1)
    value = _decode(W, nt * (K // BK) + kt, BN, BK)
    layout: gl.constexpr = gl.BlockedLayout([1, 4], [4, 8], [4, 1], [1, 0])
    value = gl.convert_layout(value, layout)
    n = nt * BN + gl.arange(0, BN, layout=gl.SliceLayout(1, layout))
    k = kt * BK + gl.arange(0, BK, layout=gl.SliceLayout(0, layout))
    gl.store(Y + n[:, None] * K + k[None, :], value)


@g.jit
def _gemm(
    X,
    W,
    Y,
    N: gl.constexpr,
    K: gl.constexpr,
    BM: gl.constexpr,
    BN: gl.constexpr,
    BK: gl.constexpr,
    PACKED: gl.constexpr,
    STAGES: gl.constexpr,
):
    layout: gl.constexpr = gl.BlockedLayout([1, 4], [4, 8], [4, 1], [1, 0])
    mma: gl.constexpr = gl.NVMMADistributedLayout(
        version=[2, 0], warps_per_cta=[2, 2], instr_shape=[16, 8]
    )
    m = gl.program_id(0) * BM + gl.arange(0, BM, layout=gl.SliceLayout(1, layout))
    k = gl.arange(0, BK, layout=gl.SliceLayout(0, layout))
    n = gl.program_id(1) * BN + gl.arange(0, BN, layout=gl.SliceLayout(1, layout))
    acc = gl.full((BM, BN), 0, gl.float32, layout=mma)
    for step in range(K // BK):
        a = gl.load(X + m[:, None] * K + step * BK + k[None, :], m[:, None] < 64, 0)
        if PACKED:
            b = _decode(W, gl.program_id(1) * (K // BK) + step, BN, BK)
        else:
            b = gl.load(W + n[:, None] * K + step * BK + k[None, :])
        a = gl.convert_layout(a, gl.DotOperandLayout(0, mma, 2))
        b = gl.convert_layout(gl.permute(b, (1, 0)), gl.DotOperandLayout(1, mma, 2))
        acc = mma_v2(a, b, acc)
    output = gl.convert_layout(acc.to(gl.bfloat16), layout)
    col = gl.program_id(1) * BN + gl.arange(0, BN, layout=gl.SliceLayout(0, layout))
    gl.store(Y + m[:, None] * N + col[None, :], output, m[:, None] < 64)


def pack(weight, bn, bk=64):
    n, k = weight.shape
    if weight.dtype != torch.bfloat16 or n % bn or k % bk or not weight.is_contiguous():
        raise ValueError("Expected aligned contiguous BF16 weights")
    exponent = (weight.view(torch.int16).to(torch.int32) >> 7) & 255
    if not bool((exponent < 247).all()):
        raise ValueError("Unsupported exponent for the reversible transform")
    output = torch.empty(
        (n // bn, k // bk, 16, bn * bk // 32), device=weight.device, dtype=torch.uint32
    )
    _pack[(n // bn, k // bk, 16)](weight.view(torch.uint16), output, k, bn, bk)
    decoded = torch.empty_like(weight)
    _unpack[(n // bn, k // bk)](output, decoded, n, k, bn, bk, num_warps=4)
    if not torch.equal(decoded.view(torch.int16), weight.view(torch.int16)):
        raise RuntimeError("Bitplane weight roundtrip changed bits")
    return output


def gemm(x, weight, n, k, config, packed=True):
    bm, bn, stages = config
    y = torch.empty((*x.shape[:-1], n), device=x.device, dtype=x.dtype)
    kernel = _gemm[(tr.cdiv(64, bm), n // bn)](
        x, weight, y, n, k, bm, bn, 64, packed, stages, num_warps=4
    )
    COMPILED[(*config, packed)] = kernel
    return y
