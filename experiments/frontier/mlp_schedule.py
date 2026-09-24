"""Exact direct-weight GEGLU with alternative tile and accumulator schedules."""

import torch
import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice

from experiments.native.kernels.candidates import gelu_corrections

COMPILED = {}


@tr.jit
def _gelu(x, IN_BITS: tl.constexpr, OUT_BITS: tl.constexpr):
    bits = x.to(tl.uint16, bitcast=True)
    f = x.to(tl.float32)
    value = (0.5 * f * (1.0 + libdevice.erf(f * 0.7071067811865476))).to(tl.bfloat16)
    for i in tl.static_range(len(IN_BITS)):
        value = tl.where(
            bits == IN_BITS[i],
            tl.full((), OUT_BITS[i], tl.uint16).to(tl.bfloat16, bitcast=True),
            value,
        )
    return value


@tr.jit
def _project(
    X,
    W,
    Y,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    DUAL: tl.constexpr,
    CACHE: tl.constexpr,
    IN_BITS: tl.constexpr,
    OUT_BITS: tl.constexpr,
):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    kk = tl.arange(0, BK)
    cm: tl.constexpr = ".cg" if CACHE == 1 else ""
    ep: tl.constexpr = "evict_first" if CACHE == 2 else ""
    if DUAL:
        n = tl.program_id(1) * (BN // 2) + tl.arange(0, BN // 2)
        act = tl.full((BM, BN // 2), 0, tl.float32)
        gate = tl.full((BM, BN // 2), 0, tl.float32)
        for start in range(0, 1024, BK):
            k = start + kk
            a = tl.load(X + m[:, None] * 1024 + k[None, :], m[:, None] < 64, 0)
            wa = tl.load(
                W + n[None, :] * 1024 + k[:, None],
                n[None, :] < 2624,
                0,
                cache_modifier=cm,
                eviction_policy=ep,
            )
            wg = tl.load(
                W + (n[None, :] + 2624) * 1024 + k[:, None],
                n[None, :] < 2624,
                0,
                cache_modifier=cm,
                eviction_policy=ep,
            )
            act = tl.dot(a, wa, act)
            gate = tl.dot(a, wg, gate)
        activation = act.to(tl.bfloat16)
        rounded_gate = gate.to(tl.bfloat16)
    else:
        nfull = tl.program_id(1) * BN + tl.arange(0, BN)
        wn = nfull // 2 + (nfull % 2) * 2624
        acc = tl.full((BM, BN), 0, tl.float32)
        for start in range(0, 1024, BK):
            k = start + kk
            a = tl.load(X + m[:, None] * 1024 + k[None, :], m[:, None] < 64, 0)
            w = tl.load(
                W + wn[None, :] * 1024 + k[:, None],
                nfull[None, :] < 5248,
                0,
                cache_modifier=cm,
                eviction_policy=ep,
            )
            acc = tl.dot(a, w, acc)
        activation, rounded_gate = tl.split(
            tl.reshape(acc.to(tl.bfloat16), (BM, BN // 2, 2))
        )
        n = tl.program_id(1) * (BN // 2) + tl.arange(0, BN // 2)
    result = _gelu(activation, IN_BITS, OUT_BITS).to(tl.float32) * rounded_gate.to(
        tl.float32
    )
    tl.store(
        Y + m[:, None] * 2624 + n[None, :],
        result,
        (m[:, None] < 64) & (n[None, :] < 2624),
    )


def project(x, weight, config):
    bm, bn, bk, warps, stages, dual, cache = config
    if x.numel() != 64 * 1024 or x.shape[-1] != 1024 or weight.shape != (5248, 1024):
        raise ValueError("Expected 64x1024 activation and 5248x1024 weights")
    if not (
        x.is_cuda
        and x.device == weight.device
        and x.dtype == weight.dtype == torch.bfloat16
    ):
        raise ValueError("Expected BF16 tensors on the same CUDA device")
    if not x.is_contiguous() or not weight.is_contiguous():
        raise ValueError("Expected contiguous tensors")
    before, after = gelu_corrections()
    output = torch.empty((*x.shape[:-1], 2624), device=x.device, dtype=x.dtype)
    kernel = _project[(tr.cdiv(64, bm), tr.cdiv(2624, bn // 2))](
        x,
        weight,
        output,
        bm,
        bn,
        bk,
        dual,
        cache,
        before,
        after,
        num_warps=warps,
        num_stages=stages,
        enable_fp_fusion=False,
    )
    COMPILED[tuple(config)] = kernel
    return output


@torch.library.custom_op(
    "laya_frontier_mlp_schedule::geglu", mutates_args=(), device_types="cuda"
)
def compiled_project(
    x: torch.Tensor, weight: torch.Tensor, config: list[int]
) -> torch.Tensor:
    return project(x, weight, config)


@compiled_project.register_fake
def _(x, weight, config):
    return torch.empty((*x.shape[:-1], 2624), device=x.device, dtype=x.dtype)
