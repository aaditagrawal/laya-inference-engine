"""Lossless row-pitch variants of retained projection kernels.

Cloned arithmetic from mlp_geglu.py, tma.py and matmul.py. Only weight-row
addressing changes. Baseline source hashes are recorded by the screen.
"""

import hashlib

import torch
import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice
from triton.tools.tensor_descriptor import TensorDescriptor

from experiments.native.kernels.candidates import gelu_corrections, gelu_lut

from .matmul import _reduce

COMPILED = {}


@tr.jit
def _project_pitch(
    X,
    W,
    LUT,
    Y,
    PITCH: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    ACCESS: tl.constexpr,
    LOOKUP: tl.constexpr,
    IN_BITS: tl.constexpr,
    OUT_BITS: tl.constexpr,
):
    m0, n0 = tl.program_id(0) * BM, tl.program_id(1) * BN
    m, n = m0 + tl.arange(0, BM), n0 + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for start in range(0, 1024, BK):
        if ACCESS == 1:
            a = X.load([m0, start])
            b = W.load([n0, start]).T
        else:
            k = start + kk
            wn = n // 2 + (n % 2) * 2624 if ACCESS == 2 else n
            a = tl.load(X + m[:, None] * 1024 + k[None, :], m[:, None] < 64, 0)
            b = tl.load(W + wn[None, :] * PITCH + k[:, None], n[None, :] < 5248, 0)
        acc = tl.dot(a, b, acc)
    # Adjacent output columns contain each activation/gate pair. Preserve
    # the projection and GELU BF16 roundings before their final product.
    rounded = acc.to(tl.bfloat16)
    activation, gate = tl.split(tl.reshape(rounded, (BM, BN // 2, 2)))
    bits = activation.to(tl.uint16, bitcast=True)
    if LOOKUP:
        g = tl.load(LUT + bits.to(tl.int32))
    else:
        af = activation.to(tl.float32)
        g = (0.5 * af * (1.0 + libdevice.erf(af * 0.7071067811865476))).to(tl.bfloat16)
        for i in tl.static_range(len(IN_BITS)):
            g = tl.where(
                bits == IN_BITS[i],
                tl.full((), OUT_BITS[i], tl.uint16).to(tl.bfloat16, bitcast=True),
                g,
            )
    col = n0 // 2 + tl.arange(0, BN // 2)
    tl.store(
        Y + m[:, None] * 2624 + col[None, :],
        g.to(tl.float32) * gate.to(tl.float32),
        (m[:, None] < 64) & (col[None, :] < 2624),
    )


@tr.jit
def _tma_pitch(
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


@tr.jit
def _matmul_pitch(
    X,
    W,
    SCALE,
    PARTIAL,
    Y,
    PITCH: tl.constexpr,
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
            W + n[None, :] * PITCH + k[:, None],
            (n[None, :] < N) & (k[:, None] < K),
            0.0,
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


def padded_copy(weight, padding):
    if padding < 0 or padding % 8:
        raise ValueError("Padding must preserve 16-byte TMA stride alignment")
    n, k = weight.shape
    storage = torch.zeros((n, k + padding), device=weight.device, dtype=weight.dtype)
    storage[:, :k].copy_(weight)
    view = storage[:, :k]
    if not torch.equal(view.view(torch.int16), weight.view(torch.int16)):
        raise RuntimeError("Weight padding changed bits")
    if padding and not bool((storage[:, k:] == 0).all()):
        raise RuntimeError("Unused weight storage must be zero initialized")
    return view


def project(x, weight, field, config, *, return_partials=False):
    if not x.is_contiguous() or weight.stride(1) != 1:
        raise ValueError("Expected contiguous input and contiguous weight columns")
    if (
        x.dtype != torch.bfloat16
        or weight.dtype != x.dtype
        or weight.device != x.device
    ):
        raise ValueError("Expected matching BF16 CUDA input and weights")
    n, k = weight.shape
    m = x.numel() // k
    pitch = weight.stride(0)
    if m != 64 or x.shape[-1] != k:
        raise ValueError("Only retained 64-row shapes are supported")
    if field == "mlp.Wi":
        bm, bn, bk, warps, stages, access, lookup = config
        if access != 2 or (n, k) != (5248, 1024):
            raise ValueError("Expected retained unpacked MLP configuration")
        inputs, outputs = gelu_corrections()
        table = gelu_lut() if lookup else weight
        y = torch.empty((*x.shape[:-1], 2624), device=x.device, dtype=x.dtype)
        kernel = _project_pitch[(tr.cdiv(64, bm), tr.cdiv(5248, bn))](
            x,
            weight,
            table,
            y,
            pitch,
            bm,
            bn,
            bk,
            access,
            lookup,
            inputs,
            outputs,
            num_warps=warps,
            num_stages=stages,
            enable_fp_fusion=False,
        )
    else:
        bm, bn, bk, split, warps, stages = config[:6]
        y = torch.empty((*x.shape[:-1], n), device=x.device, dtype=x.dtype)
        partial = (
            torch.empty(
                (split, m, n),
                device=x.device,
                dtype=torch.bfloat16 if field == "mlp.Wo" else torch.float32,
            )
            if split > 1
            else y
        )
        if field == "attn.Wqkv":
            xd = TensorDescriptor(x, [m, k], [k, 1], [bm, bk])
            wd = TensorDescriptor(weight, [n, k], [pitch, 1], [bn, bk])
            kernel = _tma_pitch[(tr.cdiv(m, bm), tr.cdiv(n, bn), split)](
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
                config[6],
                num_warps=warps,
                num_stages=stages,
            )
        else:
            kernel = _matmul_pitch[(tr.cdiv(m, bm), tr.cdiv(n, bn), split)](
                x,
                weight,
                weight,
                partial,
                y,
                pitch,
                m,
                n,
                k,
                bm,
                bn,
                bk,
                split,
                0,
                64,
                num_warps=warps,
                num_stages=stages,
            )
        if split > 1 and not return_partials:
            _reduce[(tr.cdiv(m * n, 512),)](partial, y, m * n, split, 512)
        if return_partials:
            y = partial.view(split, *x.shape[:-1], n)
    COMPILED[(field, pitch, return_partials)] = kernel
    return y


def binary_hashes():
    return {
        str(key): {
            name: hashlib.sha256(
                value if isinstance(value, bytes) else value.encode()
            ).hexdigest()
            for name, value in kernel.asm.items()
            if name in ("cubin", "ptx")
        }
        for key, kernel in sorted(COMPILED.items())
    }
