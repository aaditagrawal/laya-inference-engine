"""CUDA Welford normalization and exact BF16 GELU correction.

The CUDA kernel retains PyTorch's reduction order. The sparse correction is
derived from every BF16 input encoding against the installed CUDA GELU.
"""

import inspect
import os
import textwrap
import types
from pathlib import Path

import torch
import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice

from .paths import artifact_path

LUT = None
CORRECTIONS = None
CORRECTION_LIBDEVICE = None
_loaded = False


def load():
    global _loaded
    if _loaded:
        return
    torch.ops.load_library(str(artifact_path("vector")))

    @torch.library.register_fake("laya_fast_vector::norm")
    def norm_fake(x, residual, weight, bias, eps, ptx=False):
        return [torch.empty_like(x), torch.empty_like(x, dtype=torch.bfloat16)]

    _loaded = True


def vector_norm(x, residual, norm):
    if x.shape[-1] != 1024 or x.dtype != torch.float32:
        y = x if residual is None else x + residual
        return y, norm(y).bfloat16()
    return torch.ops.laya_fast_vector.norm(
        x, residual, norm.weight, norm.bias, norm.eps, False
    )


def pin_libdevice():
    """Freeze compiler math before deriving the finite-domain correction."""
    from triton import knobs
    from triton.backends.nvidia import compiler

    selected = os.environ.get("TRITON_LIBDEVICE_PATH") or knobs.nvidia.libdevice_path
    selected = str(selected or Path(compiler.__file__).parent / "lib/libdevice.10.bc")
    os.environ["TRITON_LIBDEVICE_PATH"] = selected
    knobs.nvidia.libdevice_path = selected
    return selected


def gelu_lut():
    global LUT
    if LUT is None:
        bits = torch.arange(65536, device="cuda", dtype=torch.int32).to(torch.int16)
        LUT = torch.nn.functional.gelu(bits.view(torch.bfloat16)).contiguous()
    return LUT


def gelu_corrections():
    global CORRECTIONS, CORRECTION_LIBDEVICE
    if CORRECTIONS is None:
        CORRECTION_LIBDEVICE = pin_libdevice()
        from laya_blackwell.kernels import geglu

        a = (
            torch.arange(65536, device="cuda", dtype=torch.int32)
            .to(torch.int16)
            .view(torch.bfloat16)
            .view(1, -1)
        )
        expected = torch.nn.functional.gelu(a)
        actual = geglu(torch.cat((a, torch.ones_like(a)), dim=-1).contiguous())
        bad = torch.isfinite(a) & ~(
            (actual == expected) | (torch.isnan(actual) & torch.isnan(expected))
        )
        indices = bad.nonzero()[:, 1]
        outputs = expected.view(torch.int16)[0, indices]
        CORRECTIONS = (
            tuple(int(i) for i in indices.cpu()),
            tuple(int(i) & 65535 for i in outputs.cpu()),
        )
        if len(CORRECTIONS[0]) > 16:
            raise RuntimeError(
                "Installed CUDA/Triton GELU exceeds the validated correction budget"
            )
    return CORRECTIONS


@tr.jit
def _gelu_corrected(
    X,
    Y,
    D: tl.constexpr,
    TOTAL: tl.constexpr,
    IN_BITS: tl.constexpr,
    OUT_BITS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = i // D
    col = i % D
    a = tl.load(X + row * 2 * D + col, i < TOTAL, 0)
    b = tl.load(X + row * 2 * D + D + col, i < TOTAL, 0).to(tl.float32)
    af = a.to(tl.float32)
    g = (0.5 * af * (1.0 + libdevice.erf(af * 0.7071067811865476))).to(tl.bfloat16)
    bits = a.to(tl.uint16, bitcast=True)
    for k in tl.static_range(len(IN_BITS)):
        g = tl.where(
            bits == IN_BITS[k],
            tl.full((), OUT_BITS[k], tl.uint16).to(tl.bfloat16, bitcast=True),
            g,
        )
    tl.store(Y + i, g.to(tl.float32) * b, i < TOTAL)


def triton_geglu_corrected(x):
    inputs, outputs = gelu_corrections()
    y = torch.empty((*x.shape[:-1], x.shape[-1] // 2), device=x.device, dtype=x.dtype)
    _gelu_corrected[(tr.cdiv(y.numel(), 512),)](
        x, y, y.shape[-1], y.numel(), inputs, outputs, 512, enable_fp_fusion=False
    )
    return y


def install(engine):
    """Bind isolated model methods before compilation and graph capture."""
    import laya_blackwell.model as module

    global CORRECTIONS
    load()
    selected = pin_libdevice()
    if CORRECTION_LIBDEVICE != selected:
        CORRECTIONS = None
    gelu_corrections()
    namespace = dict(module.__dict__, add_norm=vector_norm)
    source = textwrap.dedent(inspect.getsource(module.FastDecisionModel.forward))
    before = "act, gate = layer.mlp.Wi(n).chunk(2, dim=-1)\n        pending = layer.mlp.Wo(F.gelu(act) * gate)"
    after = "pending = layer.mlp.Wo(geglu_fn(layer.mlp.Wi(n)))"
    if before not in source:
        raise RuntimeError("FastDecisionModel changed; revalidate the GELU adapter")
    source = source.replace(before, after)
    namespace["geglu_fn"] = triton_geglu_corrected
    exec(source, namespace)  # noqa: S102 - inspected package source and fixed substitutions.
    engine.model.forward = types.MethodType(namespace["forward"], engine.model)
    head = module.FastDecisionModel._head_layer
    engine.model._head_layer = types.MethodType(
        types.FunctionType(
            head.__code__, namespace, head.__name__, head.__defaults__, head.__closure__
        ),
        engine.model,
    )
    return engine
