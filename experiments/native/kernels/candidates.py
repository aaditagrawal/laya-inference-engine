"""Measured exact CUDA/Triton inference adapters for local SM120 experiments."""

import inspect
import os
import textwrap
import types
from pathlib import Path

import torch
import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice

from .build import current_library

HERE = Path(__file__).resolve().parent
LUT = None
CORRECTIONS = None
CORRECTION_LIBDEVICE = None
VECTOR_LOADED = False
NATIVE_LOADED = False
KINDS = (
    "baseline",
    "cuda_vector_norm",
    "cuda_vector_norm_triton_geglu_corrected",
    "cuda_vector_norm_adaptive_geglu",
    "cuda_norm_triton_geglu_corrected",
)


def build_vector():
    global VECTOR_LOADED
    if not VECTOR_LOADED:
        shared = current_library("vector")
        if not shared.exists():
            raise RuntimeError(
                "Run kernels/build.py --variant vector before this experiment"
            )
        torch.ops.load_library(str(shared))

        @torch.library.register_fake("laya_native_vector::norm")
        def norm_fake(x, residual, weight, bias, eps, ptx=False):
            return [torch.empty_like(x), torch.empty_like(x, dtype=torch.bfloat16)]

        VECTOR_LOADED = True


def build_native():
    global NATIVE_LOADED
    if not NATIVE_LOADED:
        shared = current_library("native")
        if not shared.exists():
            raise RuntimeError(
                "Run kernels/build.py --variant native before this experiment"
            )
        torch.ops.load_library(str(shared))

        @torch.library.register_fake("laya_native_exp::norm")
        def norm_fake(x, residual, weight, bias, eps, ptx=False):
            return [torch.empty_like(x), torch.empty_like(x, dtype=torch.bfloat16)]

        NATIVE_LOADED = True


def original_native_norm(x, residual, norm):
    return torch.ops.laya_native_exp.norm(
        x, residual, norm.weight, norm.bias, norm.eps, False
    )


def vector_norm(x, residual, norm):
    if x.shape[-1] != 1024 or x.dtype != torch.float32:
        y = x if residual is None else x + residual
        return y, norm(y).bfloat16()
    return torch.ops.laya_native_vector.norm(
        x, residual, norm.weight, norm.bias, norm.eps, False
    )


def pin_libdevice():
    """Pin the compiler math library before deriving rounding corrections.

    Inductor's emulate_precision_casts mode otherwise selects the CUDA toolkit
    libdevice after startup. A CUDA13.1 toolkit with a CUDA13.2 PyTorch wheel
    has different erf boundaries from Triton's bundled libdevice. Respect an
    existing explicit choice, then prevent an automatic mid-process switch.
    """
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
        # Every BF16 bit pattern, evaluated with the actual installed CUDA GELU.
        bits = torch.arange(65536, device="cuda", dtype=torch.int32).to(torch.int16)
        LUT = torch.nn.functional.gelu(bits.view(torch.bfloat16)).contiguous()
    return LUT


@tr.jit
def _gelu_lut(X, L, Y, D: tl.constexpr, TOTAL: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    row = i // D
    col = i % D
    a = tl.load(X + row * 2 * D + col, i < TOTAL, 0)
    b = tl.load(X + row * 2 * D + D + col, i < TOTAL, 0)
    g = tl.load(L + a.to(tl.uint16, bitcast=True).to(tl.int32)).to(tl.float32)
    tl.store(Y + i, g * b.to(tl.float32), i < TOTAL)


def triton_geglu_lut(x):
    y = torch.empty((*x.shape[:-1], x.shape[-1] // 2), device=x.device, dtype=x.dtype)
    _gelu_lut[(tr.cdiv(y.numel(), 512),)](
        x, gelu_lut(), y, y.shape[-1], y.numel(), 512, enable_fp_fusion=False
    )
    return y


def gelu_corrections():
    """Derive a complete BF16 function-domain correction, never request outputs.

    CUDA and Triton libdevice erf implementations disagree at a few BF16
    rounding boundaries. Evaluate all 65,536 encodings against the installed
    PyTorch CUDA GELU once, then specialize the fused kernel with those exact
    corrections. The fallback LUT remains available if a future compiler has
    too many boundary differences for a sparse correction to be efficient.
    """
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
                "Installed CUDA/Triton GELU differs on more than16 BF16 inputs; use exact LUT"
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


def triton_geglu_adaptive(x):
    """Use measured block/warp choices; both branches preserve BF16 results."""
    rows = x.numel() // x.shape[-1]
    if 512 <= rows <= 2048:
        y = torch.empty(
            (*x.shape[:-1], x.shape[-1] // 2), device=x.device, dtype=x.dtype
        )
        block, warps = (1024, 8) if rows >= 1024 else (512, 4)
        _gelu_lut[(tr.cdiv(y.numel(), block),)](
            x,
            gelu_lut(),
            y,
            y.shape[-1],
            y.numel(),
            block,
            num_warps=warps,
            enable_fp_fusion=False,
        )
        return y
    return triton_geglu_corrected(x)


def install(engine, kind="cuda_vector_norm_triton_geglu_corrected"):
    """Attach isolated model methods before graph capture; no global model patch."""
    import laya_blackwell.model as module

    global CORRECTIONS
    if kind not in KINDS:
        raise ValueError(kind)
    if kind == "baseline":
        engine.experimental_kernel = {"kind": kind, **provenance()}
        return engine
    if kind == "cuda_norm_triton_geglu_corrected":
        build_native()
    else:
        build_vector()
    if "geglu" in kind:
        selected = pin_libdevice()
        if CORRECTION_LIBDEVICE != selected:
            CORRECTIONS = None
        gelu_corrections()
    if "adaptive" in kind:
        gelu_lut()
    norm = (
        original_native_norm
        if kind == "cuda_norm_triton_geglu_corrected"
        else vector_norm
    )
    namespace = dict(module.__dict__, add_norm=norm)
    source = textwrap.dedent(inspect.getsource(module.FastDecisionModel.forward))
    if "geglu" in kind:
        fn = triton_geglu_adaptive if "adaptive" in kind else triton_geglu_corrected
        before = "act, gate = layer.mlp.Wi(n).chunk(2, dim=-1)\n        pending = layer.mlp.Wo(F.gelu(act) * gate)"
        after = "pending = layer.mlp.Wo(geglu_fn(layer.mlp.Wi(n)))"
        if before not in source:
            raise RuntimeError("FastDecisionModel changed; review the GEGLU adapter")
        source = source.replace(before, after)
        namespace["geglu_fn"] = fn
    exec(source, namespace)  # noqa: S102 - trusted repository model source only.
    engine.model.forward = types.MethodType(namespace["forward"], engine.model)
    head = module.FastDecisionModel._head_layer
    engine.model._head_layer = types.MethodType(
        types.FunctionType(
            head.__code__, namespace, head.__name__, head.__defaults__, head.__closure__
        ),
        engine.model,
    )
    engine.graphs.clear()
    engine.experimental_kernel = {"kind": kind, **provenance()}
    return engine


def provenance():
    """Record the math library and finite-domain correction used by this process."""
    import hashlib

    library = Path(CORRECTION_LIBDEVICE) if CORRECTION_LIBDEVICE else None
    return {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "target_architecture": "sm_120",
        "libdevice_path": str(library) if library else None,
        "libdevice_sha256": hashlib.sha256(library.read_bytes()).hexdigest()
        if library
        else None,
        "bf16_gelu_corrections": list(zip(*CORRECTIONS)) if CORRECTIONS else [],
        "source_sha256": {
            name: hashlib.sha256((HERE / name).read_bytes()).hexdigest()
            for name in ("native.cu", "native_vector.cu")
        },
    }
