"""Use existing pointwise kernels to prefetch the next projection's weights."""

import inspect
import linecache
import textwrap
import types

import torch
import triton as tr
import triton.language as tl

from experiments.native.kernels.candidates import _gelu_corrected, gelu_corrections
from laya_blackwell.kernels import _rope

COMPILED = {}


@tr.jit
def _hint(W, BYTES: tl.constexpr, CHUNK: tl.constexpr):
    offset = tl.program_id(0) * CHUNK
    size = tl.minimum(CHUNK, tl.maximum(0, BYTES - offset))
    pointer = W.to(tl.pointer_type(tl.uint8)) + offset
    tl.inline_asm_elementwise(
        """{
            .reg .u32 tid;
            .reg .pred leader, valid, issue;
            mov.u32 tid, %tid.x;
            setp.eq.u32 leader, tid, 0;
            setp.gt.u32 valid, $2, 0;
            and.pred issue, leader, valid;
            @issue cp.async.bulk.prefetch.L2.global [$1], $2;
            mov.u32 $0, 0;
        }""",
        constraints="=r,l,r",
        args=[pointer, size],
        dtype=tl.int32,
        is_pure=False,
        pack=1,
    )


@tr.jit
def _rope_hint(
    X,
    C,
    S,
    Y,
    W,
    L: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    TOTAL: tl.constexpr,
    BLOCK: tl.constexpr,
    FP32: tl.constexpr,
    BYTES: tl.constexpr,
    CHUNK: tl.constexpr,
):
    _hint(W, BYTES, CHUNK)
    _rope(X, C, S, Y, L, H, D, TOTAL, BLOCK, FP32)


@tr.jit
def _gelu_hint(
    X,
    Y,
    W,
    D: tl.constexpr,
    TOTAL: tl.constexpr,
    IN_BITS: tl.constexpr,
    OUT_BITS: tl.constexpr,
    BLOCK: tl.constexpr,
    BYTES: tl.constexpr,
    CHUNK: tl.constexpr,
):
    _hint(W, BYTES, CHUNK)
    _gelu_corrected(X, Y, D, TOTAL, IN_BITS, OUT_BITS, BLOCK)


@torch.library.custom_op(
    "laya_frontier_prefetch::rope", mutates_args=(), device_types="cuda"
)
def rope(
    qkv: torch.Tensor,
    cosine: torch.Tensor,
    sine: torch.Tensor,
    weight: torch.Tensor,
    fp32: bool,
    chunk: int,
) -> torch.Tensor:
    output = torch.empty_like(qkv)
    _, length, _, heads, dim = qkv.shape
    kernel = _rope_hint[(tr.cdiv(qkv.numel(), 512),)](
        qkv,
        cosine,
        sine,
        output,
        weight,
        length,
        heads,
        dim,
        qkv.numel(),
        512,
        fp32,
        weight.numel() * weight.element_size(),
        chunk,
        enable_fp_fusion=False,
    )
    if not torch.compiler.is_compiling():
        COMPILED["rope"] = kernel
    return output


@rope.register_fake
def _(qkv, cosine, sine, weight, fp32, chunk):
    return torch.empty_like(qkv)


@torch.library.custom_op(
    "laya_frontier_prefetch::geglu", mutates_args=(), device_types="cuda"
)
def geglu(x: torch.Tensor, weight: torch.Tensor, chunk: int) -> torch.Tensor:
    inputs, outputs = gelu_corrections()
    y = torch.empty((*x.shape[:-1], x.shape[-1] // 2), device=x.device, dtype=x.dtype)
    kernel = _gelu_hint[(tr.cdiv(y.numel(), 512),)](
        x,
        y,
        weight,
        y.shape[-1],
        y.numel(),
        inputs,
        outputs,
        512,
        weight.numel() * weight.element_size(),
        chunk,
        enable_fp_fusion=False,
    )
    if not torch.compiler.is_compiling():
        COMPILED["geglu"] = kernel
    return y


@geglu.register_fake
def _(x, weight, chunk):
    return torch.empty(
        (*x.shape[:-1], x.shape[-1] // 2), device=x.device, dtype=x.dtype
    )


def install(model, mode, chunk):
    if mode not in {"rope", "geglu", "both"} or chunk <= 0 or chunk % 16:
        raise ValueError(
            "Expected rope/geglu/both mode and a positive, 16-byte-aligned chunk"
        )
    original = model.forward.__func__
    source = textwrap.dedent(inspect.getsource(original))
    namespace = dict(original.__globals__)
    changes = {}
    if mode in {"rope", "both"}:
        changes["rope_qkv(qkv, cos, sin, fp32=self.fp32_rope)"] = (
            "frontier_rope(qkv, cos, sin, layer.attn.Wo.weight, self.fp32_rope)"
        )

        def selected_rope(qkv, cos, sin, w, fp32):
            return rope(qkv, cos, sin, w, fp32, chunk)

        namespace["frontier_rope"] = selected_rope
    if mode in {"geglu", "both"}:
        changes["geglu_fn(layer.mlp.Wi(n))"] = (
            "frontier_geglu(layer.mlp.Wi(n), layer.mlp.Wo.weight)"
        )

        def selected_geglu(x, w):
            return geglu(x, w, chunk)

        namespace["frontier_geglu"] = selected_geglu
    for old, new in changes.items():
        if source.count(old) != 1:
            raise RuntimeError(
                "Forward source changed; revalidate prefetch installation"
            )
        source = source.replace(old, new)
    filename = str(__file__) + ":generated-" + mode + "-" + str(chunk)
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    exec(compile(source, filename, "exec"), namespace)  # noqa: S102 - inspected repository source and fixed substitutions.
    model.forward = types.MethodType(namespace[original.__name__], model)
