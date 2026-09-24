"""Exact BF16 MLP input projection and GEGLU in one kernel."""

import inspect
import json
import linecache
import textwrap
import types
from pathlib import Path

import torch
import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice
from triton.tools.tensor_descriptor import TensorDescriptor

from experiments.native.kernels.candidates import gelu_corrections, gelu_lut

COMPILED = {}


@tr.jit
def _project(
    X,
    W,
    LUT,
    Y,
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
            b = tl.load(W + wn[None, :] * 1024 + k[:, None], n[None, :] < 5248, 0)
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


def pack(weight):
    if weight.shape != (5248, 1024) or weight.dtype != torch.bfloat16:
        raise ValueError("Expected BF16 5248x1024 MLP input weight")
    packed = (
        weight.reshape(2, 2624, 1024).transpose(0, 1).contiguous().reshape(5248, 1024)
    )
    restored = (
        packed.reshape(2624, 2, 1024).transpose(0, 1).contiguous().reshape_as(weight)
    )
    if not torch.equal(restored.view(torch.int16), weight.view(torch.int16)):
        raise RuntimeError("Interleaved weight roundtrip changed bits")
    return packed


def project(x, weight, config):
    bm, bn, bk, warps, stages, access, lookup = config
    if x.numel() != 64 * 1024 or x.shape[-1] != 1024 or weight.shape != (5248, 1024):
        raise ValueError("Expected 64x1024 input and packed 5248x1024 weight")
    if (
        x.dtype != torch.bfloat16
        or weight.dtype != x.dtype
        or x.device != weight.device
    ):
        raise ValueError("Expected BF16 tensors on the same CUDA device")
    if not x.is_cuda or not x.is_contiguous() or not weight.is_contiguous():
        raise ValueError("Expected contiguous CUDA tensors")
    inputs, outputs = gelu_corrections()
    table = gelu_lut() if lookup else weight
    xd = TensorDescriptor(x, [64, 1024], [1024, 1], [bm, bk]) if access == 1 else x
    wd = (
        TensorDescriptor(weight, [5248, 1024], [1024, 1], [bn, bk])
        if access == 1
        else weight
    )
    output = torch.empty((*x.shape[:-1], 2624), device=x.device, dtype=x.dtype)
    kernel = _project[(tr.cdiv(64, bm), tr.cdiv(5248, bn))](
        xd,
        wd,
        table,
        output,
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
    if not torch.compiler.is_compiling():
        COMPILED[tuple(config)] = kernel
    return output


@torch.library.custom_op(
    "laya_frontier_mlp::geglu", mutates_args=(), device_types="cuda"
)
def compiled_project(
    x: torch.Tensor, weight: torch.Tensor, config: list[int]
) -> torch.Tensor:
    return project(x, weight, config)


@compiled_project.register_fake
def _(x, weight, config):
    return torch.empty((*x.shape[:-1], 2624), device=x.device, dtype=x.dtype)


@torch.inference_mode()
def install(model, *, unpacked=False):
    filename = "mlp-geglu-unpacked.json" if unpacked else "mlp-geglu.json"
    report = json.loads(
        (
            Path(__file__).resolve().parents[2] / "results/frontier" / filename
        ).read_text()
    )
    choices = [
        r
        for r in report["rows"]
        if r.get("mismatches") == 0 and r.get("speedup", 0) > 1.03
    ]
    if not choices:
        raise RuntimeError(
            "No exact fused MLP configuration passed the screening threshold"
        )
    selected = max(choices, key=lambda r: r["speedup"])
    config = [int(v) for v in selected["config"]]
    total = 0
    for layer in model.net.encoder.layers:
        module = layer.mlp.Wi
        if module.bias is not None:
            raise ValueError(
                "Fused MLP probe only supports bias-free input projections"
            )
        if not unpacked:
            module.register_buffer("frontier_geglu_weight", pack(module.weight))
            total += module.frontier_geglu_weight.nbytes
    gelu_corrections()
    if config[-1]:
        gelu_lut()
    original = model.forward.__func__
    source = textwrap.dedent(inspect.getsource(original))
    before = "geglu_fn(layer.mlp.Wi(n))"
    if source.count(before) != 1:
        raise RuntimeError("Forward changed; revalidate fused MLP installation")
    source = source.replace(before, "frontier_mlp_geglu(n, layer.mlp.Wi)")
    namespace = dict(original.__globals__)
    baseline_geglu = namespace["geglu_fn"]

    def selected_geglu(x, module):
        if x.numel() == 64 * 1024 and x.is_contiguous():
            weight = module.weight if unpacked else module.frontier_geglu_weight
            return compiled_project(x, weight, config)
        return baseline_geglu(module(x))

    namespace["frontier_mlp_geglu"] = selected_geglu
    filename = str(__file__) + ":generated-forward"
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    exec(compile(source, filename, "exec"), namespace)  # noqa: S102 - inspected source and fixed substitution.
    model.forward = types.MethodType(namespace[original.__name__], model)
    return {
        "config": config,
        "additional_weight_bytes": total,
        "unpacked": unpacked,
        "screening_speedup": selected["speedup"],
    }
