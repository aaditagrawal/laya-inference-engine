"""Per-engine projection fusions composed with round-one exact window attention.

Forward substitutions adapt the repository's Apache-2.0-derived model. Preserve
its NOTICE and upstream license when redistributing this experiment.
"""

import hashlib
import inspect
import textwrap
import types

from experiments.native.compiler.padded_rope import (
    attention_prepared,
    rope_qkv_padded,
    window_bias,
)
from experiments.native.kernels.candidates import gelu_lut, provenance
from laya_blackwell.kernels import rope_qkv
from laya_blackwell.model import FastDecisionModel

from .kernels import pack_qkv, pack_wi, qkv_rope, wi_geglu


def install(engine, *, wi=None, qkv=None):
    """Install configurations before any graph capture; mutate only this engine.

    A config specifies packed, lut (Wi only), and tiles mapping row thresholds to
    [BM, BN, BK, warps, stages]. Choose the first threshold >= total token rows.
    """
    if engine.base.graphs or engine.adapter.graphs:
        raise RuntimeError("Install projection fusion before capturing any graphs")
    if engine.mode != "native-window":
        raise ValueError("This adapter requires the measured native-window baseline")
    model = engine.base.model
    if not model.fp32_rope:
        raise ValueError(
            "Projection fusion currently requires Transformers 5 FP32 RoPE"
        )
    namespace = dict(model.forward.__func__.__globals__)
    if not callable(namespace.get("geglu_fn")):
        raise TypeError("Projection fusion requires the round-one exact GEGLU callback")
    source = textwrap.dedent(inspect.getsource(FastDecisionModel.forward))
    originals = []

    def replace(old, new):
        nonlocal source
        if source.count(old) != 1:
            raise RuntimeError(f"Model source changed: substitution count for {old!r}")
        originals.append(old)
        source = source.replace(old, new)

    replace(
        "window = enc.config.local_attention // 2",
        "window = enc.config.local_attention // 2\n    native_bias = window_bias(attention_mask, window) if length > 64 else None",
    )
    replace(
        "a = F.scaled_dot_product_attention(q, k, v, attn_mask=global_mask if is_global else local_mask)",
        "a = attention_prepared(q, k, v, native_bias, window) if not is_global and length > 64 else F.scaled_dot_product_attention(q, k, v, attn_mask=global_mask if is_global else local_mask)",
    )
    replace(
        "act, gate = layer.mlp.Wi(n).chunk(2, dim=-1)\n        pending = layer.mlp.Wo(F.gelu(act) * gate)",
        "pending = layer.mlp.Wo(fused_wi(layer.mlp.Wi, n))"
        if wi
        else "pending = layer.mlp.Wo(geglu_fn(layer.mlp.Wi(n)))",
    )
    original = "q, k, v = rope_qkv(qkv, cos, sin, fp32=self.fp32_rope).permute(0, 3, 2, 1, 4).unbind(2)"
    if qkv:
        replace(
            "qkv = layer.attn.Wqkv(n).view(b, length, 3, self.heads, self.dim)",
            "# Projection and rotary epilogue are fused below.",
        )
        replace(
            original,
            "q, k, v = fused_qkv(layer.attn.Wqkv, n, cos, sin, window if not is_global and length > 64 else 0)",
        )
    else:
        replace(
            original,
            "if not is_global and length > 64:\n"
            "            q, k, v = rope_qkv_padded(qkv, cos, sin, fp32=self.fp32_rope, window=window)\n"
            "        else:\n"
            "            q, k, v = rope_qkv(qkv, cos, sin, fp32=self.fp32_rope).permute(0, 3, 2, 1, 4).unbind(2)",
        )
    packed_bytes = 0
    for layer in model.net.encoder.layers:
        for cfg, linear, pack in [
            (wi, layer.mlp.Wi, pack_wi),
            (qkv, layer.attn.Wqkv, pack_qkv),
        ]:
            if cfg and cfg["packed"]:
                weight, bias = pack(linear.weight, linear.bias)
                linear.register_buffer("_fusion_weight", weight)
                linear.register_buffer("_fusion_bias", bias)
                packed_bytes += weight.numel() * weight.element_size()
                if bias is not None:
                    packed_bytes += bias.numel() * bias.element_size()
    if wi and wi.get("lut"):
        gelu_lut()

    def tile(cfg, x):
        rows = x.numel() // x.shape[-1]
        tiles = sorted(
            (int(key), tuple(value) if value is not None else None)
            for key, value in cfg["tiles"].items()
        )
        return next((value for key, value in tiles if rows <= key), tiles[-1][1])

    def weights(linear, cfg):
        return (
            (linear._fusion_weight, linear._fusion_bias)
            if cfg["packed"]
            else (linear.weight, linear.bias)
        )

    def fused_wi(linear, x):
        rows = x.numel() // x.shape[-1]
        chosen = tile(wi, x)
        if rows > wi.get("max_rows", rows) or chosen is None:
            return namespace["geglu_fn"](linear(x))
        weight, bias = weights(linear, wi)
        return wi_geglu(
            x, weight, bias, chosen, packed=wi["packed"], lut=wi.get("lut", False)
        )

    def fused_qkv(linear, x, cos, sin, padding):
        rows = x.numel() // x.shape[-1]
        chosen = tile(qkv, x)
        if rows > qkv.get("max_rows", rows) or chosen is None:
            projection = linear(x).view(*x.shape[:-1], 3, 16, 64)
            if padding:
                return rope_qkv_padded(projection, cos, sin, fp32=True, window=padding)
            return (
                rope_qkv(projection, cos, sin, fp32=True)
                .permute(0, 3, 2, 1, 4)
                .unbind(2)
            )
        weight, bias = weights(linear, qkv)
        return qkv_rope(
            x, weight, bias, cos, sin, chosen, packed=qkv["packed"], padding=padding
        )

    namespace.update(
        rope_qkv_padded=rope_qkv_padded,
        window_bias=window_bias,
        attention_prepared=attention_prepared,
        fused_wi=fused_wi,
        fused_qkv=fused_qkv,
    )
    # Only trusted repository source and fixed substitutions enter this string.
    exec(source, namespace)  # noqa: S102
    model.forward = types.MethodType(namespace["forward"], model)
    engine.fusion_metadata = {
        "wi": wi,
        "qkv": qkv,
        "packed_weight_bytes": packed_bytes,
        "forward_sha256": hashlib.sha256(source.encode()).hexdigest(),
        "kernel_provenance": provenance(),
        "semantics": "BF16 GEMM outputs, BF16 exact GELU before gating, FP32 rotary arithmetic. GEMM reduction order may differ from cuBLAS.",
    }
    return engine
