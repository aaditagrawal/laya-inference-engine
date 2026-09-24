"""Exact bidirectional windows using ATen's existing CUTLASS attention kernel.

The source transformation adapts the Apache-2.0-derived FastDecisionModel;
retain the repository NOTICE and upstream license when redistributing.
"""

import inspect
import textwrap
import types

import torch

from laya_blackwell.model import FastDecisionModel

from .padded_rope import attention_prepared, rope_qkv_padded, window_bias


def install_window(engine):
    """Install after optional native kernels and before compilation/capture.

    S query tokens and S+64 key tokens shift bottom-right causality to k<=q+64.
    A 129-token causal window adds k>=q-64. The appended keys are always masked,
    which exactly represents the checkpoint's bidirectional inclusive window.
    """
    if engine.graphs:
        raise RuntimeError("Install window attention before capturing any CUDA Graphs")
    if torch.__version__.split("+")[0].split(".")[:2] != ["2", "14"]:
        raise RuntimeError(
            "The private ATen attention operation is validated with PyTorch 2.14 only"
        )
    model = engine.model
    if hasattr(model, "_orig_mod"):
        raise RuntimeError("Install window attention before torch.compile")
    if model.net.encoder.config.local_attention != 128:
        raise ValueError(
            "This experiment is validated for the checkpoint local_attention=128"
        )
    if model.net.encoder.layers[0].attn.Wqkv.weight.dtype != torch.bfloat16:
        raise ValueError("This experiment requires BF16 matrix weights")
    namespace = dict(model.forward.__func__.__globals__)
    source = textwrap.dedent(inspect.getsource(FastDecisionModel.forward))
    if "geglu_fn" in namespace:
        source = source.replace(
            "act, gate = layer.mlp.Wi(n).chunk(2, dim=-1)\n        pending = layer.mlp.Wo(F.gelu(act) * gate)",
            "pending = layer.mlp.Wo(geglu_fn(layer.mlp.Wi(n)))",
        )
    source = source.replace(
        "window = enc.config.local_attention // 2",
        "window = enc.config.local_attention // 2\n    native_bias = window_bias(attention_mask, window) if length > 64 else None",
    )
    original = "q, k, v = rope_qkv(qkv, cos, sin, fp32=self.fp32_rope).permute(0, 3, 2, 1, 4).unbind(2)"
    replacement = (
        "if not is_global and length > 64:\n"
        "            q, k, v = rope_qkv_padded(qkv, cos, sin, fp32=self.fp32_rope, window=window)\n"
        "        else:\n"
        "            q, k, v = rope_qkv(qkv, cos, sin, fp32=self.fp32_rope).permute(0, 3, 2, 1, 4).unbind(2)"
    )
    if original not in source:
        raise RuntimeError(
            "FastDecisionModel changed; review the experimental source transformation"
        )
    source = source.replace(original, replacement)
    source = source.replace(
        "a = F.scaled_dot_product_attention(q, k, v, attn_mask=global_mask if is_global else local_mask)",
        "a = attention_prepared(q, k, v, native_bias, window) if not is_global and length > 64 else F.scaled_dot_product_attention(q, k, v, attn_mask=global_mask if is_global else local_mask)",
    )
    namespace.update(
        rope_qkv_padded=rope_qkv_padded,
        window_bias=window_bias,
        attention_prepared=attention_prepared,
    )
    # Only inspected repository code and fixed substitutions enter this string.
    exec(source, namespace)  # noqa: S102
    model.forward = types.MethodType(namespace["forward"], model)
    return engine
