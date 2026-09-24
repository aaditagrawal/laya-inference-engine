"""Experimental final-head selected-Q adapter; install before graph capture."""

import types

from .head_q_select_kernel import compiled_project


def replacement(original, bm=32):
    namespace = original.__func__.__globals__
    add_norm = namespace["add_norm"]
    functional = namespace["F"]
    head_relu = namespace["head_relu"]

    def selected(self, h, layer, mask, select=None):
        if select is None or h.shape != (1, 64, 1024) or select.shape[1] > 32:
            return original(h, layer, mask, select)
        _, n = add_norm(h, None, layer.norm1)
        q, packed = compiled_project(
            n, layer.self_attn.in_proj_weight, layer.self_attn.in_proj_bias, select, bm
        )
        _, k, v = packed.view(1, 64, 3, 16, 64).permute(0, 3, 2, 1, 4).unbind(2)
        h = h.gather(1, select[:, :, None].expand(-1, -1, 1024))
        a = functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        a = a.transpose(1, 2).reshape(1, -1, 1024)
        h, n = add_norm(h, layer.self_attn.out_proj(a), layer.norm2)
        return h + layer.linear2(head_relu(layer.linear1, n))

    return selected


def install(engine, bm=32):
    if engine.base.graphs or engine.adapter.graphs:
        raise RuntimeError("Install selected-Q before graph capture")
    model = engine.base.model
    if hasattr(model, "original"):
        model = model.original
    elif hasattr(model, "_orig_mod"):
        model = model._orig_mod
    model._head_layer = types.MethodType(replacement(model._head_layer, bm), model)
    return {
        "bm": bm,
        "scope": "Final head only; batch=1, padded length=64, selected rows<=32; full K/V retained",
    }
