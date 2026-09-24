"""Bind the manual-offset control to one engine before capture."""

import types

import torch

from .query_shard import attention, load


def install(engine, rows=32):
    if rows not in (8, 16, 32):
        raise ValueError(rows)
    load()
    if engine.base.graphs or engine.adapter.graphs:
        raise RuntimeError("Install the query-shard experiment before graph capture")
    model = engine.base.model
    if hasattr(model, "original"):
        model = model.original
    elif hasattr(model, "_orig_mod"):
        model = model._orig_mod
    original = model.forward.__func__
    namespace = dict(original.__globals__)
    if "specialized_attention" not in namespace:
        raise RuntimeError(
            "This adapter expects retained local attention specialization"
        )

    def selected_local(q, k, v, mask=None):
        bias = None
        if mask is not None:
            bias = torch.zeros_like(mask, dtype=q.dtype).masked_fill_(
                ~mask, float("-inf")
            )
        return attention(q, k, v, rows, bias)

    namespace["specialized_attention"] = selected_local
    functional = types.SimpleNamespace(**vars(namespace["F"]))
    fallback = functional.scaled_dot_product_attention

    def selected_global(q, k, v, attn_mask=None):
        if (
            q.shape == (1, 16, 64, 64)
            and attn_mask is not None
            and attn_mask.dtype == torch.bool
            and attn_mask.shape[-2:] == (1, 64)
        ):
            bias = torch.zeros_like(attn_mask, dtype=q.dtype).masked_fill_(
                ~attn_mask, float("-inf")
            )
            return attention(q, k, v, rows, bias)
        return fallback(q, k, v, attn_mask=attn_mask)

    functional.scaled_dot_product_attention = selected_global
    namespace["F"] = functional
    forward = types.FunctionType(
        original.__code__,
        namespace,
        original.__name__,
        original.__defaults__,
        original.__closure__,
    )
    forward.__kwdefaults__ = original.__kwdefaults__
    model.forward = types.MethodType(forward, model)
    return {
        "rows": rows,
        "tile_queries": 32,
        "ctas": 16 * (64 // rows),
        "scope": "18 local calls and 10 padded global calls for shape=(1,16,64,64)",
    }
