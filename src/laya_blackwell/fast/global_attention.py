"""Pinned native FlashAttention instances for fully occupied 64-token requests."""

import torch

from .paths import artifact_path

_loaded = False
# Query tile, key tile, warps, compile-time fixed dimensions.
CONFIGURATIONS = (
    (128, 128, 4, False),
    (128, 128, 4, True),
    (64, 128, 4, False),
    (64, 128, 4, True),
    (32, 128, 2, False),
    (32, 128, 2, True),
    (16, 128, 1, False),
    (16, 128, 1, True),
    (64, 64, 4, True),
    (32, 64, 2, True),
)


def load():
    global _loaded
    if _loaded:
        return
    torch.ops.load_library(str(artifact_path("global_attention")))

    @torch.library.register_fake("laya_fast_global_attention::forward")
    def fake(q, k, v, config):
        return torch.empty((1, 64, 16, 64), device=q.device, dtype=q.dtype).transpose(
            1, 2
        )

    _loaded = True


def attention(q, k, v, config):
    return torch.ops.laya_fast_global_attention.forward(q, k, v, config)


def install(engine, config=8, *, include_padding=True):
    """Replace eligible encoder attention before the first graph capture."""
    import types

    load()
    if include_padding:
        from .attention_special import attention as padded_attention
        from .attention_special import load as load_padded

        load_padded()
    if engine.base.graphs or engine.adapter.graphs:
        raise RuntimeError("Install global attention before capturing graphs")
    model = engine.base.model
    if hasattr(model, "original"):
        model = model.original
    elif hasattr(model, "_orig_mod"):
        model = model._orig_mod
    original = model.forward.__func__
    namespace = dict(original.__globals__)
    functional = types.SimpleNamespace(**vars(namespace["F"]))
    fallback = functional.scaled_dot_product_attention

    def selected(q, k, v, attn_mask=None):
        if q.shape == (1, 16, 64, 64) and attn_mask is None:
            return attention(q, k, v, config)
        if (
            include_padding
            and q.shape == (1, 16, 64, 64)
            and attn_mask is not None
            and attn_mask.dtype == torch.bool
            and attn_mask.shape[-2:] == (1, 64)
        ):
            return padded_attention(q, k, v, (32, True, True), attn_mask)
        return fallback(q, k, v, attn_mask=attn_mask)

    functional.scaled_dot_product_attention = selected
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
        "configuration": config,
        "include_padding": include_padding,
        "padding_configuration": [32, True, True] if include_padding else None,
        "scope": "Encoder attention only, shape=(1,16,64,64); unmasked Flash and optional broadcast-mask CUTLASS, all other shapes retain their prior path",
    }
