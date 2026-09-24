"""Native attention dispatch for the validated short-request shape."""

import types

from torch.nn import functional as F

from .native_attention import attention as native_attention
from .native_attention import load


def install(model):
    load()
    original = model.forward.__func__
    namespace = dict(original.__globals__)
    functional = types.SimpleNamespace(**vars(namespace["F"]))
    baseline = F.scaled_dot_product_attention

    def selected(q, k, v, attn_mask=None):
        if (
            q.shape == (1, 16, 64, 64)
            and attn_mask is not None
            and attn_mask.shape[-2] == 64
        ):
            return native_attention(q, k, v, attn_mask, query_tile=32)
        return baseline(q, k, v, attn_mask=attn_mask)

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
