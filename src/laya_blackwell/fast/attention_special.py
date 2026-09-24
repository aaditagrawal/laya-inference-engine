"""Fixed-shape attention specializations with exact padding-aware fallback."""

import inspect
import linecache
import textwrap
import types

import torch

from .paths import artifact_path

_loaded = False


def load():
    global _loaded
    if _loaded:
        return
    torch.ops.load_library(str(artifact_path("attention_special")))

    @torch.library.register_fake("laya_fast_attention_special::forward")
    def fake(q, k, v, query_tile, bias_support, special, bias=None):
        return torch.empty((1, 64, 16, 64), device=q.device, dtype=q.dtype).transpose(
            1, 2
        )

    _loaded = True


def attention(q, k, v, config, mask=None):
    bias = None
    if mask is not None:
        bias = torch.zeros_like(mask, dtype=q.dtype).masked_fill_(~mask, float("-inf"))
    return torch.ops.laya_fast_attention_special.forward(q, k, v, *config, bias)


def install(engine, config=(32, True, True), *, include_padding=False):
    """Install before capture, omitting bias only when every key is visible."""
    load()
    if engine.base.graphs or engine.adapter.graphs:
        raise RuntimeError("Install attention before capturing any graph")
    model = engine.base.model
    if hasattr(model, "original"):
        model = model.original
    elif hasattr(model, "_orig_mod"):
        model = model._orig_mod
    original = model.forward.__func__
    source = textwrap.dedent(inspect.getsource(original))
    before = "a = F.scaled_dot_product_attention(q, k, v, attn_mask=global_mask if is_global else local_mask)"
    after = "a = specialized_attention(q, k, v) if not is_global and global_attention_unmasked and b == 1 and length == 64 and window >= 63 else F.scaled_dot_product_attention(q, k, v, attn_mask=global_mask if is_global else local_mask)"
    if source.count(before) != 1:
        raise RuntimeError("Model source changed; inspect specialization replacement")
    if include_padding:
        after = "a = specialized_attention(q, k, v, None if global_attention_unmasked else local_mask) if not is_global and b == 1 and length == 64 and window >= 63 else F.scaled_dot_product_attention(q, k, v, attn_mask=global_mask if is_global else local_mask)"
        if not config[1]:
            raise ValueError("Padding requires bias-support arithmetic")
    source = source.replace(before, after)
    namespace = dict(original.__globals__)
    namespace["specialized_attention"] = lambda q, k, v, mask=None: attention(
        q, k, v, config, mask
    )
    filename = str(__file__) + ":generated-forward"
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    exec(compile(source, filename, "exec"), namespace)  # noqa: S102 - fixed substitution of inspected local source
    model.forward = types.MethodType(namespace[original.__name__], model)
    return {
        "configuration": config,
        "include_padding": include_padding,
        "scope": "Local attention, batch=1, length=64, window>=63; real bias retained for padding when enabled",
    }
