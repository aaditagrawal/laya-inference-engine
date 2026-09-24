"""Keep installed PyTorch normalization and GELU numerics during compilation."""

import types

import torch
from torch.nn import functional as F


@torch.library.custom_op(
    "laya_fast_compiler::layer_norm", mutates_args=(), device_types="cuda"
)
def layer_norm(
    x: torch.Tensor,
    shape: list[int],
    weight: torch.Tensor | None,
    bias: torch.Tensor | None,
    eps: float,
) -> torch.Tensor:
    return F.layer_norm(x, shape, weight, bias, eps)


@layer_norm.register_fake
def _(x, shape, weight, bias, eps):
    return torch.empty_like(x)


@torch.library.custom_op(
    "laya_fast_compiler::gelu", mutates_args=(), device_types="cuda"
)
def gelu(x: torch.Tensor, approximate: str) -> torch.Tensor:
    return F.gelu(x, approximate=approximate)


@gelu.register_fake
def _(x, approximate):
    return torch.empty_like(x)


@torch.library.custom_op(
    "laya_fast_compiler::attention", mutates_args=(), device_types="cuda"
)
def attention(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, mask: torch.Tensor | None
) -> torch.Tensor:
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)


@attention.register_fake
def _(q, k, v, mask):
    return torch.empty_like(q)


def exact_norm(x, normalized_shape, weight=None, bias=None, eps=1e-5):
    return layer_norm(x, list(normalized_shape), weight, bias, eps)


def exact_attention(q, k, v, attn_mask=None):
    return attention(q, k, v, attn_mask)


def norm_forward(module, x):
    return exact_norm(
        x, module.normalized_shape, module.weight, module.bias, module.eps
    )


def gelu_forward(module, x):
    return gelu(x, module.approximate)


def install(engine, *, preserve_attention=False):
    # Bind copies per model so a resident baseline keeps its original functions.
    for module in engine.model.modules():
        if isinstance(module, torch.nn.LayerNorm):
            module.forward = types.MethodType(norm_forward, module)
        elif isinstance(module, torch.nn.GELU):
            module.forward = types.MethodType(gelu_forward, module)
    for name in ("forward", "_head_layer"):
        original = getattr(engine.model, name).__func__
        namespace = dict(original.__globals__)
        functional = types.SimpleNamespace(**vars(F))
        functional.layer_norm = exact_norm
        if preserve_attention:
            functional.scaled_dot_product_attention = exact_attention
        namespace["F"] = functional
        fn = types.FunctionType(
            original.__code__,
            namespace,
            original.__name__,
            original.__defaults__,
            original.__closure__,
        )
        fn.__kwdefaults__ = original.__kwdefaults__
        setattr(engine.model, name, types.MethodType(fn, engine.model))
    return engine
