"""Opt-in head GEMMs selected only from exact isolated screening rows."""

import inspect
import textwrap
import types

import torch
from torch.nn import functional as F

from .head_gemm import gemm


@torch.library.custom_op("laya_fast_head::linear", mutates_args=(), device_types="cuda")
def linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    config: list[int],
    relu: bool,
) -> torch.Tensor:
    return gemm(x, weight, bias, tuple(config), relu=relu)


@linear.register_fake
def _(x, weight, bias, config, relu):
    return torch.empty((*x.shape[:-1], weight.shape[0]), dtype=x.dtype, device=x.device)


def _relu(module, x):
    if (
        x.numel() // x.shape[-1] == getattr(module, "fast_head_rows", None)
        and x.is_contiguous()
    ):
        return module(x)
    return F.relu(module(x))


def _qkv(x, module):
    config = getattr(module, "fast_head_config", None)
    if config is not None and x.numel() // x.shape[-1] == 64 and x.is_contiguous():
        return linear(
            x, module.in_proj_weight, module.in_proj_bias, list(config), False
        )
    return F.linear(x, module.in_proj_weight, module.in_proj_bias)


def install(model, selections):
    modules = dict(model.named_modules())
    chosen = {}
    for best in selections:
        config = tuple(int(v) for v in best["config"])
        m = best["rows"]
        for name in best["names"]:
            chosen[name] = best
            if name.endswith(".in_proj"):
                module = modules[name.removesuffix(".in_proj") + ".self_attn"]
                module.fast_head_config = config
                continue
            module = modules[name]
            original = module.forward
            if best["relu"]:
                module.fast_head_rows = m

            def forward(
                module, x, original=original, config=config, rows=m, relu=best["relu"]
            ):
                if x.numel() // x.shape[-1] == rows and x.is_contiguous():
                    return linear(x, module.weight, module.bias, list(config), relu)
                return original(x)

            module.forward = types.MethodType(forward, module)
    fn = model._head_layer.__func__
    source = textwrap.dedent(inspect.getsource(fn))
    before = "qkv = F.linear(n, layer.self_attn.in_proj_weight, layer.self_attn.in_proj_bias)"
    after = "qkv = head_qkv(n, layer.self_attn)"
    if before not in source or "F.relu(layer.linear1(n))" not in source:
        raise RuntimeError("Head source changed; review the adapter")
    source = source.replace(before, after).replace(
        "F.relu(layer.linear1(n))", "head_relu(layer.linear1, n)"
    )
    namespace = dict(fn.__globals__, head_qkv=_qkv, head_relu=_relu)
    exec(source, namespace)  # noqa: S102 - repository source checked above.
    model._head_layer = types.MethodType(namespace["_head_layer"], model)
    return chosen
