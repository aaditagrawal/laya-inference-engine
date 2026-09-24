"""Fuse split-K reduction into the following exact residual/normalization."""

import hashlib
import json
import types

import torch

from .build_reduce_norm import DIRECTORY, source
from .matmul import matmul

_loaded = False


def load():
    global _loaded
    if _loaded:
        return
    manifest = json.loads((DIRECTORY / "build.json").read_text())
    if (
        manifest["torch_git"] != torch.version.git_version
        or manifest["source_sha256"] != hashlib.sha256(source().encode()).hexdigest()
    ):
        raise RuntimeError("Rebuild the reduction/normalization extension")
    torch.ops.load_library(str(DIRECTORY / "laya_frontier_reduce_norm.so"))

    @torch.library.register_fake("laya_frontier_reduce_norm::norm")
    def fake(x, residual, weight, bias, eps, ptx=False):
        return [torch.empty_like(x), torch.empty_like(x, dtype=torch.bfloat16)]

    _loaded = True


def install(model, selection):
    if selection["variant"] != "bf16-partial" or selection["config"][3] != 4:
        raise ValueError(
            "Reduction fusion requires the validated BF16 partial configuration"
        )
    load()
    config = tuple(selection["config"])
    # The final encoder residual flows directly into final_norm, so retain its
    # ordinary reduced tensor. Other shapes also keep their installed forwards.
    for layer in model.net.encoder.layers[:-1]:
        linear = layer.mlp.Wo
        original = linear.forward

        def forward(module, x, original=original, config=config):
            if x.numel() // x.shape[-1] == 64 and x.is_contiguous():
                return matmul(
                    x, module.weight, config, partial_bf16=True, return_partials=True
                )
            return original(x)

        linear.forward = types.MethodType(forward, linear)
    original = model.forward.__func__
    namespace = dict(original.__globals__)
    baseline = namespace["add_norm"]

    def fused(x, residual, norm):
        if residual is not None and residual.ndim == 4 and residual.shape[0] == 4:
            return torch.ops.laya_frontier_reduce_norm.norm(
                x, residual, norm.weight, norm.bias, norm.eps, False
            )
        return baseline(x, residual, norm)

    namespace["add_norm"] = fused
    forward = types.FunctionType(
        original.__code__,
        namespace,
        original.__name__,
        original.__defaults__,
        original.__closure__,
    )
    forward.__kwdefaults__ = original.__kwdefaults__
    model.forward = types.MethodType(forward, model)
