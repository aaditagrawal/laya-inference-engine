"""Python adapter for pinned PyTorch/CUTLASS attention template instances."""

import torch

from .paths import artifact_path

_loaded = False


def load():
    global _loaded
    if _loaded:
        return
    torch.ops.load_library(str(artifact_path("attention")))

    @torch.library.register_fake("laya_fast_attention::forward")
    def fake(q, k, v, bias, query_tile):
        b, h, s, d = q.shape
        return torch.empty((b, s, h, d), device=q.device, dtype=q.dtype).transpose(1, 2)

    _loaded = True


def attention(q, k, v, attn_mask=None, *, query_tile=32):
    load()
    bias = None
    if attn_mask is not None:
        bias = torch.zeros_like(attn_mask, dtype=q.dtype).masked_fill_(
            ~attn_mask, float("-inf")
        )
    return torch.ops.laya_fast_attention.forward(q, k, v, bias, query_tile)
