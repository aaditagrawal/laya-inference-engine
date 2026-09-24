"""Python adapter for pinned PyTorch/CUTLASS attention template instances."""

import hashlib
import json
from pathlib import Path

import torch

_loaded = False


def load():
    global _loaded
    if not _loaded:
        root = Path(__file__).resolve().parents[2]
        directory = root / ".research/frontier-attention"
        manifest = json.loads((directory / "build.json").read_text())
        source = Path(__file__).with_suffix(".cu")
        if hashlib.sha256(source.read_bytes()).hexdigest() != manifest["source_sha256"]:
            raise RuntimeError("Native attention source changed; rebuild first")
        if torch.version.git_version != manifest["torch_revision"]:
            raise RuntimeError("Native attention requires its pinned Torch build")
        torch.ops.load_library(str(directory / "laya_frontier_attention.so"))

        @torch.library.register_fake("laya_frontier_attention::forward")
        def fake(q, k, v, bias, query_tile):
            b, h, s, d = q.shape
            return torch.empty((b, s, h, d), device=q.device, dtype=q.dtype).transpose(
                1, 2
            )

        _loaded = True


def attention(q, k, v, attn_mask=None, *, query_tile=32):
    load()
    bias = None
    if attn_mask is not None:
        bias = torch.zeros_like(attn_mask, dtype=q.dtype).masked_fill_(
            ~attn_mask, float("-inf")
        )
    return torch.ops.laya_frontier_attention.forward(q, k, v, bias, query_tile)
