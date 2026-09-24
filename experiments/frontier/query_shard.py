"""Fixed-shape attention specializations with exact padding-aware fallback."""

import hashlib
import json
import subprocess
from pathlib import Path

import torch

_loaded = False
ROOT = Path(__file__).resolve().parents[2]


def load():
    global _loaded
    if _loaded:
        return
    directory = ROOT / ".research/frontier-query-shard"
    manifest = json.loads((directory / "build.json").read_text())
    source = Path(__file__).with_name("query_shard_kernel.cu")
    library = directory / "laya_frontier_query_shard.so"
    header = (
        Path(torch.__file__).parent
        / "include/ATen/native/transformers/cuda/mem_eff_attention/kernel_forward.h"
    )
    for path, field in [
        (source, "source_sha256"),
        (library, "library_sha256"),
        (header, "torch_header_sha256"),
    ]:
        if hashlib.sha256(path.read_bytes()).hexdigest() != manifest[field]:
            raise RuntimeError(f"Attention specialization build mismatch: {path}")
    if torch.version.git_version != manifest["torch_revision"]:
        raise RuntimeError("Attention specialization requires its pinned Torch build")
    revision = subprocess.check_output(
        [
            "git",
            "-C",
            str(ROOT / ".research/frontier-torch-cutlass"),
            "rev-parse",
            "HEAD",
        ],
        text=True,
    ).strip()
    if revision != manifest["cutlass_revision"]:
        raise RuntimeError("Attention specialization requires pinned CUTLASS")
    torch.ops.load_library(str(library))

    @torch.library.register_fake("laya_frontier_query_shard::forward")
    def fake(q, k, v, rows, bias=None):
        return torch.empty((1, 64, 16, 64), device=q.device, dtype=q.dtype).transpose(
            1, 2
        )

    _loaded = True


def attention(q, k, v, rows, bias=None):
    return torch.ops.laya_frontier_query_shard.forward(q, k, v, rows, bias)
