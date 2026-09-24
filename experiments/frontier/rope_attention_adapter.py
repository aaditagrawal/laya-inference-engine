"""Private exact RoPE/attention shared-staging fusion loader."""

import hashlib
import json
import subprocess
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
_loaded = False


def load():
    global _loaded
    if _loaded:
        return
    directory = ROOT / ".research/frontier-rope-attention"
    manifest = json.loads((directory / "build.json").read_text())
    checks = [
        (Path(path), digest) for path, digest in manifest["headers_sha256"].items()
    ]
    for name, field in [
        ("rope_attention_kernel.cu", "source_sha256"),
        ("rope_attention_helpers.h", "helper_sha256"),
        ("rope_attention_build.py", "generator_sha256"),
    ]:
        checks.append((Path(__file__).with_name(name), manifest[field]))
    checks.append((directory / "laya_rope_attention.so", manifest["library_sha256"]))
    for path, digest in checks:
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"RoPE attention source/build changed: {path}")
    if torch.version.git_version != manifest["torch_revision"]:
        raise RuntimeError("RoPE attention requires pinned Torch")
    for name in ("cutlass", "flash"):
        actual = subprocess.check_output(
            [
                "git",
                "-C",
                str(ROOT / f".research/frontier-torch-{name}"),
                "rev-parse",
                "HEAD",
            ],
            text=True,
        ).strip()
        if actual != manifest[f"{name}_revision"]:
            raise RuntimeError(f"RoPE attention requires pinned {name}")
    torch.ops.load_library(str(directory / "laya_rope_attention.so"))

    @torch.library.register_fake("laya_rope_attention::forward")
    def fake(qkv, cos, sin, config, bias=None):
        return torch.empty(
            (1, 64, 16, 64), device=qkv.device, dtype=qkv.dtype
        ).transpose(1, 2)

    _loaded = True


def attention(qkv, cos, sin, config, mask=None):
    bias = None
    if mask is not None:
        bias = torch.zeros_like(mask, dtype=qkv.dtype).masked_fill_(
            ~mask, float("-inf")
        )
    return torch.ops.laya_rope_attention.forward(qkv, cos, sin, config, bias)
