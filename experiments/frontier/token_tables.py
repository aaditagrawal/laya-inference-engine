"""Precompute token-local work from frozen weights, without caching requests."""

import inspect
import textwrap
import time
import types

import torch

from laya_blackwell.model import FastDecisionModel


@torch.inference_mode()
def install(model):
    encoder = model.net.encoder
    embeddings = encoder.embeddings
    if model.net.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError("Token tables require a frozen inference model")
    count, hidden = embeddings.tok_embeddings.weight.shape
    device = embeddings.tok_embeddings.weight.device
    qkv_size = encoder.layers[0].attn.Wqkv.weight.shape[0]
    normalized = torch.empty((count, hidden), device=device, dtype=torch.float32)
    projections = torch.empty((count, qkv_size), device=device, dtype=torch.bfloat16)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for offset in range(0, count, 64):
        # Keep the request's 64-row GEMM and normalization geometry. A large
        # vocabulary GEMM could select a different accumulation order.
        ids = torch.arange(offset, offset + 64, device=device).clamp_max(count - 1)
        h = embeddings(ids.view(1, 64))
        qkv = encoder.layers[0].attn.Wqkv(h.bfloat16())
        rows = min(64, count - offset)
        normalized[offset : offset + rows].copy_(h[0, :rows])
        projections[offset : offset + rows].copy_(qkv[0, :rows])
    torch.cuda.synchronize()
    setup_seconds = time.perf_counter() - start
    encoder.register_buffer("frontier_token_norm", normalized)
    encoder.register_buffer("frontier_token_qkv", projections)

    original = model.forward.__func__
    # The window adapter generates its function without a source file. Rebuild
    # only the short branch from inspected repository code; all other shapes
    # dispatch to the original window-aware method.
    source = textwrap.dedent(inspect.getsource(FastDecisionModel.forward))
    replacements = {
        "enc = self.net.encoder": (
            "if input_ids.numel() != 64:\n"
            "        return self.frontier_original_forward(input_ids, attention_mask, "
            "marker_pos, marker_mask, qtype, global_attention_unmasked)\n"
            "    enc = self.net.encoder"
        ),
        "act, gate = layer.mlp.Wi(n).chunk(2, dim=-1)\n        pending = layer.mlp.Wo(F.gelu(act) * gate)": (
            "pending = layer.mlp.Wo(geglu_fn(layer.mlp.Wi(n)))"
        ),
        "h = enc.embeddings(input_ids)": (
            "h = (F.embedding(input_ids, enc.frontier_token_norm) "
            "if input_ids.numel() == 64 else enc.embeddings(input_ids))"
        ),
        "qkv = layer.attn.Wqkv(n).view(b, length, 3, self.heads, self.dim)": (
            "qkv = (F.embedding(input_ids, enc.frontier_token_qkv) "
            "if i == 0 and input_ids.numel() == 64 else layer.attn.Wqkv(n))"
            ".view(b, length, 3, self.heads, self.dim)"
        ),
    }
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise RuntimeError("Model forward changed; revalidate token-table rewrite")
        source = source.replace(old, new)
    # Keep the per-instance native normalization/GEGLU namespace. Store source
    # in linecache so later inspection and Dynamo diagnostics retain it.
    import linecache

    filename = str(__file__) + ":generated-forward"
    linecache.cache[filename] = (len(source), None, source.splitlines(True), filename)
    namespace = dict(original.__globals__)
    if "geglu_fn" not in namespace:
        raise RuntimeError("Expected the installed exact GEGLU adapter")
    exec(compile(source, filename, "exec"), namespace)  # noqa: S102 - inspected repository code and fixed substitutions.
    model.frontier_original_forward = model.forward
    model.forward = types.MethodType(namespace[original.__name__], model)
    return {
        "vocabulary_rows": count,
        "bytes": normalized.nbytes + projections.nbytes,
        "setup_seconds": setup_seconds,
        "precompute_rows": 64,
        "scope": "Frozen token embeddings and first QKV, before positional mixing",
    }
