"""Inference-only ModernBERT and decision head, preserving FP32 residuals.

Adapted from Apache-2.0 Laya and Transformers computations. This version
changes weight storage, rotary execution and final-head selection. See NOTICE.
"""
import copy

import torch
from torch import nn
from torch.nn import functional as F

from .kernels import rope_qkv


def add_norm(x, residual, norm):
    # Keep PyTorch's Welford LayerNorm reduction. Small rounding differences
    # from alternative reductions can accumulate across the 28 BF16 layers.
    y = x if residual is None else x + residual
    return y, norm(y).bfloat16()


class FastDecisionModel(nn.Module):
    def __init__(self, reference):
        super().__init__()
        self.net = copy.deepcopy(reference).eval().requires_grad_(False)
        if self.net.encoder.config.model_type != "modernbert":
            raise ValueError("Only ModernBERT checkpoints are supported")
        for module in self.net.modules():
            if isinstance(module, nn.Linear):
                module.to(dtype=torch.bfloat16)
        # MultiheadAttention keeps the packed projection outside an nn.Linear.
        if self.net.head is not None:
            for layer in self.net.head.layers:
                layer.self_attn.in_proj_weight.data = layer.self_attn.in_proj_weight.data.bfloat16()
                layer.self_attn.in_proj_bias.data = layer.self_attn.in_proj_bias.data.bfloat16()
        self.heads = self.net.encoder.config.num_attention_heads
        self.dim = self.net.encoder.config.hidden_size // self.heads
        max_len = self.net.encoder.config.max_position_embeddings
        pos = torch.arange(max_len, device=next(reference.parameters()).device)[None]
        enc = self.net.encoder
        # Transformers 5 shares RoPE across layers and rotates Q/K in FP32.
        # Preserve each version's precision rather than merely adapting names.
        self.fp32_rope = hasattr(enc, "rotary_emb")
        dummy = torch.empty(1, device=pos.device,
                            dtype=torch.float32 if self.fp32_rope else torch.bfloat16)
        self.global_layers = tuple(
            layer.attention_type == "full_attention" if self.fp32_rope
            else layer.attn.local_attention == (-1, -1) for layer in enc.layers
        )
        for name, layer_type, index in (("global", "full_attention", 0), ("local", "sliding_attention", 1)):
            if self.fp32_rope:
                cos, sin = enc.rotary_emb(dummy, pos, layer_type)
            else:
                cos, sin = enc.layers[index].attn.rotary_emb(dummy, pos)
            self.register_buffer(name + "_cos", cos[0].contiguous())
            self.register_buffer(name + "_sin", sin[0].contiguous())

    def _head_layer(self, h, layer, mask, select=None):
        _, n = add_norm(h, None, layer.norm1)
        b, length, d = n.shape
        heads = layer.self_attn.num_heads
        qkv = F.linear(n, layer.self_attn.in_proj_weight, layer.self_attn.in_proj_bias)
        q, k, v = qkv.view(b, length, 3, heads, d // heads).permute(0, 3, 2, 1, 4).unbind(2)
        if select is not None:
            q = q.gather(2, select[:, None, :, None].expand(-1, heads, -1, d // heads))
            h = h.gather(1, select[:, :, None].expand(-1, -1, d))
        a = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        a = a.transpose(1, 2).reshape(b, -1, d)
        h, n = add_norm(h, layer.self_attn.out_proj(a), layer.norm2)
        return h + layer.linear2(F.relu(layer.linear1(n)))

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype,
                global_attention_unmasked=False):
        enc = self.net.encoder
        b, length = input_ids.shape
        # These masks are reused across all 28 layers; True means visible in SDPA.
        mask = attention_mask[:, None, None, :].bool()
        # HF omits the global mask for fully occupied sequences. An all-true
        # mask selects different SDPA kernels and can change close decisions.
        global_mask = None if global_attention_unmasked else mask
        pos = torch.arange(length, device=input_ids.device)
        window = enc.config.local_attention // 2
        local_mask = mask & ((pos[:, None] - pos[None, :]).abs() <= window)[None, None]
        h = enc.embeddings(input_ids)
        pending = None
        for i, layer in enumerate(enc.layers):
            if i == 0:
                n = h.bfloat16()
            else:
                h, n = add_norm(h, pending, layer.attn_norm)
            qkv = layer.attn.Wqkv(n).view(b, length, 3, self.heads, self.dim)
            is_global = self.global_layers[i]
            cos, sin = (self.global_cos, self.global_sin) if is_global else (self.local_cos, self.local_sin)
            q, k, v = rope_qkv(qkv, cos, sin, fp32=self.fp32_rope).permute(0, 3, 2, 1, 4).unbind(2)
            a = F.scaled_dot_product_attention(q, k, v, attn_mask=global_mask if is_global else local_mask)
            a = a.transpose(1, 2).reshape(b, length, -1)
            h, n = add_norm(h, layer.attn.Wo(a), layer.mlp_norm)
            # Keep upstream GELU rounding. Tiny differences at BF16 rounding
            # boundaries can accumulate across all encoder layers.
            act, gate = layer.mlp.Wi(n).chunk(2, dim=-1)
            pending = layer.mlp.Wo(F.gelu(act) * gate)
        h = F.layer_norm(h + pending, enc.final_norm.normalized_shape,
                         enc.final_norm.weight, enc.final_norm.bias, enc.final_norm.eps)
        h = h + self.net.type_emb(qtype)[:, None, :]
        # Only option markers and CLS are observed after the final head layer.
        # Its K/V use the entire sequence; query and FFN work is pruned exactly.
        select = torch.cat((torch.zeros_like(marker_pos[:, :1]), marker_pos), dim=1)
        if self.net.head is not None:
            for i, layer in enumerate(self.net.head.layers):
                h = self._head_layer(h, layer, mask, select if i == len(self.net.head.layers)-1 else None)
        else:
            h = h.gather(1, select[:, :, None].expand(-1, -1, h.shape[-1]))
        m = self.net.scorer[0](h[:, 1:]).bfloat16()
        logits = self.net.scorer[3](self.net.scorer[2](self.net.scorer[1](m))).squeeze(-1).float()
        logits = logits.masked_fill(~marker_mask, -1e4)
        p = logits.softmax(-1)
        count = marker_mask.sum(-1).clamp(min=2).float()
        ent = -(p * p.clamp_min(1e-9).log()).sum(-1) / count.log()
        # The engine always pads marker count to at least two.
        top = p.topk(2, -1).values
        feats = torch.stack((top[:, 0], top[:, 0]-top[:, 1], ent, count / 255), -1)
        act = self.net.act_head(torch.cat((h[:, 0], feats), -1).bfloat16()).float()
        return logits, act
