"""Fuse rotary execution and the fixed masked tail for exact window attention."""

import torch
import triton as tr
import triton.language as tl


@tr.jit
def _rope_pad(
    X,
    C,
    S,
    Y,
    L: tl.constexpr,
    PADDED: tl.constexpr,
    H: tl.constexpr,
    D: tl.constexpr,
    TOTAL: tl.constexpr,
    BLOCK: tl.constexpr,
    FP32: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    d = i % D
    part = (i // (D * H)) % 3
    pos = (i // (D * H * 3)) % PADDED
    batch = i // (D * H * 3 * PADDED)
    valid = (i < TOTAL) & (pos < L)
    source = ((batch * L + pos) * 3 + part) * (D * H) + (i % (D * H))
    a = tl.load(X + source, valid, 0).to(tl.float32)
    other = source + tl.where(d < D // 2, D // 2, -D // 2)
    b = tl.load(X + other, valid, 0).to(tl.float32)
    b = tl.where(d < D // 2, -b, b)
    c = tl.load(C + pos * D + d, valid, 0).to(tl.float32)
    s = tl.load(S + pos * D + d, valid, 0).to(tl.float32)
    if FP32:
        ac = a * c
        bs = b * s
    else:
        ac = (a * c).to(tl.bfloat16).to(tl.float32)
        bs = (b * s).to(tl.bfloat16).to(tl.float32)
    value = tl.where(part < 2, ac + bs, a)
    tl.store(Y + i, value, i < TOTAL)


def rope_qkv_padded(qkv, cos, sin, *, fp32=False, window=64):
    batch, length, _, heads, dim = qkv.shape
    y = torch.empty(
        (batch, length + window, 3, heads, dim), dtype=qkv.dtype, device=qkv.device
    )
    _rope_pad[(tr.cdiv(y.numel(), 512),)](
        qkv,
        cos,
        sin,
        y,
        length,
        length + window,
        heads,
        dim,
        y.numel(),
        512,
        fp32,
        enable_fp_fusion=False,
    )
    q, k, v = y.permute(0, 3, 2, 1, 4).unbind(2)
    return q[:, :, :length], k, v


def window_bias(padding, window=64):
    visible = torch.nn.functional.pad(padding.bool(), (0, window), value=False)
    return torch.zeros(
        (padding.shape[0], 1, 1, padding.shape[1] + window),
        dtype=torch.bfloat16,
        device=padding.device,
    ).masked_fill(~visible[:, None, None, :], float("-inf"))


def attention_prepared(q, k, v, bias, window=64):
    batch, heads, length, _dim = q.shape
    return torch.ops.aten._efficient_attention_forward(
        q.transpose(1, 2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        bias.expand(batch, heads, length, length + window),
        None,
        None,
        None,
        None,
        0.0,
        2,
        False,
        window_size=2 * window + 1,
    )[0].transpose(1, 2)
