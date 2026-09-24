"""Small-sequence attention probes; each numerical variant requires validation."""

import types

import torch
import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice


@tr.jit
def _attention(
    Q,
    K,
    V,
    MASK,
    O,
    QB: tl.constexpr,
    QH: tl.constexpr,
    QS: tl.constexpr,
    KB: tl.constexpr,
    KH: tl.constexpr,
    KS: tl.constexpr,
    VB: tl.constexpr,
    VH: tl.constexpr,
    VS: tl.constexpr,
    B: tl.constexpr,
    H: tl.constexpr,
    S: tl.constexpr,
    D: tl.constexpr,
    HAS_MASK: tl.constexpr,
    MASK_BATCH: tl.constexpr,
    BM: tl.constexpr,
    PRE_NORMALIZE: tl.constexpr,
    EXP: tl.constexpr,
):
    bh = tl.program_id(0)
    b, h = bh // H, bh % H
    m = tl.program_id(1) * BM + tl.arange(0, BM)
    n = tl.arange(0, 64)
    d = tl.arange(0, D)
    q = tl.load(Q + b * QB + h * QH + m[:, None] * QS + d[None, :], m[:, None] < S, 0)
    k = tl.load(K + b * KB + h * KH + n[None, :] * KS + d[:, None], n[None, :] < S, 0)
    score = tl.dot(q, k).to(tl.float32) * (D**-0.5)
    valid = n < S
    if HAS_MASK:
        valid = valid & tl.load(MASK + b * MASK_BATCH + n, n < S, 0).to(tl.int1)
    score = tl.where(valid[None, :], score, float("-inf"))
    if EXP >= 2:
        score *= 1.4426950408889634
    shifted = score - tl.max(score, 1)[:, None]
    if EXP == 0:
        probability = tl.exp2(shifted * 1.4426950408889634)
    elif EXP == 1:
        probability = libdevice.exp(shifted)
    elif EXP == 2 or EXP == 4:
        probability = tl.exp2(shifted)
    else:
        probability = libdevice.exp2(shifted)
    if EXP >= 4:
        # Mirror the installed CUTLASS accumulator row iteration: each lane
        # sums pairs at +0,+8,+16,+24, then adjacent lanes, then the two warps.
        lane = tl.arange(0, 8)
        partial = tl.full((BM, 8), 0, tl.float32)
        for step in tl.static_range(8):
            column = (lane // 4) * 32 + (lane % 4) * 2 + (step // 2) * 8 + step % 2
            partial += tl.gather(
                probability, tl.broadcast_to(column[None, :], (BM, 8)), 1
            )
        even, odd = tl.split(partial.reshape(BM, 2, 2, 2))
        denominator = tl.sum(tl.sum(even + odd, 2), 1)
    else:
        denominator = tl.sum(probability, 1)
    if PRE_NORMALIZE:
        probability /= denominator[:, None]
    v = tl.load(V + b * VB + h * VH + n[:, None] * VS + d[None, :], n[:, None] < S, 0)
    result = tl.dot(probability.to(tl.bfloat16), v)
    if not PRE_NORMALIZE:
        if EXP >= 2:
            result *= tl.div_rn(1.0, denominator)[:, None]
        else:
            result /= denominator[:, None]
    tl.store(O + (bh * S + m[:, None]) * D + d[None, :], result, m[:, None] < S)


def triton_attention(q, k, v, attn_mask=None, *, config=(16, False, 0, 4)):
    b, h, s, d = q.shape
    if s != 64 or d != 64 or k.shape != q.shape or v.shape != q.shape:
        raise ValueError(
            "This experiment supports encoder Q/K/V with sequence=64, head_dim=64"
        )
    bm, pre_normalize, exp, warps = config
    output = torch.empty((b, h, s, d), dtype=q.dtype, device=q.device)
    if attn_mask is not None:
        # With sequence <= 64 the inclusive +/-64 local window covers every key.
        # Callers only pass the repository's broadcast padding/window masks here.
        mask = attn_mask[:, 0, 0]
    else:
        mask = q
    _attention[(b * h, tr.cdiv(s, bm))](
        q,
        k,
        v,
        mask,
        output,
        *q.stride()[:3],
        *k.stride()[:3],
        *v.stride()[:3],
        b,
        h,
        s,
        d,
        attn_mask is not None,
        mask.stride(0),
        bm,
        pre_normalize,
        exp,
        num_warps=warps,
    )
    return output


def cudnn_attention(q, k, v, attn_mask=None):
    bias = None
    if attn_mask is not None:
        bias = torch.zeros_like(attn_mask, dtype=q.dtype).masked_fill_(
            ~attn_mask, float("-inf")
        )
    return torch.ops.aten._scaled_dot_product_cudnn_attention(q, k, v, bias, False)[0]


def install(model, kind="triton"):
    """Replace only the encoder's single-request, 64-token attention calls."""
    from torch.nn import functional as F

    if kind == "native":
        from .native_attention import attention as native_attention
        from .native_attention import load

        load()

    original = model.forward.__func__
    namespace = dict(original.__globals__)
    functional = types.SimpleNamespace(**vars(namespace["F"]))
    baseline = F.scaled_dot_product_attention

    def selected(q, k, v, attn_mask=None):
        if q.shape == (1, 16, 64, 64):
            if kind == "triton":
                return triton_attention(q, k, v, attn_mask, config=(16, False, 1, 4))
            if kind == "cudnn":
                return cudnn_attention(q, k, v, attn_mask)
            if kind == "native" and attn_mask is not None and attn_mask.shape[-2] == 64:
                return native_attention(q, k, v, attn_mask, query_tile=32)
        return baseline(q, k, v, attn_mask=attn_mask)

    functional.scaled_dot_product_attention = selected
    namespace["F"] = functional
    forward = types.FunctionType(
        original.__code__,
        namespace,
        original.__name__,
        original.__defaults__,
        original.__closure__,
    )
    forward.__kwdefaults__ = original.__kwdefaults__
    model.forward = types.MethodType(forward, model)
