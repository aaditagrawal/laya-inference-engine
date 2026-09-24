"""Final-head Q-row pruning with unchanged K reduction and full K/V projections."""

import hashlib

import torch
import triton as tr
import triton.language as tl

COMPILED = {}


@tr.jit
def _project(X, W, B, SELECT, Q, PACKED, S: tl.constexpr, BM: tl.constexpr):
    pid = tl.program_id(0)
    kv_blocks: tl.constexpr = (64 // BM) * 32
    is_query = pid >= kv_blocks
    m = tl.where(is_query, (pid - kv_blocks) // 16, pid // 32) * BM + tl.arange(0, BM)
    n = tl.where(is_query, (pid - kv_blocks) % 16, pid % 32 + 16) * 64 + tl.arange(
        0, 64
    )
    if is_query:
        rows = tl.load(SELECT + m, m < S, 0).to(tl.int32)
        valid_rows = m < S
    else:
        rows = m
        valid_rows = m < 64
    kk = tl.arange(0, 64)
    acc = tl.full((BM, 64), 0, tl.float32)
    for step in range(16):
        k = step * 64 + kk
        a = tl.load(X + rows[:, None] * 1024 + k[None, :], valid_rows[:, None], 0)
        w = tl.load(W + n[None, :] * 1024 + k[:, None])
        acc = tl.dot(a, w, acc)
    acc += tl.load(B + n).to(tl.float32)[None, :]
    if is_query:
        offsets = (n[None, :] // 64 * S + m[:, None]) * 64 + n[None, :] % 64
        tl.store(Q + offsets, acc, valid_rows[:, None])
    else:
        tl.store(PACKED + m[:, None] * 3072 + n[None, :], acc, valid_rows[:, None])
        # The packed Q region is unused. Initialize it to finite zeros inside
        # the K CTAs, without an extra launch or races with compact Q output.
        tl.store(
            PACKED + m[:, None] * 3072 + n[None, :] - 1024,
            0,
            valid_rows[:, None] & (n[None, :] < 2048),
        )


def project(x, weight, bias, select, bm=32):
    if (
        x.shape != (1, 64, 1024)
        or weight.shape != (3072, 1024)
        or bias.shape != (3072,)
        or select.ndim != 2
        or select.shape[0] != 1
        or not 1 <= select.shape[1] <= 32
        or select.dtype != torch.int64
        or bm not in (16, 32)
    ):
        raise ValueError("Expected the final short-request head shape")
    if any(
        t.device != x.device or not t.is_contiguous() for t in (x, weight, bias, select)
    ):
        raise ValueError("Expected contiguous tensors on the same GPU")
    if not x.is_cuda or any(t.dtype != torch.bfloat16 for t in (x, weight, bias)):
        raise ValueError("Expected BF16 CUDA matrices")
    s = select.shape[1]
    q = torch.empty((1, 16, s, 64), device=x.device, dtype=x.dtype)
    packed = torch.empty((1, 64, 3072), device=x.device, dtype=x.dtype)
    compiled = _project[((64 // bm) * 32 + tr.cdiv(s, bm) * 16,)](
        x,
        weight,
        bias,
        select,
        q,
        packed,
        s,
        bm,
        num_warps=4,
        num_stages=3,
    )
    if not torch.compiler.is_compiling():
        COMPILED[(s, bm)] = compiled
    return q, packed


@torch.library.custom_op(
    "laya_head_q_select::project", mutates_args=(), device_types="cuda"
)
def compiled_project(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
    select: torch.Tensor,
    bm: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    return project(x, weight, bias, select, bm)


@compiled_project.register_fake
def fake(x, weight, bias, select, bm):
    return (
        torch.empty((1, 16, select.shape[1], 64), device=x.device, dtype=x.dtype),
        torch.empty((1, 64, 3072), device=x.device, dtype=x.dtype),
    )


def binary_hashes():
    return {
        str(key): {
            kind: hashlib.sha256(
                value if isinstance(value, bytes) else value.encode()
            ).hexdigest()
            for kind, value in kernel.asm.items()
            if kind in ("cubin", "ptx")
        }
        for key, kernel in sorted(COMPILED.items())
    }
