"""Skip padded split-K iterations while preserving the four BF16 partials."""

import hashlib

import torch
import triton as tr
import triton.language as tl

COMPILED = {}


@tr.jit
def _accumulate(X, W, m, n, part, STEPS: tl.constexpr):
    kk = tl.arange(0, 64)
    acc = tl.full((32, 32), 0, tl.float32)
    for i in range(STEPS):
        k = part * 704 + i * 64 + kk
        a = tl.load(X + m[:, None] * 2624 + k[None, :], k[None, :] < 2624, 0)
        b = tl.load(W + n[None, :] * 2624 + k[:, None], k[:, None] < 2624, 0)
        acc = tl.dot(a, b, acc)
    return acc


@tr.jit
def _partial(X, W, Y, MODE: tl.constexpr, ORDER: tl.constexpr):
    pid = tl.program_id(0)
    if ORDER == 0:
        mtile, ntile, part = pid % 2, pid // 2 % 32, pid // 64
    elif ORDER == 1:
        mtile, ntile, part = pid % 2, pid // 8, pid // 2 % 4
    else:
        mtile, ntile, part = pid // 128, pid // 4 % 32, pid % 4
    m = mtile * 32 + tl.arange(0, 32)
    n = ntile * 32 + tl.arange(0, 32)
    if MODE == 2:
        if part == 3:
            acc = _accumulate(X, W, m, n, part, 8)
        else:
            acc = _accumulate(X, W, m, n, part, 11)
    else:
        kk = tl.arange(0, 64)
        acc = tl.full((32, 32), 0, tl.float32)
        steps = tl.minimum(11, 41 - part * 11) if MODE == 1 else 11
        for i in range(steps):
            k = part * 704 + i * 64 + kk
            a = tl.load(X + m[:, None] * 2624 + k[None, :], k[None, :] < 2624, 0)
            b = tl.load(W + n[None, :] * 2624 + k[:, None], k[:, None] < 2624, 0)
            acc = tl.dot(a, b, acc)
    tl.store(Y + part * 64 * 1024 + m[:, None] * 1024 + n[None, :], acc)


def partial(x, weight, mode=1, order=0):
    if (
        x.numel() != 64 * 2624
        or x.shape[-1] != 2624
        or weight.shape != (1024, 2624)
        or x.dtype != torch.bfloat16
        or weight.dtype != x.dtype
        or not x.is_cuda
        or weight.device != x.device
        or not x.is_contiguous()
        or not weight.is_contiguous()
        or mode not in (0, 1, 2)
        or order not in (0, 1, 2)
    ):
        raise ValueError("Expected retained MLP-output BF16 input and weight shape")
    output = torch.empty((4, *x.shape[:-1], 1024), dtype=x.dtype, device=x.device)
    kernel = _partial[(256,)](x, weight, output, mode, order, num_warps=4, num_stages=3)
    COMPILED[(mode, order)] = kernel
    return output


@torch.library.custom_op(
    "laya_split_tail::partial", mutates_args=(), device_types="cuda"
)
def compiled_partial(
    x: torch.Tensor, weight: torch.Tensor, mode: int, order: int
) -> torch.Tensor:
    return partial(x, weight, mode, order)


@compiled_partial.register_fake
def fake(x, weight, mode, order):
    return torch.empty((4, *x.shape[:-1], 1024), dtype=x.dtype, device=x.device)


def binary_hashes():
    return {
        str(key): {
            name: hashlib.sha256(
                value if isinstance(value, bytes) else value.encode()
            ).hexdigest()
            for name, value in kernel.asm.items()
            if name in {"ptx", "cubin"}
        }
        for key, kernel in sorted(COMPILED.items())
    }
