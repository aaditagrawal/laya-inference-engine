"""Prearranged BF16 weight tiles for contiguous TMA reads, without quantization."""

import json
import types
from dataclasses import dataclass
from pathlib import Path

import torch
import triton as tr
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor

from .matmul import _reduce


@dataclass
class PackedWeight:
    data: torch.Tensor
    n: int
    k: int
    bn: int
    bk: int
    split: int
    transposed: bool


@torch.inference_mode()
def pack(weight, bn, bk, split, transposed):
    n, k = weight.shape
    if weight.dtype != torch.bfloat16 or n % bn:
        raise ValueError("Expected BF16 weights and an exact output-dimension tile")
    padded_k = tr.cdiv(tr.cdiv(k, bk), split) * split * bk
    padded = torch.nn.functional.pad(weight, (0, padded_k - k))
    blocked = padded.view(n // bn, bn, padded_k // bk, bk).permute(0, 2, 1, 3)
    if transposed:
        blocked = blocked.transpose(2, 3)
    data = blocked.contiguous()
    restored = data.transpose(2, 3) if transposed else data
    restored = restored.permute(0, 2, 1, 3).reshape(n, padded_k)[:, :k]
    if not torch.equal(restored.view(torch.int16), weight.view(torch.int16)):
        raise RuntimeError("Weight permutation changed a BF16 bit")
    return PackedWeight(data, n, k, bn, bk, split, transposed)


@tr.jit
def _packed(
    X,
    W,
    P,
    Y,
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    SPLIT: tl.constexpr,
    TRANSPOSED: tl.constexpr,
    ACCESS: tl.constexpr,
):
    m0, tile_n, part = tl.program_id(0) * BM, tl.program_id(1), tl.program_id(2)
    steps: tl.constexpr = tl.cdiv(tl.cdiv(K, BK), SPLIT)
    acc = tl.full((BM, BN), 0, tl.float32)
    for step in range(steps):
        tile_k = part * steps + step
        if ACCESS == 0:
            m = m0 + tl.arange(0, BM)
            k = tile_k * BK + tl.arange(0, BK)
            a = tl.load(
                X + m[:, None] * K + k[None, :], (m[:, None] < M) & (k[None, :] < K), 0
            )
            start = (tile_n * steps * SPLIT + tile_k) * BN * BK
            nk, kk = tl.arange(0, BN), tl.arange(0, BK)
            if TRANSPOSED:
                w = tl.load(W + start + kk[:, None] * BN + nk[None, :])
            else:
                w = tl.load(W + start + nk[None, :] * BK + kk[:, None])
        else:
            a = X.load([m0, tile_k * BK])
            if TRANSPOSED:
                if ACCESS == 1:
                    w = W.load([(tile_n * steps * SPLIT + tile_k) * BK, 0])
                else:
                    w = W.load([tile_n, tile_k * BK, 0]).reshape((BK, BN))
            else:
                if ACCESS == 1:
                    w = W.load([(tile_n * steps * SPLIT + tile_k) * BN, 0]).T
                else:
                    w = W.load([tile_n, tile_k * BN, 0]).reshape((BN, BK)).T
        acc = tl.dot(a, w, acc)
    m, n = m0 + tl.arange(0, BM), tile_n * BN + tl.arange(0, BN)
    offset = m[:, None] * N + n[None, :]
    if SPLIT == 1:
        tl.store(Y + offset, acc, m[:, None] < M)
    else:
        tl.store(P + part * M * N + offset, acc, m[:, None] < M)


COMPILED = {}


def matmul(x, weight, config, return_partials=False, access=2):
    bm, warps, stages = config
    n, k, bn, bk, split = weight.n, weight.k, weight.bn, weight.bk, weight.split
    m = x.numel() // k
    if x.shape[-1] != k or not x.is_contiguous():
        raise ValueError("Expected matching contiguous input")
    nt, kt, inner_a, inner_b = weight.data.shape
    if access == 0:
        xd, wd = x, weight.data
    else:
        xd = TensorDescriptor(x, [m, k], [k, 1], [bm, bk])
        wd = (
            TensorDescriptor(
                weight.data,
                [nt * kt * inner_a, inner_b],
                [inner_b, 1],
                [inner_a, inner_b],
            )
            if access == 1
            else TensorDescriptor(
                weight.data,
                [nt, kt * inner_a, inner_b],
                [kt * inner_a * inner_b, inner_b, 1],
                [1, inner_a, inner_b],
            )
        )
    y = torch.empty((*x.shape[:-1], n), device=x.device, dtype=torch.bfloat16)
    partial = (
        torch.empty((split, m, n), device=x.device, dtype=torch.bfloat16)
        if split > 1
        else y
    )
    kernel = _packed[(tr.cdiv(m, bm), n // bn, split)](
        xd,
        wd,
        partial,
        y,
        m,
        n,
        k,
        bm,
        bn,
        bk,
        split,
        weight.transposed,
        access,
        num_warps=warps,
        num_stages=stages,
    )
    COMPILED[(n, k, bn, bk, split, weight.transposed, *config, access)] = kernel
    if return_partials:
        return partial.view(split, *x.shape[:-1], n)
    if split > 1:
        _reduce[(tr.cdiv(m * n, 512),)](partial, y, m * n, split, 512)
    return y


@torch.library.custom_op(
    "laya_frontier_packed::linear", mutates_args=(), device_types="cuda"
)
def compiled_matmul(
    x: torch.Tensor, data: torch.Tensor, n: int, k: int, config: list[int]
) -> torch.Tensor:
    bm, bn, bk, split, warps, stages, transposed, access = config
    weight = PackedWeight(data, n, k, bn, bk, split, bool(transposed))
    return matmul(x, weight, (bm, warps, stages), access=access)


@compiled_matmul.register_fake
def _(x, data, n, k, config):
    return torch.empty((*x.shape[:-1], n), device=x.device, dtype=x.dtype)


def install_qkv(model):
    rows = []
    root = Path(__file__).resolve().parents[2] / "results/frontier"
    for filename, access in [
        ("matmul-packed-tma.json", 2),
        ("matmul-packed-tma2.json", 1),
        ("matmul-packed-pointer.json", 0),
    ]:
        report = json.loads((root / filename).read_text())
        if report.get("access", access) != access:
            raise RuntimeError(
                f"Access method in {filename} does not match its expected layout"
            )
        rows += [
            dict(row, access=access, source=filename)
            for row in report["rows"]
            if row["field"] == "attn.Wqkv"
            and row.get("mismatches") == 0
            and row.get("speedup", 0) > 1.015
        ]
    if not rows:
        raise RuntimeError("No qualifying packed QKV configuration")
    choice = max(rows, key=lambda row: row["speedup"])
    bm, bn, bk, split, warps, stages, transposed = choice["config"]
    config = [
        int(v) for v in [bm, bn, bk, split, warps, stages, transposed, choice["access"]]
    ]
    for layer in model.net.encoder.layers:
        module = layer.attn.Wqkv
        weight = pack(module.weight, bn, bk, split, transposed)
        module.register_buffer("frontier_packed_qkv", weight.data)
        original = module.forward

        def forward(module, x, original=original, config=config):
            if x.numel() // x.shape[-1] == 64 and x.is_contiguous():
                n, k = module.weight.shape
                return compiled_matmul(x, module.frontier_packed_qkv, n, k, config)
            return original(x)

        module.forward = types.MethodType(forward, module)
    return choice
