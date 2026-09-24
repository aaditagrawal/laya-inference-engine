"""Exact bitplane encoding permuted into native PTX MMA fragment order."""

import ctypes
import hashlib
import json

import torch
import triton as tr
import triton.language as tl

from .bitplane_pipeline_build import DIRECTORY, SOURCE

_LIBRARY = None
TILES = {0: (32, 32), 1: (64, 32), 2: (32, 64), 3: (64, 16)}


def load():
    global _LIBRARY
    if _LIBRARY is None:
        report = json.loads((DIRECTORY / "build.json").read_text())
        if report["source_sha256"] != hashlib.sha256(SOURCE.read_bytes()).hexdigest():
            raise RuntimeError("Native bitplane source changed. Rebuild first.")
        library_path = DIRECTORY / "bitplane_pipeline.so"
        if (
            report["library_sha256"]
            != hashlib.sha256(library_path.read_bytes()).hexdigest()
        ):
            raise RuntimeError("Native bitplane library hash changed")
        library = ctypes.CDLL(str(library_path))
        p, i = ctypes.c_void_p, ctypes.c_int
        library.bitplane_unpack.argtypes = [p, p, i, i, i, p]
        library.bitplane_gemm.argtypes = [p, p, p, i, i, i, i, i, p]
        _LIBRARY = library
    return _LIBRARY


@tr.jit
def _pack(W, P, K: tl.constexpr, BN: tl.constexpr):
    group = tl.arange(0, BN * 2)
    lane = tl.arange(0, 32)
    nt, kt, plane = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    nb = group // 16
    kk = (group // 4) % 4
    value = group % 4
    n = nt * BN + nb[:, None] * 8 + lane[None, :] // 4
    k = (
        kt * 64
        + kk[:, None] * 16
        + (lane[None, :] % 4) * 2
        + (value[:, None] % 2)
        + (value[:, None] // 2) * 8
    )
    bits = tl.load(W + n * K + k).to(tl.uint32)
    exponent = ((bits >> 7) & 255).to(tl.int32)
    delta = exponent - 119
    z = (delta << 1) ^ (delta >> 31)
    code = (bits & 127) | ((bits >> 15) << 7) | (z.to(tl.uint32) << 8)
    word = tl.sum(((code >> plane) & 1) << lane[None, :], 1)
    tl.store(P + ((nt * (K // 64) + kt) * 16 + plane) * (BN * 2) + group, word)


def pack(weight, bn):
    if (
        weight.dtype != torch.bfloat16
        or not weight.is_cuda
        or not weight.is_contiguous()
    ):
        raise ValueError("Expected contiguous CUDA BF16 weight")
    n, k = weight.shape
    if bn not in [16, 32, 64] or n % bn or k % 64:
        raise ValueError("Invalid weight geometry")
    exponent = (weight.view(torch.int16).to(torch.int32) >> 7) & 255
    if not bool((exponent < 247).all()):
        raise ValueError("Exponent outside the reversible transform range")
    packed = torch.empty(
        (n // bn, k // 64, 16, bn * 2), device=weight.device, dtype=torch.uint32
    )
    _pack[(n // bn, k // 64, 16)](weight.view(torch.uint16), packed, k, bn)
    decoded = torch.empty_like(weight)
    error = load().bitplane_unpack(
        packed.data_ptr(),
        decoded.data_ptr(),
        n,
        k,
        bn,
        torch.cuda.current_stream().cuda_stream,
    )
    if error:
        raise RuntimeError(f"Native decode launch error {error}")
    if not torch.equal(decoded.view(torch.int16), weight.view(torch.int16)):
        raise RuntimeError("Native warp bit transpose changed weight bits")
    return packed


class Operation:
    def __init__(self, x, w, n, k, tile, stages, paired=True):
        if tile not in TILES or stages not in [2, 3] or k % 64 or k < 64 * stages:
            raise ValueError("Unsupported pipeline configuration")
        _, bn = TILES[tile]
        if (
            n % bn
            or x.numel() != 64 * k
            or x.dtype != torch.bfloat16
            or w.dtype != torch.uint32
        ):
            raise ValueError("Invalid tensor shape or dtype")
        if (
            x.device != w.device
            or not x.is_cuda
            or not x.is_contiguous()
            or not w.is_contiguous()
        ):
            raise ValueError("Expected contiguous CUDA tensors on the same device")
        if w.numel() != n * k // 2:
            raise ValueError("Invalid encoded weight shape")
        self.x, self.w, self.n, self.k, self.tile, self.stages = (
            x,
            w,
            n,
            k,
            tile,
            stages,
        )
        self.paired = bool(paired)
        self.output = torch.empty(
            (*x.shape[:-1], n), device=x.device, dtype=torch.bfloat16
        )
        self.library = load()

    def __call__(self):
        error = self.library.bitplane_gemm(
            self.x.data_ptr(),
            self.w.data_ptr(),
            self.output.data_ptr(),
            self.n,
            self.k,
            self.tile,
            self.stages,
            int(self.paired),
            torch.cuda.current_stream().cuda_stream,
        )
        if error:
            raise RuntimeError(f"Native bitplane GEMM launch error {error}")
        return self.output
