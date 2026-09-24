"""Checked native interface for dedicated producer lossless BF16 decoding."""

import ctypes
import hashlib
import json

import torch

from .native_lossless import _pack
from .ws_lossless_build import DIRECTORY, SOURCE

LIBRARY = None
TILES = {0: (32, 32), 1: (64, 32), 2: (32, 64)}


def load():
    global LIBRARY
    if LIBRARY is None:
        report = json.loads((DIRECTORY / "build.json").read_text())
        library = DIRECTORY / "ws_lossless.so"
        if report["source_sha256"] != hashlib.sha256(SOURCE.read_bytes()).hexdigest():
            raise RuntimeError("Native source changed; rebuild first")
        if report["library_sha256"] != hashlib.sha256(library.read_bytes()).hexdigest():
            raise RuntimeError("Native library hash changed")
        LIBRARY = ctypes.CDLL(str(library))
        p, i = ctypes.c_void_p, ctypes.c_int
        LIBRARY.ws_gemm.argtypes = [p, p, p, i, i, i, i, i, i, p]
        LIBRARY.ws_unpack.argtypes = [p, p, i, i, i, p]
    return LIBRARY


def pack(weight, packed=True):
    if (
        weight.dtype != torch.bfloat16
        or not weight.is_cuda
        or not weight.is_contiguous()
    ):
        raise ValueError("Expected contiguous CUDA BF16 weights")
    n, k = weight.shape
    if n % 8 or k % 16:
        raise ValueError("Weights must divide the native fragment geometry")
    if packed:
        bits = weight.view(torch.int16).to(torch.int32) & 65535
        exponent = (bits >> 7) & 255
        valid = ((exponent >= 103) & (exponent <= 133)) | (
            (exponent == 0) & ((bits & 127) == 0)
        )
        if not bool(valid.all()):
            raise ValueError("Unsupported exponent or subnormal")
    storage = torch.empty(
        (n // 8, k // 16, 52 if packed else 64),
        device=weight.device,
        dtype=torch.uint32,
    )
    _pack[(n // 8, k // 16)](weight.view(torch.uint16), storage, k, packed, num_warps=1)
    decoded = torch.empty_like(weight)
    error = load().ws_unpack(
        storage.data_ptr(),
        decoded.data_ptr(),
        n,
        k,
        int(packed),
        torch.cuda.current_stream().cuda_stream,
    )
    if error:
        raise RuntimeError(f"Native unpack launch error {error}")
    if not torch.equal(decoded.view(torch.int16), weight.view(torch.int16)):
        raise RuntimeError("Native decoder changed BF16 weight bits")
    return storage


class Operation:
    def __init__(
        self, x, w, n, k, tile, producers, stages, packed=True, shared_copy=False
    ):
        if tile not in TILES or producers not in [1, 2] or stages not in [2, 3]:
            raise ValueError("Unsupported pipeline configuration")
        if k != 1024 or n != 5248 or x.numel() != 64 * k:
            raise ValueError("This bounded experiment requires actual MLPWi shapes")
        if x.dtype != torch.bfloat16 or w.dtype != torch.uint32:
            raise ValueError("Invalid tensor dtype")
        if (
            x.device != w.device
            or not x.is_cuda
            or not x.is_contiguous()
            or not w.is_contiguous()
        ):
            raise ValueError("Expected contiguous tensors on the same CUDA device")
        if w.numel() != n // 8 * (k // 16) * (52 if packed else 64):
            raise ValueError("Wrong encoded weight size")
        self.x, self.w = x, w
        self.n, self.k, self.tile = n, k, tile
        self.producers, self.stages, self.packed = producers, stages, packed
        self.format = 2 if packed and shared_copy else int(packed)
        self.output = torch.empty((*x.shape[:-1], n), device=x.device, dtype=x.dtype)
        self.library = load()

    def __call__(self):
        error = self.library.ws_gemm(
            self.x.data_ptr(),
            self.w.data_ptr(),
            self.output.data_ptr(),
            self.n,
            self.k,
            self.tile,
            self.producers,
            self.stages,
            self.format,
            torch.cuda.current_stream().cuda_stream,
        )
        if error:
            raise RuntimeError(f"Native GEMM launch error {error}")
        return self.output
