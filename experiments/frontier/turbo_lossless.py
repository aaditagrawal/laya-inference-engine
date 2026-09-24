"""Exact split-12 BF16 storage and inspected native TMA GEMM experiment."""

import ctypes
import hashlib
import json
from dataclasses import dataclass

import numpy as np
import torch
import triton as tr

from .matmul import _reduce
from .turbo_build import DIRECTORY, SOURCES

_LIBRARY = None


def load():
    global _LIBRARY
    if _LIBRARY is None:
        build = json.loads((DIRECTORY / "build.json").read_text())
        for source in SOURCES:
            if (
                build["source_sha256"][source.name]
                != hashlib.sha256(source.read_bytes()).hexdigest()
            ):
                raise RuntimeError(
                    "Rebuild Turbo-Lossless experiment after source changes"
                )
        library = ctypes.CDLL(str(DIRECTORY / "turbo_lossless.so"))
        ptr, integer = ctypes.c_void_p, ctypes.c_int
        library.turbo_prepare.argtypes = [
            ptr,
            ptr,
            ptr,
            integer,
            integer,
            integer,
            integer,
        ]
        library.turbo_prepare.restype = ptr
        library.turbo_descriptors.argtypes = [ptr, ptr, ptr, ptr, integer]
        library.turbo_run.argtypes = [
            ptr,
            integer,
            ptr,
            ptr,
            ptr,
            ptr,
            integer,
            integer,
            integer,
            ptr,
        ]
        library.turbo_decode.argtypes = [
            ptr,
            ptr,
            integer,
            ptr,
            ptr,
            ptr,
            ptr,
            integer,
            integer,
            ptr,
        ]
        library.turbo_free.argtypes = [ptr]
        _LIBRARY = library
    return _LIBRARY


@dataclass
class PackedWeight:
    sm: torch.Tensor
    groups: torch.Tensor
    offsets: torch.Tensor
    cols: torch.Tensor
    values: torch.Tensor
    base: int
    n: int
    k: int

    @property
    def patches(self):
        return self.cols.numel()

    @property
    def storage_bytes(self):
        return sum(
            t.numel() * t.element_size()
            for t in [self.sm, self.groups, self.offsets, self.cols, self.values]
        )


def pack(weight):
    if weight.dtype != torch.bfloat16 or weight.ndim != 2 or not weight.is_contiguous():
        raise ValueError("Expected contiguous 2D BF16 weight")
    n, k = weight.shape
    if n % 32 or k % 32:
        raise ValueError("Expected N and K multiples of 32")
    bits = weight.view(torch.int16).cpu().numpy().view(np.uint16)
    exponent = ((bits >> 7) & 255).astype(np.int32)
    count = np.bincount(exponent.reshape(-1), minlength=256)
    base = (
        int(np.argmax(np.convolve(count, np.ones(15, dtype=np.int64), mode="valid")))
        - 1
    )
    group = np.where(
        (exponent > base) & (exponent <= base + 15), exponent - base, 0
    ).astype(np.uint8)
    sm = ((bits & 127) | ((bits >> 8) & 128)).astype(np.uint8)
    groups = group[:, ::2] | (group[:, 1::2] << 4)
    rr, cc = np.nonzero(group == 0)
    offsets = np.concatenate(([0], np.cumsum(np.bincount(rr, minlength=n)))).astype(
        np.int32
    )
    values = bits[rr, cc].copy().view(np.int16)
    decoded = (
        (sm.astype(np.uint16) & 127)
        | ((sm.astype(np.uint16) & 128) << 8)
        | ((base + group.astype(np.int32)) << 7)
    ).astype(np.uint16)
    decoded[rr, cc] = bits[rr, cc]
    if not np.array_equal(decoded, bits):
        raise RuntimeError("Independent CPU decode is not exact")

    def gpu(array):
        return torch.from_numpy(array).to(device=weight.device)

    packed = PackedWeight(
        gpu(sm),
        gpu(groups),
        gpu(offsets),
        gpu(cc.astype(np.int32)),
        gpu(values),
        base,
        n,
        k,
    )
    decoded_gpu = torch.empty_like(weight)
    error = load().turbo_decode(
        packed.sm.data_ptr(),
        packed.groups.data_ptr(),
        base,
        packed.offsets.data_ptr(),
        packed.cols.data_ptr(),
        packed.values.data_ptr(),
        decoded_gpu.data_ptr(),
        n,
        k,
        torch.cuda.current_stream().cuda_stream,
    )
    if error:
        raise RuntimeError(f"Native decode launch error {error}")
    if not torch.equal(decoded_gpu.view(torch.int16), weight.view(torch.int16)):
        raise RuntimeError("Native GPU decode is not bit-exact")
    return packed


class Operation:
    """Prepared stable descriptors and output storage, suitable for graph capture."""

    def __init__(self, x, packed, tokens=32, weight_rows=64, split=1, exact=True):
        if (
            x.dtype != torch.bfloat16
            or not x.is_contiguous()
            or x.numel() != 64 * packed.k
        ):
            raise ValueError("Expected 64 contiguous BF16 activation rows")
        if (
            x.device != packed.sm.device
            or tokens not in [16, 32, 64]
            or weight_rows not in [32, 64]
        ):
            raise ValueError("Invalid device or tile geometry")
        if packed.n % weight_rows or split not in [1, 4]:
            raise ValueError("Invalid weight rows or split count")
        self.library = load()
        self.x, self.packed, self.tokens, self.weight_rows, self.split, self.exact = (
            x,
            packed,
            tokens,
            weight_rows,
            split,
            exact,
        )
        self.handle = self.library.turbo_prepare(
            packed.sm.data_ptr(),
            packed.groups.data_ptr(),
            x.data_ptr(),
            packed.n,
            packed.k,
            64,
            tokens,
        )
        if not self.handle:
            raise RuntimeError("Invalid descriptor configuration")
        error = self.library.turbo_descriptors(
            self.handle,
            packed.sm.data_ptr(),
            packed.groups.data_ptr(),
            x.data_ptr(),
            weight_rows,
        )
        if error:
            self.close()
            raise RuntimeError(f"Tensor map encoding error {error}")
        self.partial = torch.empty(
            (split, 64, packed.n), dtype=torch.bfloat16, device=x.device
        )
        self.output = (
            torch.empty(
                (*x.shape[:-1], packed.n), dtype=torch.bfloat16, device=x.device
            )
            if split > 1
            else self.partial[0].view(*x.shape[:-1], packed.n)
        )

    def __call__(self):
        p = self.packed
        error = self.library.turbo_run(
            self.handle,
            p.base,
            self.partial.data_ptr(),
            p.offsets.data_ptr(),
            p.cols.data_ptr(),
            p.values.data_ptr(),
            self.split,
            self.weight_rows,
            int(self.exact),
            torch.cuda.current_stream().cuda_stream,
        )
        if error:
            raise RuntimeError(f"Turbo GEMM launch error {error}")
        if self.split > 1:
            _reduce[(tr.cdiv(64 * p.n, 512),)](
                self.partial, self.output, 64 * p.n, self.split, 512
            )
        return self.output

    def close(self):
        if getattr(self, "handle", None):
            self.library.turbo_free(self.handle)
            self.handle = None

    def __del__(self):
        self.close()
