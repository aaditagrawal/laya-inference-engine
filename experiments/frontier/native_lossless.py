"""Lossless BF16 values packed directly in PTX MMA B-fragment lane order."""

import hashlib
import json

import torch
import triton as tr
import triton.language as tl

from .matmul import _reduce
from .native_lossless_build import DIRECTORY, SOURCE

_LOADED = False


def load():
    global _LOADED
    if not _LOADED:
        report = json.loads((DIRECTORY / "build.json").read_text())
        if report["source_sha256"] != hashlib.sha256(SOURCE.read_bytes()).hexdigest():
            raise RuntimeError("Native lossless library needs rebuilding")
        if report["torch_git"] != torch.version.git_version:
            raise RuntimeError(
                "Native lossless library targets a different torch build"
            )
        torch.ops.load_library(str(DIRECTORY / "laya_native_lossless.so"))
        _LOADED = True


@tr.jit
def _value(W, nb, kb, lane, i, K: tl.constexpr):
    n = nb * 8 + lane // 4
    k = kb * 16 + (lane % 4) * 2 + i % 2 + (i // 2) * 8
    return tl.load(W + n * K + k).to(tl.uint32)


@tr.jit
def _pack(W, P, K: tl.constexpr, PACKED: tl.constexpr):
    nb, kb = tl.program_id(0), tl.program_id(1)
    lane = tl.arange(0, 32)
    a = _value(W, nb, kb, lane, 0, K)
    b = _value(W, nb, kb, lane, 1, K)
    c = _value(W, nb, kb, lane, 2, K)
    d = _value(W, nb, kb, lane, 3, K)
    if PACKED:
        base = P + (nb * (K // 16) + kb) * 52
        sm = (a & 127) | ((a >> 8) & 128)
        sm |= ((b & 127) | ((b >> 8) & 128)) << 8
        sm |= ((c & 127) | ((c >> 8) & 128)) << 16
        sm |= ((d & 127) | ((d >> 8) & 128)) << 24
        tl.store(base + lane, sm)
        first = lane * 32 // 5
        shift = lane * 32 % 5
        word = tl.full((32,), 0, tl.uint32)
        for j in tl.static_range(8):
            code_index = first + j
            bits = _value(W, nb, kb, tl.minimum(code_index // 4, 31), code_index % 4, K)
            exponent = (bits >> 7) & 255
            code = tl.where(exponent == 0, 0, exponent - 102)
            if j == 0:
                word |= code >> shift
            else:
                word |= tl.where(j * 5 - shift < 32, code << (j * 5 - shift), 0)
        tl.store(base + 32 + lane, word, lane < 20)
    else:
        base = P + (nb * (K // 16) + kb) * 64
        tl.store(base + lane, a | (b << 16))
        tl.store(base + 32 + lane, c | (d << 16))


@tr.jit
def _unpack(P, Y, K: tl.constexpr, PACKED: tl.constexpr):
    nb, kb = tl.program_id(0), tl.program_id(1)
    lane = tl.arange(0, 32)
    if PACKED:
        base = P + (nb * (K // 16) + kb) * 52
        sm = tl.load(base + lane)
        word, shift = lane * 20 // 32, lane * 20 % 32
        low = tl.load(base + 32 + word)
        high = tl.load(base + 33 + word, word < 19, 0)
        exp = (low >> shift) | tl.where(shift > 0, high << (32 - shift), 0)
        for i in tl.static_range(4):
            e = (exp >> (i * 5)) & 31
            byte = (sm >> (i * 8)) & 255
            bits = (
                (byte & 127) | ((byte & 128) << 8) | (tl.where(e == 0, 0, e + 102) << 7)
            )
            n = nb * 8 + lane // 4
            k = kb * 16 + lane % 4 * 2 + i % 2 + i // 2 * 8
            tl.store(Y + n * K + k, bits.to(tl.uint16))
    else:
        base = P + (nb * (K // 16) + kb) * 64
        for i in tl.static_range(4):
            bits = tl.load(base + (i // 2) * 32 + lane) >> ((i % 2) * 16)
            n = nb * 8 + lane // 4
            k = kb * 16 + lane % 4 * 2 + i % 2 + i // 2 * 8
            tl.store(Y + n * K + k, bits.to(tl.uint16))


def pack(weight, packed=True):
    if weight.dtype != torch.bfloat16 or not weight.is_contiguous():
        raise ValueError("Expected contiguous BF16 weight")
    n, k = weight.shape
    if n % 8 or k % 16:
        raise ValueError("Expected N multiple of 8 and K multiple of 16")
    if packed:
        bits = weight.view(torch.int16).to(torch.int32) & 65535
        exp = (bits >> 7) & 255
        valid = ((exp >= 103) & (exp <= 133)) | ((exp == 0) & ((bits & 127) == 0))
        if not bool(valid.all()):
            raise ValueError("Unsupported BF16 exponent or subnormal")
    storage = torch.empty(
        (n // 8, k // 16, 52 if packed else 64),
        device=weight.device,
        dtype=torch.uint32,
    )
    _pack[(n // 8, k // 16)](weight.view(torch.uint16), storage, k, packed, num_warps=1)
    decoded = torch.empty_like(weight)
    _unpack[(n // 8, k // 16)](
        storage, decoded.view(torch.uint16), k, packed, num_warps=1
    )
    if not torch.equal(decoded.view(torch.int16), weight.view(torch.int16)):
        raise RuntimeError("Packed-weight decode differs at bit level")
    load()
    native_decoded = torch.ops.laya_native_lossless.decode(storage, n, k, packed)
    if not torch.equal(native_decoded.view(torch.int16), weight.view(torch.int16)):
        raise RuntimeError("Native register decode differs at bit level")
    return storage


def matmul(x, weight, n, split=1, tile=0, unroll=1, packed=True):
    load()
    partial = torch.ops.laya_native_lossless.run(
        x, weight, n, split, tile, unroll, packed
    )
    if split == 1:
        return partial.view(*x.shape[:-1], n)
    y = torch.empty((*x.shape[:-1], n), device=x.device, dtype=torch.bfloat16)
    _reduce[(tr.cdiv(64 * n, 512),)](partial, y, 64 * n, split, 512)
    return y
