"""Exhaustively derived compact BF16 GELU table and fixed retained MLP geometry."""

import hashlib

import torch
import triton as tr
import triton.language as tl

from experiments.native.kernels.candidates import gelu_lut

COMPILED = {}


@torch.inference_mode()
def derive():
    reference = gelu_lut()
    bits = torch.arange(65536, device="cuda", dtype=torch.int32).to(torch.int16)
    values = bits.view(torch.bfloat16)
    half = (values.float() * 0.5).bfloat16().view(torch.int16)
    expected = reference.view(torch.int16)
    ref_cpu = expected.cpu().to(torch.int32).bitwise_and(65535).tolist()
    half_cpu = half.cpu().to(torch.int32).bitwise_and(65535).tolist()
    starts, stops = [], []
    for sign in (0, 32768):
        start = next(
            i for i in range(0x7F80) if ref_cpu[sign + i] != half_cpu[sign + i]
        )
        stop = 0x7F80
        target = (lambda i: i) if sign == 0 else (lambda i: 0x8000)
        while stop > start and ref_cpu[sign + stop - 1] == target(sign + stop - 1):
            stop -= 1
        starts.append(start)
        stops.append(stop)
    p0, n0 = starts
    p1, n1 = stops
    table = torch.cat(
        (
            reference[p0:p1],
            reference[32768 + n0 : 32768 + n1],
            reference[0x7F80:0x8000],
            reference[0xFF80:0x10000],
        )
    ).contiguous()
    report = {
        "positive_middle_start": p0,
        "positive_middle_stop": p1,
        "negative_middle_start": n0,
        "negative_middle_stop": n1,
        "positive_middle_values": p1 - p0,
        "negative_middle_values": n1 - n0,
        "special_values": 256,
        "table_bytes": table.nbytes,
        "reference_sha256": hashlib.sha256(
            reference.cpu().view(torch.uint16).numpy().tobytes()
        ).hexdigest(),
        "table_sha256": hashlib.sha256(
            table.cpu().view(torch.uint16).numpy().tobytes()
        ).hexdigest(),
        "derivation": "Maximal contiguous finite prefixes matching BF16-rounded x/2; finite suffixes matching positive identity or negative signed zero; exact reference middle and all Inf/NaN encodings retained in compact table.",
    }
    return table, (p0, p1, n0, n1), report


@tr.jit
def _compact(
    activation,
    TABLE,
    P0: tl.constexpr,
    P1: tl.constexpr,
    N0: tl.constexpr,
    N1: tl.constexpr,
):
    bits = activation.to(tl.uint16, bitcast=True).to(tl.int32)
    mag = bits & 32767
    negative = bits >= 32768
    start = tl.where(negative, N0, P0)
    stop = tl.where(negative, N1, P1)
    middle = (mag >= start) & (mag < stop)
    middle_offset = tl.where(negative, P1 - P0 + mag - N0, mag - P0)
    value_bits = (
        (activation.to(tl.float32) * 0.5).to(tl.bfloat16).to(tl.uint16, bitcast=True)
    )
    value_bits = tl.where(mag >= stop, tl.where(negative, 32768, bits), value_bits)
    table_bits = tl.load(TABLE + middle_offset, middle, 0).to(tl.uint16, bitcast=True)
    value_bits = tl.where(middle, table_bits, value_bits)
    special = mag >= 0x7F80
    special_offset = (P1 - P0) + (N1 - N0) + tl.where(negative, 128, 0) + (mag - 0x7F80)
    special_bits = tl.load(TABLE + special_offset, special, 0).to(
        tl.uint16, bitcast=True
    )
    value_bits = tl.where(special, special_bits, value_bits).to(tl.uint16)
    return value_bits.to(tl.bfloat16, bitcast=True)


@tr.jit
def _exhaustive(
    TABLE,
    OUT,
    P0: tl.constexpr,
    P1: tl.constexpr,
    N0: tl.constexpr,
    N1: tl.constexpr,
    BLOCK: tl.constexpr,
):
    bits = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    activation = bits.to(tl.uint16).to(tl.bfloat16, bitcast=True)
    result = _compact(activation, TABLE, P0, P1, N0, N1)
    tl.store(OUT + bits, result.to(tl.uint16, bitcast=True))


def exhaustive(table, ranges):
    result = torch.empty(65536, device="cuda", dtype=torch.uint16)
    kernel = _exhaustive[(128,)](table, result, *ranges, 512, enable_fp_fusion=False)
    COMPILED["exhaustive"] = kernel
    reference = gelu_lut().view(torch.uint16)
    bad = (result != reference).nonzero().flatten()
    return {
        "patterns": 65536,
        "bitwise_mismatches": bad.numel(),
        "mismatch_codes": bad.cpu().tolist(),
    }


@tr.jit
def _project_compact(
    X,
    W,
    TABLE,
    Y,
    P0: tl.constexpr,
    P1: tl.constexpr,
    N0: tl.constexpr,
    N1: tl.constexpr,
):
    m0, n0 = tl.program_id(0) * 32, tl.program_id(1) * 64
    m, n = m0 + tl.arange(0, 32), n0 + tl.arange(0, 64)
    kk = tl.arange(0, 64)
    acc = tl.full((32, 64), 0, tl.float32)
    for start in range(0, 1024, 64):
        k = start + kk
        wn = n // 2 + (n % 2) * 2624
        a = tl.load(X + m[:, None] * 1024 + k[None, :], m[:, None] < 64, 0)
        b = tl.load(W + wn[None, :] * 1024 + k[:, None], n[None, :] < 5248, 0)
        acc = tl.dot(a, b, acc)
    rounded = acc.to(tl.bfloat16)
    activation, gate = tl.split(tl.reshape(rounded, (32, 32, 2)))
    g = _compact(activation, TABLE, P0, P1, N0, N1)
    col = n0 // 2 + tl.arange(0, 32)
    tl.store(
        Y + m[:, None] * 2624 + col[None, :],
        g.to(tl.float32) * gate.to(tl.float32),
        (m[:, None] < 64) & (col[None, :] < 2624),
    )


def project(x, weight, table, ranges):
    if x.shape != (1, 64, 1024) or weight.shape != (5248, 1024):
        raise ValueError("Expected retained short MLP shape")
    if not x.is_contiguous() or not weight.is_contiguous():
        raise ValueError("Expected contiguous tensors")
    output = torch.empty((1, 64, 2624), device=x.device, dtype=x.dtype)
    kernel = _project_compact[(2, 82)](
        x,
        weight,
        table,
        output,
        *ranges,
        num_warps=4,
        num_stages=3,
        enable_fp_fusion=False,
    )
    COMPILED["project"] = kernel
    return output


def binary_hashes():
    return {
        key: {
            kind: hashlib.sha256(
                value if isinstance(value, bytes) else value.encode()
            ).hexdigest()
            for kind, value in kernel.asm.items()
            if kind in ("cubin", "ptx")
        }
        for key, kernel in sorted(COMPILED.items())
    }
