"""Packed BF16 final GEGLU multiply; all preceding retained arithmetic unchanged."""

import hashlib
import subprocess
from pathlib import Path

import torch
import triton as tr
import triton.language as tl
from triton.language.extra.cuda import libdevice

from experiments.native.kernels.candidates import gelu_corrections

COMPILED = {}


@tr.jit
def _multiply(a, b):
    return tl.inline_asm_elementwise(
        "mul.rn.bf16x2 $0, $1, $2;",
        constraints="=r,r,r",
        args=[a, b],
        dtype=tl.bfloat16,
        is_pure=True,
        pack=2,
    )


@tr.jit
def _project_packed(
    X,
    W,
    LUT,
    Y,
    BM: tl.constexpr,
    BN: tl.constexpr,
    BK: tl.constexpr,
    ACCESS: tl.constexpr,
    LOOKUP: tl.constexpr,
    IN_BITS: tl.constexpr,
    OUT_BITS: tl.constexpr,
):
    m0, n0 = tl.program_id(0) * BM, tl.program_id(1) * BN
    m, n = m0 + tl.arange(0, BM), n0 + tl.arange(0, BN)
    kk = tl.arange(0, BK)
    acc = tl.full((BM, BN), 0, tl.float32)
    for start in range(0, 1024, BK):
        if ACCESS == 1:
            a = X.load([m0, start])
            b = W.load([n0, start]).T
        else:
            k = start + kk
            wn = n // 2 + (n % 2) * 2624 if ACCESS == 2 else n
            a = tl.load(X + m[:, None] * 1024 + k[None, :], m[:, None] < 64, 0)
            b = tl.load(W + wn[None, :] * 1024 + k[:, None], n[None, :] < 5248, 0)
        acc = tl.dot(a, b, acc)
    # Adjacent output columns contain each activation/gate pair. Preserve
    # the projection and GELU BF16 roundings before their final product.
    rounded = acc.to(tl.bfloat16)
    activation, gate = tl.split(tl.reshape(rounded, (BM, BN // 2, 2)))
    bits = activation.to(tl.uint16, bitcast=True)
    if LOOKUP:
        g = tl.load(LUT + bits.to(tl.int32))
    else:
        af = activation.to(tl.float32)
        g = (0.5 * af * (1.0 + libdevice.erf(af * 0.7071067811865476))).to(tl.bfloat16)
        for i in tl.static_range(len(IN_BITS)):
            g = tl.where(
                bits == IN_BITS[i],
                tl.full((), OUT_BITS[i], tl.uint16).to(tl.bfloat16, bitcast=True),
                g,
            )
    col = n0 // 2 + tl.arange(0, BN // 2)
    tl.store(
        Y + m[:, None] * 2624 + col[None, :],
        _multiply(g, gate),
        (m[:, None] < 64) & (col[None, :] < 2624),
    )


@tr.jit
def _domain(GATES, OUT, REF, BLOCK: tl.constexpr):
    bits = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    a = bits.to(tl.uint16).to(tl.bfloat16, bitcast=True)
    b = tl.load(GATES + tl.program_id(1)).to(tl.bfloat16)
    result = _multiply(a, b)
    reference_float = tl.inline_asm_elementwise(
        "mul.rn.f32 $0, $1, $2;",
        constraints="=f,f,f",
        args=[a.to(tl.float32), b.to(tl.float32)],
        dtype=tl.float32,
        is_pure=True,
        pack=1,
    )
    offsets = tl.program_id(1) * 65536 + bits
    tl.store(OUT + offsets, result.to(tl.uint16, bitcast=True))
    tl.store(REF + offsets, reference_float.to(tl.bfloat16).to(tl.uint16, bitcast=True))


def domain():
    gates = [
        0x0000,
        0x8000,
        0x0001,
        0x8001,
        0x007F,
        0x807F,
        0x0080,
        0x8080,
        0x0081,
        0x8081,
        0x3F00,
        0xBF00,
        0x3F7F,
        0xBF7F,
        0x3F80,
        0xBF80,
        0x3F81,
        0xBF81,
        0x4000,
        0xC000,
        0x4049,
        0xC049,
        0x7F7F,
        0xFF7F,
        0x7F80,
        0xFF80,
        0x7F81,
        0xFF81,
        0x7FC0,
        0xFFC0,
        0x7FFF,
        0xFFFF,
    ]
    inputs = (
        torch.tensor(gates, device="cuda", dtype=torch.int32)
        .to(torch.int16)
        .view(torch.bfloat16)
    )
    output = torch.empty((len(gates), 65536), device="cuda", dtype=torch.uint16)
    reference = torch.empty_like(output)
    kernel = _domain[(128, len(gates))](
        inputs, output, reference, 512, enable_fp_fusion=False
    )
    COMPILED["domain"] = kernel
    mismatch = output != reference
    indices = mismatch.nonzero()
    examples = []
    for g, a in indices[:32].cpu().tolist():
        examples.append(
            {
                "a": a,
                "b": gates[g],
                "native": int(output[g, a]),
                "reference": int(reference[g, a]),
            }
        )
    return {
        "a_patterns": 65536,
        "gate_patterns": gates,
        "pairs": output.numel(),
        "bitwise_mismatches": int(mismatch.sum()),
        "per_gate_mismatches": mismatch.sum(1).cpu().tolist(),
        "examples": examples,
    }


def project(x, weight):
    if (
        x.shape != (1, 64, 1024)
        or weight.shape != (5248, 1024)
        or not x.is_contiguous()
        or not weight.is_contiguous()
    ):
        raise ValueError("Expected contiguous retained short MLP matrices")
    inputs, outputs = gelu_corrections()
    y = torch.empty((1, 64, 2624), device=x.device, dtype=x.dtype)
    kernel = _project_packed[(2, 82)](
        x,
        weight,
        weight,
        y,
        32,
        64,
        64,
        2,
        False,
        inputs,
        outputs,
        num_warps=4,
        num_stages=3,
        enable_fp_fusion=False,
    )
    COMPILED["project"] = kernel
    return y


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


def inspect_sass():
    directory = Path(".research/packed_bf16")
    directory.mkdir(parents=True, exist_ok=True)
    report = {}
    for name, kernel in COMPILED.items():
        path = directory / (name + ".cubin")
        path.write_bytes(kernel.asm["cubin"])
        sass = subprocess.check_output(
            ["/usr/local/cuda-13.1/bin/cuobjdump", "-sass", str(path)], text=True
        )
        path.with_suffix(".sass").write_text(sass)
        lines = [line.strip() for line in sass.splitlines() if "HMUL2.BF16" in line]
        report[name] = {
            "hmul2_bf16": lines,
            "sass_sha256": hashlib.sha256(sass.encode()).hexdigest(),
        }
    return report
