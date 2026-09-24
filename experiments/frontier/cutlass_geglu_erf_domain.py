"""Derive native CUDA erf corrections across every BF16 input encoding."""

import ctypes
import fcntl
import json
import subprocess
from pathlib import Path

import torch

from experiments.native.kernels.candidates import gelu_lut

from .cutlass_geglu_build import ROOT, digest

DIRECTORY = ROOT / ".research/frontier-cutlass-geglu-erf"
SOURCE = ROOT / "experiments/frontier/cutlass_geglu_erf_domain.cu"
MATH = ROOT / "experiments/frontier/cutlass_geglu_erf_math.cuh"


def main():
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    binary = DIRECTORY / "domain.so"
    command = [
        "/usr/local/cuda-13.1/bin/nvcc",
        "-O3",
        "-std=c++20",
        "-arch=sm_120",
        "--compiler-options",
        "-fPIC",
        "-shared",
        str(SOURCE),
        "-o",
        str(binary),
    ]
    with Path("/tmp/laya-gpu-experiments.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH)
        build = subprocess.run(command, check=True, capture_output=True, text=True)
        (DIRECTORY / "domain-build.log").write_text(build.stdout + build.stderr)
    with Path("/tmp/laya-gpu-experiments.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        torch.set_num_threads(4)
        library = ctypes.CDLL(str(binary))
        library.cutlass_geglu_erf_domain.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        output = torch.empty(65536, device="cuda", dtype=torch.int16)
        error = library.cutlass_geglu_erf_domain(
            output.data_ptr(), torch.cuda.current_stream().cuda_stream
        )
        if error:
            raise RuntimeError(f"Domain launch failed: {error}")
        reference = gelu_lut().view(torch.int16)
        bad = (output != reference).nonzero().flatten()
        codes = bad.cpu().tolist()
        values = [int(x) & 65535 for x in reference[bad].cpu().tolist()]
        actual = [int(x) & 65535 for x in output[bad].cpu().tolist()]
    report = {
        "scope": "All 65,536 BF16 bit patterns, compared bitwise with installed PyTorch GELU table",
        "domain_size": 65536,
        "correction_input_bits": codes,
        "correction_output_bits": values,
        "uncorrected_output_bits": actual,
        "command": command,
        "library_sha256": digest(binary),
        "source_sha256": {
            str(p.relative_to(ROOT)): digest(p) for p in [SOURCE, MATH, Path(__file__)]
        },
    }
    path = ROOT / "results/frontier/cutlass-geglu-erf-domain.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    if len(codes) > 16:
        raise RuntimeError(
            f"Native expression needs {len(codes)} corrections; inspect before proceeding"
        )
    header = [
        "// Exhaustive BF16 domain-derived corrections. See cutlass-geglu-erf-domain.json.",
        "__device__ __forceinline__ __nv_bfloat16 corrected_erf_gelu(__nv_bfloat16 input) {",
        "  unsigned short bits=__bfloat16_as_ushort(input);",
    ]
    for code, value in zip(codes, values):
        header.append(f"  if(bits=={code})return __ushort_as_bfloat16({value});")
    header.extend(["  return native_erf_gelu(input);", "}"])
    (DIRECTORY / "cutlass_geglu_erf_corrections.cuh").write_text(
        "\n".join(header) + "\n"
    )
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
