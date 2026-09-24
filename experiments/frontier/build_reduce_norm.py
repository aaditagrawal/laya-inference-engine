"""Reuse the exact CUDA Welford kernel, loading four rounded GEMM partials.

Generated source retains the original PyTorch attribution and lives under
.research. Build with the shared experiment lock, before GPU benchmarks.
"""

import hashlib
import json
import os
import subprocess
from pathlib import Path
from time import perf_counter

import torch
from torch.utils.cpp_extension import include_paths

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / ".research/frontier-reduce-norm"


def source():
    original = ROOT / "experiments/native/kernels/native_vector.cu"
    text = original.read_text()
    changes = {
        "else rb=reinterpret_cast<const BF4*>(static_cast<const __nv_bfloat16*>(residual)+base)[t+j*128];": """else {
        float sums[4]={0.f,0.f,0.f,0.f};
        #pragma unroll
        for(int part=0;part<4;part++) {
          BF4 piece=reinterpret_cast<const BF4*>(static_cast<const __nv_bfloat16*>(residual)+part*gridDim.x*d+base)[t+j*128];
          #pragma unroll
          for(int k=0;k<4;k++) sums[k]=__fadd_rn(sums[k],__bfloat162float(piece.v[k]));
        }
        #pragma unroll
        for(int k=0;k<4;k++) rb.v[k]=__float2bfloat16_rn(sums[k]);
      }""",
        "r->sizes()==x.sizes()": "r->numel()==4*x.numel()",
        "(r->scalar_type()==at::kFloat || r->scalar_type()==at::kBFloat16)": "(r->scalar_type()==at::kBFloat16)",
        "Matching contiguous FP32/BF16 residual required": "Four contiguous BF16 partial matrices required",
        "TORCH_LIBRARY(laya_native_vector,m)": "TORCH_LIBRARY(laya_frontier_reduce_norm,m)",
    }
    for before, after in changes.items():
        if text.count(before) != 1:
            raise RuntimeError("Native vector source changed; review reduction fusion")
        text = text.replace(before, after)
    return text


def main():
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    generated = source()
    path = DIRECTORY / "kernel.cu"
    path.write_text(generated)
    cuda = Path("/usr/local/cuda-13.1")
    torch_root = Path(torch.__file__).parent
    obj = DIRECTORY / "kernel.o"
    library = DIRECTORY / "laya_frontier_reduce_norm.so"
    start = perf_counter()
    subprocess.run(
        [
            str(cuda / "bin/nvcc"),
            "-O3",
            "-std=c++20",
            "--expt-relaxed-constexpr",
            "-gencode=arch=compute_120,code=sm_120",
            "--compiler-options",
            "-fPIC",
            *["-I" + p for p in include_paths(device_type="cuda")],
            "-c",
            str(path),
            "-o",
            str(obj),
        ],
        check=True,
    )
    subprocess.run(
        [
            os.environ.get("CXX", "c++"),
            str(obj),
            "-shared",
            "-L" + str(torch_root / "lib"),
            "-lc10",
            "-lc10_cuda",
            "-ltorch_cpu",
            "-ltorch_cuda",
            "-ltorch",
            "-ltorch_python",
            "-L" + str(cuda / "lib64"),
            "-lcudart",
            "-o",
            str(library),
        ],
        check=True,
    )
    torch.ops.load_library(str(library))
    report = {
        "source_sha256": hashlib.sha256(generated.encode()).hexdigest(),
        "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "torch_git": torch.version.git_version,
        "architecture": "sm_120",
        "seconds": perf_counter() - start,
    }
    (DIRECTORY / "build.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report, flush=True)


if __name__ == "__main__":
    main()
