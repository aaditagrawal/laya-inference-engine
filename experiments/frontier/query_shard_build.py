"""Build pinned PyTorch/CUTLASS attention instantiations for SM120."""

import hashlib
import json
import os
import subprocess
from pathlib import Path
from time import perf_counter

import torch
from torch.utils.cpp_extension import include_paths

TORCH_REVISION = "08187d9e0fba026dc8217405802ab5381dc88d90"
CUTLASS_REVISION = "e05f953a5b3d38adc240df2ff928e0421c2abba3"


def main():
    if torch.version.git_version != TORCH_REVISION:
        raise RuntimeError(
            "The attention template experiment requires the pinned Torch revision"
        )
    checkout = Path(".research/frontier-torch-cutlass").resolve()
    revision = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != CUTLASS_REVISION:
        raise RuntimeError(
            "CUTLASS must match the submodule used by the installed Torch build"
        )
    directory = Path(".research/frontier-query-shard").resolve()
    directory.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).with_name("query_shard_kernel.cu").resolve()
    cuda = Path("/usr/local/cuda-13.1")
    torch_root = Path(torch.__file__).parent
    includes = ["-I" + path for path in include_paths(device_type="cuda")]
    includes += ["-I" + str(checkout / "include")]
    obj = directory / "attention.o"
    library = directory / "laya_frontier_query_shard.so"
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
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
            *includes,
            "-c",
            str(source),
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
        "seconds": perf_counter() - start,
        "torch_revision": TORCH_REVISION,
        "cutlass_revision": revision,
        "torch_header_sha256": hashlib.sha256(
            (
                torch_root
                / "include/ATen/native/transformers/cuda/mem_eff_attention/kernel_forward.h"
            ).read_bytes()
        ).hexdigest(),
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "architecture": "sm_120",
        "fast_math": False,
    }
    (directory / "build.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report, flush=True)


if __name__ == "__main__":
    main()
