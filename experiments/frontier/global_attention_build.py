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
    flash_checkout = Path(".research/frontier-torch-flash").resolve()
    flash_revision = subprocess.check_output(
        ["git", "-C", str(flash_checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    if flash_revision != "14c377950125c70b7a9dabf9c561fca53715ac7d":
        raise RuntimeError("FlashAttention revision must match pinned Torch submodule")
    flash_headers = flash_checkout / "csrc/flash_attn/src"
    directory = Path(".research/frontier-global-attention").resolve()
    directory.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).with_name("global_attention_kernel.cu").resolve()
    cuda = Path("/usr/local/cuda-13.1")
    torch_root = Path(torch.__file__).parent
    includes = ["-I" + path for path in include_paths(device_type="cuda")]
    includes += ["-I" + str(checkout / "include"), "-I" + str(flash_headers)]
    obj = directory / "attention.o"
    library = directory / "laya_global_attention.so"
    start = perf_counter()
    subprocess.run(
        [
            str(cuda / "bin/nvcc"),
            "-O3",
            "-DFLASH_NAMESPACE=laya_global_flash",
            "-DFLASHATTENTION_DISABLE_DROPOUT",
            "-DUNFUSE_FMA",
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
        "flash_revision": flash_revision,
        "flash_headers_sha256": {
            str(p.relative_to(flash_checkout)): hashlib.sha256(
                p.read_bytes()
            ).hexdigest()
            for p in sorted(flash_headers.glob("*"))
            if p.is_file()
        },
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "architecture": "sm_120",
        "fast_math": False,
    }
    (directory / "build.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report, flush=True)


if __name__ == "__main__":
    main()
