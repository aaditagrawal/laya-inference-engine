"""Build research CUDA kernels without Ninja or shell path quoting.

Run under the shared experiment lock before benchmarks. Defaults to SM120.
"""

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path

import torch
from torch.utils.cpp_extension import CUDA_HOME, include_paths

HERE = Path(__file__).resolve().parent
VARIANTS = {
    "native": ("native.cu", "laya_native_exp"),
    "vector": ("native_vector.cu", "laya_native_vector"),
}


def library_path(variant="vector", arch="120"):
    version = str(torch.__version__).replace("+", "_")
    directory = (
        HERE.parents[2]
        / ".research/native-build/kernels"
        / f"sm_{arch}"
        / version
        / variant
    )
    return directory / f"{VARIANTS[variant][1]}.so"


def fingerprint(variant="vector", arch="120"):
    source = HERE / VARIANTS[variant][0]
    return {
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "builder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "arch": arch,
        "torch": str(torch.__version__),
        "torch_cuda": torch.version.cuda,
        "torch_git": torch.version.git_version,
        "cxx11_abi": int(torch._C._GLIBCXX_USE_CXX11_ABI),
        "cxx": os.environ.get("CXX", "c++"),
        "nvcc": subprocess.run(
            [str(Path(CUDA_HOME) / "bin/nvcc"), "--version"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout,
    }


def current_library(variant="vector", arch="120"):
    shared = library_path(variant, arch)
    manifest = shared.parent / "build.json"
    if not shared.exists() or not manifest.exists():
        raise RuntimeError(f"Run kernels/build.py --variant {variant} first")
    saved = json.loads(manifest.read_text())
    if saved.get("fingerprint") != fingerprint(variant, arch):
        raise RuntimeError(
            f"Stale CUDA build; run kernels/build.py --variant {variant}"
        )
    return shared


def build(variant="vector", arch="120"):
    if CUDA_HOME is None:
        raise RuntimeError("A CUDA toolkit with nvcc is required")
    if not re.fullmatch(r"[0-9]+[af]?", arch):
        raise ValueError("Expected a CUDA architecture such as120")
    source = HERE / VARIANTS[variant][0]
    shared = library_path(variant, arch)
    directory = shared.parent
    directory.mkdir(parents=True, exist_ok=True)
    obj = directory / "kernel.o"
    source_hash = hashlib.sha256(source.read_bytes()).hexdigest()
    configuration = fingerprint(variant, arch)
    manifest = directory / "build.json"
    if shared.exists() and manifest.exists():
        previous = json.loads(manifest.read_text())
        if previous.get("fingerprint") == configuration:
            return shared
    includes = [
        arg for path in include_paths(device_type="cuda") for arg in ["-isystem", path]
    ]
    temporary_shared = shared.with_suffix(f".so.tmp.{os.getpid()}")
    subprocess.run(
        [
            str(Path(CUDA_HOME) / "bin/nvcc"),
            "-O3",
            "-lineinfo",
            "-std=c++20",
            "--expt-relaxed-constexpr",
            "--compiler-options",
            "-fPIC",
            f"-gencode=arch=compute_{arch},code=sm_{arch}",
            f"-D_GLIBCXX_USE_CXX11_ABI={int(torch._C._GLIBCXX_USE_CXX11_ABI)}",
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
    torch_lib = Path(torch.__file__).parent / "lib"
    subprocess.run(
        [
            os.environ.get("CXX", "c++"),
            str(obj),
            "-shared",
            f"-L{torch_lib}",
            "-lc10",
            "-lc10_cuda",
            "-ltorch_cpu",
            "-ltorch_cuda",
            "-ltorch",
            "-ltorch_python",
            f"-L{Path(CUDA_HOME) / 'lib64'}",
            "-lcudart",
            "-o",
            str(temporary_shared),
        ],
        check=True,
    )
    os.replace(temporary_shared, shared)
    manifest.write_text(
        json.dumps(
            {
                "source": source.name,
                "fingerprint": configuration,
                "source_sha256": source_hash,
                "shared_sha256": hashlib.sha256(shared.read_bytes()).hexdigest(),
                "arch": arch,
                "torch": str(torch.__version__),
                "torch_cuda": torch.version.cuda,
                "nvcc": subprocess.run(
                    [str(Path(CUDA_HOME) / "bin/nvcc"), "--version"],
                    capture_output=True,
                    text=True,
                    check=True,
                ).stdout,
                "flags": [
                    "-O3",
                    "-lineinfo",
                    "-std=c++20",
                    "FP32 Welford and RN-even BF16 conversion",
                ],
            },
            indent=2,
        )
        + "\n"
    )
    return shared


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--variant", choices=[*VARIANTS, "all"], default="vector")
    parser.add_argument("--arch", default="120")
    args = parser.parse_args()
    for variant in VARIANTS if args.variant == "all" else [args.variant]:
        print(build(variant, args.arch))
