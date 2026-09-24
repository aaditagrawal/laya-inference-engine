"""Build the direct-fragment lossless experiment for SM120."""

import hashlib
import json
import os
import subprocess
from pathlib import Path
from time import perf_counter

import torch
from torch.utils.cpp_extension import include_paths

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / ".research/frontier-native-lossless"
SOURCE = ROOT / "experiments/frontier/native_lossless.cu"


def main():
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    cuda = Path("/usr/local/cuda-13.1")
    obj, library = DIRECTORY / "kernel.o", DIRECTORY / "laya_native_lossless.so"
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
            "--ptxas-options=-v",
            *["-I" + p for p in include_paths(device_type="cuda")],
            "-c",
            str(SOURCE),
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
            "-L" + str(Path(torch.__file__).parent / "lib"),
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
    report = {
        "seconds": perf_counter() - start,
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "torch_git": torch.version.git_version,
        "architecture": "sm_120",
    }
    (DIRECTORY / "build.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
