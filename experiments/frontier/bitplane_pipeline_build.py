"""Build the native asynchronous exact bitplane GEMM experiment."""

import hashlib
import json
import subprocess
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / ".research/frontier-bitplane-pipeline"
SOURCE = ROOT / "experiments/frontier/bitplane_pipeline.cu"


def main():
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    library = DIRECTORY / "bitplane_pipeline.so"
    cuda = Path("/usr/local/cuda-13.1")
    command = [
        str(cuda / "bin/nvcc"),
        "-O3",
        "-std=c++20",
        "-gencode=arch=compute_120,code=sm_120",
        "--compiler-options",
        "-fPIC",
        "--ptxas-options=-v",
        "-shared",
        str(SOURCE),
        "-o",
        str(library),
    ]
    start = perf_counter()
    subprocess.run(command, check=True)
    report = {
        "seconds": perf_counter() - start,
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "architecture": "sm_120",
        "command": command,
    }
    (DIRECTORY / "build.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
