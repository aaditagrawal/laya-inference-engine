"""Build only the inspected local Turbo-Lossless adaptation for SM120."""

import hashlib
import json
import subprocess
from pathlib import Path
from time import perf_counter

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / ".research/frontier-turbo-lossless"
SOURCES = [
    ROOT / "experiments/frontier" / f"turbo_{name}"
    for name in ["native.cu", "kernel.cuh"]
]
REVISION = "50b72e5520f93cc855f0bd3a0114a7eb97338588"


def main():
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    cuda = Path("/usr/local/cuda-13.1")
    library = DIRECTORY / "turbo_lossless.so"
    command = [
        str(cuda / "bin/nvcc"),
        "-O3",
        "-std=c++20",
        "-gencode=arch=compute_120,code=sm_120",
        "--compiler-options",
        "-fPIC",
        "--ptxas-options=-v",
        "-shared",
        str(SOURCES[0]),
        "-L" + str(cuda / "lib64"),
        "-lcudart",
        "-lcuda",
        "-o",
        str(library),
    ]
    start = perf_counter()
    subprocess.run(command, check=True)
    report = {
        "seconds": perf_counter() - start,
        "upstream_revision": REVISION,
        "architecture": "sm_120",
        "source_sha256": {
            s.name: hashlib.sha256(s.read_bytes()).hexdigest() for s in SOURCES
        },
        "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "command": command,
    }
    (DIRECTORY / "build.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
