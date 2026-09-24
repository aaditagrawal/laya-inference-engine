"""Build the warp-specialized exact BF16 producer/consumer experiment."""

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / ".research/frontier-ws-lossless"
SOURCE = ROOT / "experiments/frontier/ws_lossless.cu"


def main():
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    library = DIRECTORY / "ws_lossless.so"
    cuda = Path("/usr/local/cuda-13.1")
    command = [
        str(cuda / "bin/nvcc"),
        "-O3",
        "-std=c++20",
        "-gencode=arch=compute_120,code=sm_120",
        "--compiler-options",
        "-fPIC",
        "--ptxas-options=-v",
        "-lineinfo",
        "-shared",
        str(SOURCE),
        "-o",
        str(library),
    ]
    build = subprocess.run(command, check=True, text=True, capture_output=True)
    (DIRECTORY / "build.log").write_text(build.stdout + build.stderr)
    sass = subprocess.run(
        [str(cuda / "bin/cuobjdump"), "--dump-sass", str(library)],
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    (DIRECTORY / "ws_lossless.sass").write_text(sass)
    report = {
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
        "sass_sha256": hashlib.sha256(sass.encode()).hexdigest(),
        "architecture": "sm_120",
        "command": command,
    }
    (DIRECTORY / "build.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
