"""Build the exact duplicated-Welford prologue feasibility control."""

import fcntl
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / ".research/frontier-norm-geglu"
SOURCE = ROOT / "experiments/frontier/norm_geglu_prologue.cu"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    binary = DIRECTORY / "norm_geglu.so"
    command = [
        "/usr/local/cuda-13.1/bin/nvcc",
        "-O3",
        "-std=c++20",
        "-arch=sm_120",
        "--compiler-options",
        "-fPIC",
        "--ptxas-options=-v",
        "-lineinfo",
        "-shared",
        str(SOURCE),
        "-o",
        str(binary),
    ]
    with Path("/tmp/laya-gpu-experiments.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH)
        build = subprocess.run(command, capture_output=True, text=True, check=False)
        (DIRECTORY / "build.log").write_text(build.stdout + build.stderr)
        if build.returncode:
            print(build.stderr[-10000:])
            raise SystemExit(build.returncode)
        sass = subprocess.check_output(
            ["/usr/local/cuda-13.1/bin/cuobjdump", "-sass", str(binary)], text=True
        )
    (DIRECTORY / "norm_geglu.sass").write_text(sass)
    report = {
        "command": command,
        "source_sha256": digest(SOURCE),
        "builder_sha256": digest(Path(__file__)),
        "library_sha256": digest(binary),
        "sass_sha256": digest(DIRECTORY / "norm_geglu.sass"),
    }
    (DIRECTORY / "build.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
