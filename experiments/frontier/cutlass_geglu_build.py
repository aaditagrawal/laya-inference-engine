"""Build native CUTLASS mainloops and a directly fused exact GEGLU visitor."""

import fcntl
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / ".research/frontier-cutlass-geglu"
SOURCE = ROOT / "experiments/frontier/cutlass_geglu.cu"
CUTLASS = ROOT / ".research/frontier-torch-cutlass"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    library = DIRECTORY / "cutlass_geglu.so"
    cuda = Path("/usr/local/cuda-13.1")
    command = [
        str(cuda / "bin/nvcc"),
        "-O3",
        "-std=c++20",
        "-arch=sm_120",
        "--compiler-options",
        "-fPIC",
        "--ptxas-options=-v",
        "-lineinfo",
        "--expt-relaxed-constexpr",
        "-I" + str(CUTLASS / "include"),
        "-shared",
        str(SOURCE),
        "-o",
        str(library),
    ]
    with Path("/tmp/laya-gpu-experiments.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_SH)
        build = subprocess.run(command, text=True, capture_output=True, check=False)
        (DIRECTORY / "build.log").write_text(build.stdout + build.stderr)
        if build.returncode:
            print(build.stderr[-14000:])
            raise SystemExit(build.returncode)
        sass = subprocess.check_output(
            [str(cuda / "bin/cuobjdump"), "-sass", str(library)], text=True
        )
        (DIRECTORY / "cutlass_geglu.sass").write_text(sass)
        dependencies = subprocess.check_output(
            [
                str(cuda / "bin/nvcc"),
                "-std=c++20",
                "-arch=sm_120",
                "--expt-relaxed-constexpr",
                "-I" + str(CUTLASS / "include"),
                "-M",
                str(SOURCE),
            ],
            text=True,
        )
    (DIRECTORY / "dependencies.txt").write_text(dependencies)
    report = {
        "source_sha256": digest(SOURCE),
        "library_sha256": digest(library),
        "sass_sha256": digest(DIRECTORY / "cutlass_geglu.sass"),
        "builder_sha256": digest(Path(__file__)),
        "command": command,
        "cutlass_revision": subprocess.check_output(
            ["git", "-C", str(CUTLASS), "rev-parse", "HEAD"], text=True
        ).strip(),
        "cutlass_headers_sha256": {
            str(p.relative_to(CUTLASS)): digest(p)
            for p in sorted((CUTLASS / "include").rglob("*"))
            if p.is_file() and str(p) in dependencies.replace("\\ ", " ")
        },
    }
    assert report["cutlass_headers_sha256"]
    (DIRECTORY / "build.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in report.items() if k != "cutlass_headers_sha256"}),
        flush=True,
    )


if __name__ == "__main__":
    main()
