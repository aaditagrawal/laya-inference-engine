"""Build an isolated nvCOMPDx ANS probe against the pinned research SDK."""

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / ".research/frontier-nvcompdx"
SDK = DIRECTORY / "nvidia-mathdx-26.06.1-cuda13/nvidia/mathdx/26.06"
SOURCE = ROOT / "experiments/frontier/nvcompdx_ans.cu"


def sha(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    archive = ROOT / ".research/frontier-nvcompdx.tar.gz"
    archive_sha = sha(archive)
    if (
        archive_sha
        != "59a9233db34b75568acbcc5284e6cefe6fad5577ee644f85044971d62eeea353"
    ):
        raise RuntimeError("SDK archive differs from pinned NVIDIA release")
    library = DIRECTORY / "nvcompdx_ans.so"
    command = [
        "/usr/local/cuda-13.1/bin/nvcc",
        "-O3",
        "-std=c++20",
        "-arch=sm_120",
        "-rdc=true",
        "-dlto",
        "--expt-relaxed-constexpr",
        "--compiler-options",
        "-fPIC",
        "--ptxas-options=-v",
        "-lineinfo",
        "-I",
        str(SDK / "include"),
        "-shared",
        str(SOURCE),
        str(SDK / "lib/libnvcompdx.a"),
        "-o",
        str(library),
    ]
    build = subprocess.run(command, text=True, capture_output=True, check=False)
    (DIRECTORY / "probe-build.log").write_text(build.stdout + build.stderr)
    if build.returncode:
        raise RuntimeError(build.stderr[-8000:])
    sass = subprocess.run(
        ["/usr/local/cuda-13.1/bin/cuobjdump", "--dump-sass", str(library)],
        check=True,
        text=True,
        capture_output=True,
    ).stdout
    (DIRECTORY / "nvcompdx_ans.sass").write_text(sass)
    vendor = [
        SDK / "include/nvcompdx.hpp",
        SDK / "lib/libnvcompdx.a",
        *(SDK / "include/nvcompdx").rglob("*.hpp"),
        *(SDK / "include/commondx").rglob("*.hpp"),
    ]
    vendor_sha = {str(p.relative_to(SDK)): sha(p) for p in sorted(vendor)}
    report = {
        "source_sha256": sha(SOURCE),
        "library_sha256": sha(library),
        "sdk_archive_sha256": archive_sha,
        "sdk_files_sha256": vendor_sha,
        "sass_sha256": hashlib.sha256(sass.encode()).hexdigest(),
        "sdk_url": "https://developer.download.nvidia.com/compute/nvcompdx/redist/nvcompdx/cuda13/nvidia-mathdx-26.06.1-cuda13.tar.gz",
        "architecture": "sm_120",
        "device_lto": True,
        "command": command,
    }
    (DIRECTORY / "probe-build.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in report.items() if k != "sdk_files_sha256"}),
        flush=True,
    )


if __name__ == "__main__":
    main()
