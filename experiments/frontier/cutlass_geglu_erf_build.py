"""Build the best native tile with matched LUT and corrected-erf epilogues."""

import fcntl
import json
import subprocess
from pathlib import Path

from .cutlass_geglu_build import CUTLASS, ROOT, digest
from .cutlass_geglu_build import SOURCE as ORIGINAL
from .cutlass_geglu_erf_domain import DIRECTORY, MATH

SOURCE = ROOT / "experiments/frontier/cutlass_geglu_erf.cu"


def main():
    core = DIRECTORY / "cutlass_geglu_core.cuh"
    core.write_text(ORIGINAL.read_text().split('extern "C" int cutlass_geglu(')[0])
    corrections = DIRECTORY / "cutlass_geglu_erf_corrections.cuh"
    binary = DIRECTORY / "cutlass_geglu_erf.so"
    command = [
        "/usr/local/cuda-13.1/bin/nvcc",
        "-O3",
        "-std=c++20",
        "-arch=sm_120",
        "--compiler-options",
        "-fPIC",
        "--ptxas-options=-v",
        "-lineinfo",
        "--expt-relaxed-constexpr",
        "-I" + str(CUTLASS / "include"),
        "-I" + str(DIRECTORY),
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
    sass_path = DIRECTORY / "cutlass_geglu_erf.sass"
    sass_path.write_text(sass)
    report = {
        "source_sha256": {
            str(p.relative_to(ROOT)): digest(p)
            for p in [SOURCE, ORIGINAL, MATH, Path(__file__), core, corrections]
        },
        "library_sha256": digest(binary),
        "sass_sha256": digest(sass_path),
        "command": command,
        "initial_build": json.loads(
            (ROOT / ".research/frontier-cutlass-geglu/build.json").read_text()
        ),
        "domain_report_sha256": digest(
            ROOT / "results/frontier/cutlass-geglu-erf-domain.json"
        ),
    }
    (DIRECTORY / "build.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in report.items() if k != "initial_build"}),
        flush=True,
    )


if __name__ == "__main__":
    main()
