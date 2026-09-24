"""Build the native C++ host path; no GPU work occurs during compilation.

Run with a shared experiment lock so compilation does not disturb measurements:
flock -s /tmp/laya-gpu-experiments.lock uv run --no-sync python experiments/native/host/build.py

Artifacts go to the ignored .research/native-build/host directory by default.
For a custom destination, point LAYA_HOST_EXTENSION at the resulting .so file.
"""

import argparse
import json
import os
import shlex
import shutil
import subprocess
import sysconfig
import time
from pathlib import Path

import torch


def cuda_directory(explicit):
    if explicit:
        return Path(explicit).expanduser().resolve()
    configured = os.environ.get("CUDA_HOME")
    if configured:
        return Path(configured).expanduser().resolve()
    nvcc = shutil.which("nvcc")
    if nvcc:
        return Path(nvcc).resolve().parent.parent
    for candidate in ("/usr/local/cuda-13.1", "/usr/local/cuda"):
        if Path(candidate, "include", "cuda_runtime_api.h").is_file():
            return Path(candidate)
    raise RuntimeError("CUDA headers were not found. Set CUDA_HOME or use --cuda-home.")


def main():
    source = Path(__file__).resolve().parent
    default_output = source.parents[2] / ".research" / "native-build" / "host"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=default_output)
    parser.add_argument("--cuda-home")
    args = parser.parse_args()
    cuda = cuda_directory(args.cuda_home)
    if not (cuda / "include" / "cuda_runtime_api.h").is_file():
        raise RuntimeError(f"CUDA headers are missing under {cuda}")
    destination = args.output_dir.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / ("laya_native_host" + sysconfig.get_config_var("EXT_SUFFIX"))
    compiler = shlex.split(os.environ.get("CXX", "c++"))
    command = compiler + [
        "-O3",
        "-DNDEBUG",
        "-std=c++17",
        "-shared",
        "-fPIC",
        str(source / "native_host.cpp"),
        "-I" + str(Path(torch.__file__).parent / "include"),
        "-I" + sysconfig.get_paths()["include"],
        "-I" + str(cuda / "include"),
        "-L" + str(cuda / "lib64"),
        "-Wl,-rpath," + str(cuda / "lib64"),
        "-lcudart",
        "-o",
        str(output),
    ]
    start = time.perf_counter()
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    record = {
        "command": command,
        "seconds": time.perf_counter() - start,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "artifact": str(output),
    }
    (destination / "build.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2), flush=True)
    result.check_returncode()


if __name__ == "__main__":
    main()
