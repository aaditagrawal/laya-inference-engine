"""Build under the shared GPU/build lock; output remains in .research."""

import json
import subprocess
import sysconfig
from pathlib import Path
from time import perf_counter

import torch


def main():
    cuda = Path("/usr/local/cuda-13.1")
    destination = Path(".research/frontier-build")
    destination.mkdir(parents=True, exist_ok=True)
    command = [
        "c++",
        "-O3",
        "-std=c++17",
        "-shared",
        "-fPIC",
        str(Path(__file__).with_name("lt.cpp")),
        "-I" + str(Path(torch.__file__).parent / "include"),
        "-I" + sysconfig.get_paths()["include"],
        "-I" + str(cuda / "include"),
        "-L" + str(cuda / "lib64"),
        "-Wl,-rpath," + str(cuda / "lib64"),
        "-lcublasLt",
        "-lcudart",
        "-o",
        str(
            destination / ("laya_frontier_lt" + sysconfig.get_config_var("EXT_SUFFIX"))
        ),
    ]
    start = perf_counter()
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    report = {
        "seconds": perf_counter() - start,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "command": command,
    }
    (destination / "build.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)
    result.check_returncode()


if __name__ == "__main__":
    main()
