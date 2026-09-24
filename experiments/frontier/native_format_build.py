"""Build the CPU formatter under the shared experiment lock."""

import hashlib
import json
import subprocess
import sysconfig
import time
from pathlib import Path

import numpy as np
import torch


def main():
    root = Path(__file__).resolve().parents[2]
    source = Path(__file__).with_suffix(".cpp").with_name("native_format.cpp")
    destination = root / ".research/frontier-native-format"
    destination.mkdir(parents=True, exist_ok=True)
    output = destination / (
        "laya_native_format" + sysconfig.get_config_var("EXT_SUFFIX")
    )
    command = [
        "c++",
        "-O3",
        "-DNDEBUG",
        "-std=c++17",
        "-shared",
        "-fPIC",
        "-ffp-contract=off",
        "-fno-fast-math",
        str(source),
        "-I" + str(Path(torch.__file__).parent / "include"),
        "-I" + sysconfig.get_paths()["include"],
        "-I" + np.get_include(),
        "-o",
        str(output),
    ]
    start = time.perf_counter()
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    record = {
        "command": command,
        "seconds": time.perf_counter() - start,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
        "numpy": np.__version__,
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "library_sha256": hashlib.sha256(output.read_bytes()).hexdigest()
        if output.is_file()
        else None,
        "artifact": str(output),
    }
    (destination / "build.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2), flush=True)
    completed.check_returncode()


if __name__ == "__main__":
    main()
