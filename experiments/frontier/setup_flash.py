"""Record optional CuTe dependencies from a uv overlay, without changing .venv.

Run through uv --with .research/frontier-flash-attention/flash_attn/cute
--with 'nvidia-cutlass-dsl[cu13]>=4.6.2'. The probe imports base Torch first
and appends these dependency paths, keeping the measured Torch/CUDA stack fixed.
"""

import json
import sys
from pathlib import Path

import torch
from flash_attn.cute import interface


def main():
    report = {
        "dependency_paths": [p for p in sys.path if "site-packages" in p],
        "setup_torch": str(torch.__version__),
        "flash_interface": interface.__file__,
        "note": "This overlay is for dependency setup only. Its Torch build must not be used for a benchmark comparison.",
    }
    Path(".research/frontier-flash-dependencies.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(report, flush=True)


if __name__ == "__main__":
    main()
