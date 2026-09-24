"""Reuse the established request/error suite for direct special-token lookup.

Run under the exclusive experiment lock. The existing probe's `single` column
is remapped to retained batch preparation, and `batch` is this candidate.
"""

import hashlib
import json
import sys
from pathlib import Path

from . import host_probe
from .header_prepare import prepare


def main():
    output = Path("results/frontier/header-prepare.json")
    original = host_probe.prepare

    def candidate(*args, mode="single", templates=None, **kwargs):
        if mode == "batch":
            return prepare(*args, **kwargs)
        return original(
            *args,
            mode="batch" if mode == "single" else mode,
            templates=templates,
            **kwargs,
        )

    # This standalone probe owns its module namespace; engine globals stay intact.
    host_probe.prepare = candidate
    sys.argv = [__file__, "--output", str(output)]
    host_probe.main()
    report = json.loads(output.read_text())
    report["column_meaning"] = {
        "baseline": "original offset-producing tokenizer adapter",
        "single": "retained batched offset-free preparation",
        "batch": "direct special-token lookup candidate",
        "template": "existing immutable grammar-token experiment",
    }
    report["source_sha256"].update(
        {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path("experiments/frontier").glob("header_prepare*.py"))
        }
    )
    output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
