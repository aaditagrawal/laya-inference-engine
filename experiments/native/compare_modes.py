"""Direct native versus compiled comparison after both modes pass validation."""

import argparse
import json
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch

from . import common
from .engine import ExperimentalEngine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    report = {
        "metadata": common.metadata(),
        "purpose": "Direct mode comparison; both modes independently passed the full regression.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        engines = {
            mode: stack.enter_context(ExperimentalEngine(mode=mode, max_graphs=2))
            for mode in ("native-window", "compiled")
        }
        cases = [(1, "short"), (16, "long")]
        for batch, length in cases:
            request = common.workload(batch, length)
            arrays = [
                engine.run_prepared(engine.prepare(**request))
                for engine in engines.values()
            ]
            if not all(np.array_equal(arrays[0][i], arrays[1][i]) for i in (0, 1)):
                raise RuntimeError(f"Output mismatch at {batch}-{length}")
        report["timings"] = common.benchmark_interleaved(
            engines, cases=cases, rounds=5, repeats=30
        )
        report["exact_benchmark_logits_and_actions"] = True
        report["status"] = "complete"
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
