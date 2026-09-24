"""Replay retained graphs after repeated eviction/capture of a second engine."""

import argparse
import gc
import json
import traceback
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments.latency.padding.engine import DenseEngine
from experiments.native.common import metadata, model_path
from laya_blackwell.workloads import workload

from .graph_adapter import replace_adapter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", action="store_true")
    parser.add_argument("--cycles", type=int, default=12)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = {
        "metadata": metadata(),
        "unique_capture_streams": not args.original,
        "cycles": args.cycles,
        "rows": [],
        "status": "running",
    }

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    try:
        with (
            DenseEngine(
                model=model_path(), shape_policy="stock", max_graphs=8
            ) as baseline,
            DenseEngine(
                model=model_path(), shape_policy="batch-exact", max_graphs=8
            ) as candidate,
        ):
            if not args.original:
                replace_adapter(baseline)
                replace_adapter(candidate)
            requests = [workload(count, "short") for count in range(1, 11)]
            prepared = [baseline.prepare(**request) for request in requests]
            expected = [baseline.run_prepared(p) for p in prepared]
            expected_candidate = [candidate.run_prepared(p) for p in prepared]
            for cycle in range(args.cycles):
                for label, engine, reference in (
                    ("churning", candidate, expected_candidate),
                    ("retained", baseline, expected),
                ):
                    for index, p in enumerate(prepared):
                        report["current"] = {
                            "cycle": cycle,
                            "engine": label,
                            "index": index,
                        }
                        save()
                        start = perf_counter()
                        output = engine.run_prepared(p)
                        check = all(
                            np.array_equal(output[j], reference[index][j])
                            for j in (0, 1)
                        )
                        report["rows"].append(
                            {
                                "cycle": cycle,
                                "engine": label,
                                "index": index,
                                "ms": (perf_counter() - start) * 1000,
                                "exact_logits_and_actions": check,
                                **output[2],
                            }
                        )
                        assert check, "Captured graph output changed"
                gc.collect()
                torch.cuda.empty_cache()
                print(
                    json.dumps(
                        {
                            "cycle": cycle,
                            "rows": len(report["rows"]),
                            "cuda_bytes": torch.cuda.memory_allocated(),
                        }
                    ),
                    flush=True,
                )
        gc.collect()
        report["cuda_bytes_after_close"] = torch.cuda.memory_allocated()
        report["status"] = "passed"
    except BaseException:
        report["status"] = "failed"
        report["traceback"] = traceback.format_exc()
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
