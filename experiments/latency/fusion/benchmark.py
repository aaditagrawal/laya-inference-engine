"""Numerical gates and randomized full-request comparisons for fusion variants."""

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch

from experiments.latency.serving.graph_adapter import replace_adapter
from experiments.native.common import (
    CASES,
    benchmark_interleaved,
    compare_outputs,
    metadata,
    validate,
)
from experiments.native.engine import ExperimentalEngine
from laya_blackwell.workloads import workload

from .adapter import install


def build_engine():
    engine = ExperimentalEngine()
    try:
        return replace_adapter(engine)
    except BaseException:
        engine.close()
        raise


def benchmark_parity(baseline, candidate, cases):
    rows = []
    for batch, length in cases:
        original = workload(batch, length)
        changed = {
            **original,
            "state": original["state"]
            .replace("Refund", "Cancel")
            .replace("interruption", "restoration"),
        }
        for label, request in [
            ("original", original),
            ("changed", changed),
            ("original-replay", original),
        ]:
            prepared = candidate.prepare(**request)
            baseline_prepared = baseline.prepare(**request)
            if prepared.items != baseline_prepared.items:
                raise RuntimeError("Baseline and candidate tokenization diverged")
            expected = baseline.run_prepared(baseline_prepared)
            actual = candidate.run_prepared(prepared)
            row = {
                "case": f"{batch}-{length}",
                "replay": label,
                **compare_outputs(actual[:2], expected[:2], prepared, candidate.agent),
                "exact_logits": bool(np.array_equal(actual[0], expected[0])),
                "exact_actions": bool(np.array_equal(actual[1], expected[1])),
                "candidate_info": actual[2],
                "baseline_info": expected[2],
            }
            rows.append(row)
            print(json.dumps(row), flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument(
        "--case",
        choices=["all", "1-short", "16-short", "1-long", "16-long"],
        default="all",
    )
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=3)
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(1701)
    config = json.loads(args.config.read_text())
    report = {
        "metadata": metadata(),
        "graph_adapter": "StableHostAdapter with a unique CUDA stream per graph",
        "configuration": config,
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in Path(__file__).parent.glob("*.py")
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2))

    with torch.inference_mode():
        start = time.perf_counter()
        candidate = build_engine()
        install(candidate, **config)
        report["setup_seconds"] = time.perf_counter() - start
        report["fusion"] = candidate.fusion_metadata
        save()
        try:
            if not args.skip_validation:
                start = time.perf_counter()
                report["validation"] = validate(candidate)
                report["validation_seconds"] = time.perf_counter() - start
                save()
                print(
                    json.dumps(
                        {
                            key: val
                            for key, val in report["validation"].items()
                            if key != "cases"
                        }
                    ),
                    flush=True,
                )
            if not args.validation_only:
                baseline = build_engine()
                try:
                    cases = (
                        CASES
                        if args.case == "all"
                        else [(int(args.case.split("-")[0]), args.case.split("-")[1])]
                    )
                    report["baseline_reference_probe"] = validate(
                        baseline, indices=[0, 9, 31, 65]
                    )
                    report["benchmark_parity"] = benchmark_parity(
                        baseline, candidate, cases
                    )
                    save()
                    report["benchmark"] = benchmark_interleaved(
                        {"native-window": baseline, "fusion": candidate},
                        cases=cases,
                        rounds=args.rounds,
                        repeats=args.repeats,
                    )
                    save()
                finally:
                    baseline.close()
        finally:
            candidate.close()
            del candidate
            gc.collect()
            torch.cuda.empty_cache()
    print(args.output, flush=True)


if __name__ == "__main__":
    main()
