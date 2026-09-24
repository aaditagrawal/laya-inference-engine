"""Compare the opt-in engine with the existing engine in randomized blocks.

Invoke under flock /tmp/laya-gpu-experiments.lock. Timed calls include
preparation, inference and response formatting. HTTP and startup are separate.
"""

import argparse
import json
import time
from pathlib import Path

import torch

from . import common
from .engine import ExperimentalEngine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["native", "native-window", "compiled", "autotuned"],
        default="native-window",
    )
    parser.add_argument("--kernel", default="cuda_vector_norm_triton_geglu_corrected")
    parser.add_argument(
        "--cases", nargs="+", default=["1-short", "16-short", "1-long", "16-long"]
    )
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--max-graphs", type=int, default=8)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for name in ("rounds", "repeats", "threads", "max_graphs"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    cases = []
    for case in args.cases:
        batch_text, separator, length = case.partition("-")
        if not separator or not batch_text.isdecimal() or int(batch_text) < 1:
            parser.error("cases must have a positive question count, e.g. 16-short")
        if length not in {"short", "medium", "long"}:
            parser.error("case length must be short, medium or long")
        if int(batch_text) > 64:
            parser.error("case question count exceeds the tested 64-question limit")
        cases.append((int(batch_text), length))
    torch.set_num_threads(args.threads)
    report = {"metadata": common.metadata(), "mode": args.mode, "kernel": args.kernel}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    baseline = candidate = None
    try:
        baseline = common.BlackwellEngine(
            common.model_path(), max_graphs=args.max_graphs
        )
        start = time.perf_counter()
        candidate = ExperimentalEngine(
            model=common.model_path(),
            mode=args.mode,
            kernel=args.kernel,
            max_graphs=args.max_graphs,
        )
        report["candidate_load_setup_ms"] = (time.perf_counter() - start) * 1000
        report["experimental_kernel"] = getattr(
            candidate.base, "experimental_kernel", None
        )
        report["experimental_compiler"] = getattr(
            candidate.base, "experimental_compiler", None
        )
        report["first_shape_ms"] = {"baseline": {}, "candidate": {}}
        report["benchmark_output_checks"] = []
        for batch, length in cases:
            outputs = []
            for name, engine in (("baseline", baseline), ("candidate", candidate)):
                request = common.workload(batch, length)
                start = time.perf_counter()
                engine.predict(**request)
                report["first_shape_ms"][name][f"{batch}-{length}"] = (
                    time.perf_counter() - start
                ) * 1000
                outputs.append(engine.run_prepared(engine.prepare(**request)))
            parity = common.compare_outputs(
                outputs[1], outputs[0], candidate.prepare(**request), candidate.agent
            )
            report["benchmark_output_checks"].append(
                {"case": f"{batch}-{length}", **parity}
            )
            if not parity["passed"] or (
                args.mode != "autotuned" and not parity["exact_logits_and_actions"]
            ):
                raise RuntimeError(f"Output mismatch for {batch}-{length}")
        report["timings"] = common.benchmark_interleaved(
            {"baseline": baseline, "candidate": candidate},
            cases=cases,
            rounds=args.rounds,
            repeats=args.repeats,
            warmups=5,
        )
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        if args.validate:
            for name, engine in (
                ("baseline_validation", baseline),
                ("validation", candidate),
            ):
                validation = common.validate(engine)
                report[name] = validation
                required = ["passed_current_engine", "passed_sdk", "passed_actions"]
                if args.mode != "autotuned" or name == "baseline_validation":
                    required += ["exact_logits_all", "exact_actions_all"]
                if not all(validation[field] for field in required):
                    raise RuntimeError(
                        f"{name} failed numerical/SDK validation: {required}"
                    )
        report["status"] = "complete"
    except BaseException as exc:
        import traceback

        report.update(status="failed", error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        try:
            if candidate is not None:
                candidate.close()
        finally:
            try:
                if baseline is not None:
                    baseline.close()
            finally:
                args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
