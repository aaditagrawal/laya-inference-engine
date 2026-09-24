"""Run under flock /tmp/laya-gpu-experiments.lock using uv."""

import argparse
import hashlib
import json
import random
import time
import traceback
from pathlib import Path

import torch

from experiments.native import common

from .engine import POLICIES, DenseEngine


def benchmark_requests(engines, requests, *, rounds, repeats):
    rng = random.Random(982451653)
    report = {"rows": [], "order": [], "rounds": rounds, "repeats_per_round": repeats}
    for name, request in requests.items():
        for engine in engines.values():
            engine.predict(**request)
        for round_id in range(rounds):
            variants = list(engines)
            rng.shuffle(variants)
            report["order"].append({"case": name, "round": round_id, "names": variants})
            for variant in variants:
                engine = engines[variant]
                for _ in range(5):
                    engine.predict(**request)
                samples = []
                for _ in range(repeats):
                    start = time.perf_counter()
                    response = engine.predict(**request)
                    samples.append((time.perf_counter() - start) * 1000)
                row = {
                    "case": name,
                    "variant": variant,
                    "round": round_id,
                    "questions": len(request["questions"]),
                    "input_tokens": response["usage"]["input_tokens"],
                    **common.stats(samples),
                }
                report["rows"].append(row)
                print(variant, name, round_id, row["p50_ms"], flush=True)
    return report


def sequence_request(engine, count, target):
    request = {
        "state": common.STATES[0],
        "questions": {f"q{i}": common.QUESTIONS["department"] for i in range(count)},
    }
    initial = len(engine.prepare(**request).items[0]["ids"])
    request["state"] += " context" * max(0, target - initial)
    actual = len(engine.prepare(**request).items[0]["ids"])
    if actual != target:
        raise RuntimeError(f"Token length construction changed: {actual} != {target}")
    return request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", choices=sorted(POLICIES - {"stock"}), required=True)
    parser.add_argument(
        "--cases",
        nargs="+",
        default=[
            "1-short",
            "5-short",
            "17-short",
            "33-short",
            "1-medium",
            "5-medium",
            "17-medium",
            "5-long",
            "17-long",
            "16-long",
        ],
    )
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--sequence-sweep", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.rounds < 1 or args.repeats < 1:
        parser.error("rounds and repeats must be positive")
    cases = []
    for case in args.cases:
        batch, _, length = case.partition("-")
        if (
            not batch.isdecimal()
            or not 1 <= int(batch) <= 64
            or length not in {"short", "medium", "long"}
        ):
            parser.error("Invalid benchmark case")
        cases.append((int(batch), length))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    report = {"metadata": common.metadata(), "policy": args.policy, "status": "running"}

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    try:
        with (
            DenseEngine(
                model=common.model_path(), shape_policy="stock", max_graphs=2
            ) as baseline,
            DenseEngine(
                model=common.model_path(), shape_policy=args.policy, max_graphs=2
            ) as candidate,
        ):
            if args.sequence_sweep:
                requests = {
                    f"{batch}-tokens{length}": sequence_request(baseline, batch, length)
                    for length in (80, 144, 272, 400)
                    for batch in (1, 5)
                }
            else:
                requests = {
                    f"{batch}-{length}": common.workload(batch, length)
                    for batch, length in cases
                }
            report["benchmark_checks"] = []
            for case, request in requests.items():
                prepared = baseline.prepare(**request)
                expected = baseline.run_prepared(prepared)
                actual = candidate.run_prepared(prepared)
                parity = common.compare_outputs(
                    actual, expected, prepared, candidate.agent
                )
                tokens = sum(len(item["ids"]) for item in prepared.items)
                report["benchmark_checks"].append(
                    {
                        "case": case,
                        "request_sha256": hashlib.sha256(
                            json.dumps(request, sort_keys=True).encode()
                        ).hexdigest(),
                        "input_tokens": tokens,
                        "baseline_shape": expected[2]["shape"],
                        "candidate_shape": actual[2]["shape"],
                        **parity,
                    }
                )
                print("parity", report["benchmark_checks"][-1], flush=True)
                save()
            # Measure even rejected numerical variants, but label every gate explicitly.
            report["timings"] = benchmark_requests(
                {"baseline": baseline, "candidate": candidate},
                requests,
                rounds=args.rounds,
                repeats=args.repeats,
            )
            save()
            if args.validate:
                started = time.perf_counter()
                report["validation"] = common.validate(candidate)
                report["validation_ms"] = (time.perf_counter() - started) * 1000
            report["status"] = "complete"
    except BaseException as exc:
        report.update(status="failed", error=str(exc), traceback=traceback.format_exc())
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
