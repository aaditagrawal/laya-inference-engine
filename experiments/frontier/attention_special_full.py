"""Full-request parity and paired timing of the guarded local attention kernel."""

import argparse
import json
import random
import statistics
import time
from pathlib import Path

import torch

from experiments.native import common

from .attention_special_adapter import install
from .engine import FrontierEngine
from .holdout import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-padding", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    options = json.loads(Path("results/frontier/summary.json").read_text())[
        "recommended_constructor"
    ]
    # The retained comparison baseline predates this optional specialization.
    options.pop("attention_special", None)
    config = (32, True, True)
    path = Path(
        "results/frontier/attention-special-full.json"
        if args.include_padding
        else "results/frontier/attention-special-full-unmasked.json"
    )
    report = {
        "metadata": common.metadata(),
        "constructor": options,
        "configuration": config,
        "build": json.loads(
            Path(".research/frontier-attention-special/build.json").read_text()
        ),
    }

    def save():
        path.write_text(json.dumps(report, indent=2) + "\n")

    with (
        FrontierEngine(**options, max_graphs=6) as baseline,
        FrontierEngine(**options, max_graphs=6) as candidate,
    ):
        report["installation"] = install(
            candidate, config, include_padding=args.include_padding
        )
        fixtures = common.validation_requests() + requests()
        extra = json.loads(
            Path("results/frontier/attention-special-screen.json").read_text()
        )["requests"][1:]
        report["parity"] = []
        benchmark_requests = [
            common.workload(b, length)
            for b, length in [(1, "short"), (1, "long"), (16, "short")]
        ]
        for index, request in enumerate(fixtures + extra + benchmark_requests):
            prepared = candidate.prepare(**request)
            actual = candidate.run_prepared(prepared)
            reference = baseline.run_prepared(baseline.prepare(**request))
            key = candidate.base._graph_key(prepared)
            report["parity"].append(
                {
                    "index": index,
                    "suite": "original"
                    if index < 66
                    else "holdout"
                    if index < len(fixtures)
                    else "extra-full-attention"
                    if index < len(fixtures) + len(extra)
                    else "benchmark",
                    "graph_key": key,
                    **common.compare_outputs(
                        actual, reference, prepared, candidate.agent
                    ),
                }
            )
        report["all_exact"] = all(
            r["exact_logits_and_actions"] for r in report["parity"]
        )
        print("Parity", len(report["parity"]), report["all_exact"], flush=True)
        save()
        if not report["all_exact"]:
            return
        engines = {"retained": baseline, "special": candidate}
        report["fixed"] = common.benchmark_interleaved(
            engines,
            cases=[(1, "short"), (1, "long"), (16, "short")],
            rounds=9,
            repeats=100,
        )
        report["changing_inputs"] = {"rows": [], "order": []}
        rng = random.Random(2424)
        for name, changing in [("holdout", requests()), ("full-attention", extra)]:
            for engine in engines.values():
                for request in changing:
                    engine.predict(**request)
            for round_id in range(9):
                indices = list(range(len(changing)))
                rng.shuffle(indices)
                order = list(engines)
                rng.shuffle(order)
                report["changing_inputs"]["order"].append(
                    {
                        "case": name,
                        "round": round_id,
                        "variants": order,
                        "indices": indices,
                    }
                )
                for variant in order:
                    samples = []
                    for index in indices:
                        start = time.perf_counter()
                        engines[variant].predict(**changing[index])
                        samples.append((time.perf_counter() - start) * 1000)
                    report["changing_inputs"]["rows"].append(
                        {
                            "case": name,
                            "round": round_id,
                            "variant": variant,
                            **common.stats(samples),
                        }
                    )
        rows = report["fixed"]["rows"] + report["changing_inputs"]["rows"]
        report["summary"] = {}
        for case in ("1-short", "1-long", "16-short", "holdout", "full-attention"):
            subset = [r for r in rows if r["case"] == case]
            pooled = {
                name: statistics.median(
                    v for r in subset if r["variant"] == name for v in r["samples_ms"]
                )
                for name in engines
            }
            by_round = {
                name: {r["round"]: r["p50_ms"] for r in subset if r["variant"] == name}
                for name in engines
            }
            savings = [
                by_round["retained"][i] - by_round["special"][i] for i in range(9)
            ]
            report["summary"][case] = {
                "median_ms": pooled,
                "paired_savings_ms": savings,
                "faster_rounds": sum(v > 0 for v in savings),
            }
        save()
        print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
