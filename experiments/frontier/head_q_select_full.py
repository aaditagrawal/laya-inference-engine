"""Full-request parity and paired confirmation of final-head selected Q rows."""

import json
import random
import statistics
import time
from pathlib import Path

import torch

from experiments.native import common

from .engine import FrontierEngine
from .head_q_select_adapter import install
from .head_q_select_kernel import binary_hashes
from .head_q_select_probe import source_hashes
from .holdout import requests


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    constructor = json.loads(Path("results/frontier/summary.json").read_text())[
        "recommended_constructor"
    ]
    path = Path("results/frontier/head_q_select-full.json")
    before = source_hashes()
    report = {
        "metadata": common.metadata(),
        "constructor": constructor,
        "source_before": before,
        "parity": [],
    }

    def save():
        path.write_text(json.dumps(report, indent=2) + "\n")

    with (
        FrontierEngine(**constructor, max_graphs=6) as baseline,
        FrontierEngine(**constructor, max_graphs=6) as candidate,
    ):
        report["installation"] = install(candidate, 32)
        extras = json.loads(
            Path("results/frontier/attention-special-screen.json").read_text()
        )["requests"][1:]
        for suite, fixtures in [
            ("original", common.validation_requests()),
            ("holdout", requests()),
            ("extra-fully-occupied", extras),
            (
                "benchmark",
                [
                    common.workload(b, length)
                    for b, length in [(1, "short"), (1, "long"), (16, "short")]
                ],
            ),
        ]:
            for index, request in enumerate(fixtures):
                prepared = candidate.prepare(**request)
                actual = candidate.run_prepared(prepared)
                reference = baseline.run_prepared(baseline.prepare(**request))
                report["parity"].append(
                    {
                        "suite": suite,
                        "index": index,
                        "graph_key": candidate.base._graph_key(prepared),
                        **common.compare_outputs(
                            actual, reference, prepared, candidate.agent
                        ),
                    }
                )
        report["all_exact"] = all(
            r["exact_logits_and_actions"] for r in report["parity"]
        )
        print("Full parity", len(report["parity"]), report["all_exact"], flush=True)
        report["binary_before_timing"] = binary_hashes()
        save()
        if not report["all_exact"]:
            return
        engines = {"retained": baseline, "selected-q": candidate}
        report["fixed"] = common.benchmark_interleaved(
            engines, cases=[(1, "short")], rounds=15, repeats=200
        )
        report["changing"] = []
        rng = random.Random(75163)
        for case, fixtures in [("holdout", requests()), ("full-attention", extras)]:
            for engine in engines.values():
                for request in fixtures:
                    engine.predict(**request)
            for round_id in range(15):
                indices = list(range(len(fixtures)))
                rng.shuffle(indices)
                order = list(engines)
                rng.shuffle(order)
                for variant in order:
                    samples = []
                    for index in indices:
                        start = time.perf_counter()
                        engines[variant].predict(**fixtures[index])
                        samples.append((time.perf_counter() - start) * 1000)
                    report["changing"].append(
                        {
                            "case": case,
                            "round": round_id,
                            "variant": variant,
                            "indices": indices,
                            "order": order,
                            **common.stats(samples),
                        }
                    )
        report["summary"] = {}
        for case in ["1-short", "holdout", "full-attention"]:
            rows = [
                r
                for r in report["fixed"]["rows"] + report["changing"]
                if r["case"] == case
            ]
            by_round = {
                name: {r["round"]: r["p50_ms"] for r in rows if r["variant"] == name}
                for name in engines
            }
            savings = [
                by_round["retained"][i] - by_round["selected-q"][i] for i in range(15)
            ]
            report["summary"][case] = {
                "pooled_p50_ms": {
                    name: statistics.median(
                        v for r in rows if r["variant"] == name for v in r["samples_ms"]
                    )
                    for name in engines
                },
                "paired_savings_ms": savings,
                "faster_rounds": sum(s > 0 for s in savings),
                "median_paired_savings_ms": statistics.median(savings),
            }
        report["source_after"] = source_hashes()
        report["binary_after_timing"] = binary_hashes()
        assert report["source_after"] == before
        assert report["binary_before_timing"] == report["binary_after_timing"]
        save()
        print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
