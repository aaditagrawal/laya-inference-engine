"""Full-request check of the small isolated split-TMA improvement."""

import hashlib
import json
import random
import statistics
import time
import types
from pathlib import Path

import torch

from experiments.native import common

from .engine import FrontierEngine
from .holdout import requests
from .split_tma import compiled_partial

ROOT = Path(__file__).resolve().parents[2]


def install(engine, config):
    if engine.adapter.graphs or engine.base.graphs:
        raise RuntimeError("Install split TMA before graph capture")
    model = engine.base.model.original
    for layer in model.net.encoder.layers[:-1]:
        module = layer.mlp.Wo
        original = module.forward

        def forward(self, x, original=original):
            if x.numel() == 64 * 2624 and x.is_contiguous():
                return compiled_partial(x, self.weight, list(config))
            return original(x)

        module.forward = types.MethodType(forward, module)


def main():
    torch.set_num_threads(4)
    ctor = json.loads((ROOT / "results/frontier/summary.json").read_text())[
        "recommended_constructor"
    ]
    screening = json.loads((ROOT / "results/frontier/split-tma.json").read_text())
    config = tuple(int(v) for v in screening["confirmation"]["config"])
    path = ROOT / "results/frontier/split-tma-full.json"
    report = {
        "metadata": common.metadata(),
        "baseline_constructor": ctor,
        "configuration": config,
        "scope": "Replace only the first 27 short-shape MLP output projections. Fused reduction/normalization, final encoder layer, and all other shapes retain the baseline path.",
        "source_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__), Path(__file__).with_name("split_tma.py")]
        },
        "parity": [],
    }

    def save():
        path.write_text(json.dumps(report, indent=2) + "\n")

    with (
        FrontierEngine(**ctor, max_graphs=6) as baseline,
        FrontierEngine(**ctor, max_graphs=6) as candidate,
    ):
        install(candidate, config)
        fixtures = common.validation_requests() + requests()
        fixtures += [
            common.workload(batch, length)
            for batch, length in [(1, "short"), (1, "long"), (16, "short")]
        ]
        for index, request in enumerate(fixtures):
            prepared = candidate.prepare(**request)
            actual = candidate.run_prepared(prepared)
            expected = baseline.run_prepared(baseline.prepare(**request))
            report["parity"].append(
                {
                    "index": index,
                    **common.compare_outputs(
                        actual, expected, prepared, candidate.agent
                    ),
                }
            )
        report["all_exact"] = all(
            row["exact_logits_and_actions"] for row in report["parity"]
        )
        save()
        print("parity", len(fixtures), report["all_exact"], flush=True)
        if not report["all_exact"]:
            return
        engines = {"retained": baseline, "tma": candidate}
        report["fixed"] = common.benchmark_interleaved(
            engines,
            cases=[(1, "short"), (1, "long"), (16, "short")],
            rounds=9,
            repeats=100,
        )
        report["changing"] = {"rows": [], "order": []}
        changing = requests()
        rng = random.Random(3707)
        for round_id in range(9):
            order = list(engines)
            rng.shuffle(order)
            indices = list(range(len(changing)))
            rng.shuffle(indices)
            report["changing"]["order"].append({"variants": order, "indices": indices})
            for variant in order:
                samples = []
                for index in indices:
                    start = time.perf_counter()
                    engines[variant].predict(**changing[index])
                    samples.append((time.perf_counter() - start) * 1000)
                report["changing"]["rows"].append(
                    {
                        "variant": variant,
                        "round": round_id,
                        "case": "changing",
                        **common.stats(samples),
                    }
                )
        rows = report["fixed"]["rows"] + report["changing"]["rows"]
        report["summary"] = {}
        for case in ["1-short", "1-long", "16-short", "changing"]:
            subset = [r for r in rows if r["case"] == case]
            medians = {
                name: statistics.median(
                    v for r in subset if r["variant"] == name for v in r["samples_ms"]
                )
                for name in engines
            }
            by_round = {
                name: {r["round"]: r["p50_ms"] for r in subset if r["variant"] == name}
                for name in engines
            }
            savings = [by_round["retained"][i] - by_round["tma"][i] for i in range(9)]
            report["summary"][case] = {
                "median_ms": medians,
                "paired_savings_ms": savings,
                "faster_rounds": sum(v > 0 for v in savings),
            }
        save()
        print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
