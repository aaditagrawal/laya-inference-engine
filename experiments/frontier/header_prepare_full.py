"""Same-resident-engine full request comparison of direct header preparation.

Run under exclusive /tmp/laya-gpu-experiments.lock. Both variants share weights,
graphs, I/O, and installed native formatting. Switch bound preparation outside
each timed block. Never mutate global formatting or engine defaults.
"""

import hashlib
import json
import random
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch

from experiments.native import common

from .engine import FrontierEngine
from .header_prepare import install
from .holdout import requests

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sources():
    result = {str(Path(__file__).resolve()): digest(__file__)}
    for module in list(sys.modules.values()):
        path = getattr(module, "__file__", None)
        if path is None:
            continue
        path = Path(path).resolve()
        if path.suffix == ".py" and any(
            path.is_relative_to(ROOT / prefix) for prefix in ("src", "experiments")
        ):
            result[str(path)] = digest(path)
    return result


def libraries():
    paths = set()
    for line in Path("/proc/self/maps").read_text().splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) == 6:
            path = Path(parts[-1])
            if path.is_relative_to(ROOT / ".research") and ".so" in path.name:
                paths.add(path)
    return {str(path): digest(path) for path in sorted(paths)}


def main():
    extra = json.loads(Path("results/frontier/header-prepare-extra.json").read_text())
    if not extra["all_exact"] or not extra["all_fallbacks_as_expected"]:
        raise RuntimeError("Targeted mutation/fallback gate did not pass")
    candidate_path = ROOT / "experiments/frontier/header_prepare.py"
    if digest(candidate_path) != extra["candidate_hardened_sha256"]:
        raise RuntimeError("Candidate changed after the targeted gate")
    probe = json.loads(Path("results/frontier/header-prepare.json").read_text())
    if not probe["all_exact"]:
        raise RuntimeError("Established CPU correctness gate did not pass")
    fixed_cpu = probe["summary_ms"]["1-short"]
    if fixed_cpu["batch"] >= fixed_cpu["single"]:
        raise RuntimeError("Candidate failed the fixed-workload CPU latency gate")
    torch.set_num_threads(4)
    constructor = json.loads(Path("results/frontier/summary.json").read_text())[
        "recommended_constructor"
    ]
    if constructor.get("host_prepare") != "batch":
        raise RuntimeError("Expected retained batched host preparation")
    before = sources()
    report = {
        "metadata": common.metadata(),
        "constructor": constructor,
        "method": "Same resident engine; bound adapter.prepare swapped outside paired timing blocks",
        "candidate_initial_sha256": extra["candidate_initial_sha256"],
        "candidate_hardened_sha256": extra["candidate_hardened_sha256"],
        "targeted_probe_sha256": digest("results/frontier/header-prepare-extra.json"),
        "cpu_probe_sha256": digest("results/frontier/header-prepare.json"),
        "source_before": before,
    }
    fixtures = common.validation_requests() + requests()
    extras = [
        common.workload(n, length)
        for n, length in ((1, "short"), (1, "long"), (16, "short"))
    ]
    with FrontierEngine(**constructor, max_graphs=16) as engine:
        original = install(engine)
        hooks = {"retained": original, "direct-header": engine.adapter.prepare}
        report["installed_policy"] = engine.policy
        # Include lazily imported native and compiler adapters after setup.
        before.update(sources())
        parity = []
        for index, fixture in enumerate(fixtures + extras):
            outputs = {}
            for name, hook in hooks.items():
                engine.adapter.prepare = hook
                prepared = engine.prepare(**fixture)
                logits, actions, _ = engine.run_prepared(prepared)
                public = engine.predict(**fixture)
                metrics = public.pop("engine")
                if set(metrics) != {
                    "graph_miss",
                    "graph_build_ms",
                    "shape",
                    "backend",
                    "total_ms",
                }:
                    raise RuntimeError("Unexpected engine metrics")
                outputs[name] = (prepared, logits, actions, public, metrics)
            baseline, actual = outputs.values()
            checks = {
                "prepared_exact": baseline[0] == actual[0],
                "choice_bits_exact": np.array_equal(
                    baseline[1].view(np.uint8), actual[1].view(np.uint8)
                ),
                "action_bits_exact": np.array_equal(
                    baseline[2].view(np.uint8), actual[2].view(np.uint8)
                ),
                "public_exact": json.dumps(baseline[3]) == json.dumps(actual[3]),
                "metadata_exact": all(
                    baseline[4][k] == actual[4][k] for k in ("shape", "backend")
                ),
            }
            parity.append(
                {"request": index, "decisions": len(baseline[0].ids), **checks}
            )
            if not all(checks.values()):
                raise RuntimeError(f"Parity failed at {index}: {checks}")
            if index % 32 == 0:
                print(f"Validated request {index}", flush=True)
        report["parity"] = {
            "all_exact": True,
            "requests": len(parity),
            "decisions": sum(r["decisions"] for r in parity),
            "original_and_holdout_requests": len(fixtures),
            "benchmark_extras": len(extras),
            "rows": parity,
        }
        before.update(sources())
        report["native_libraries_before"] = libraries()
        rows, orders = [], []
        rng = random.Random(982437)
        for case in ("1-short", "1-long", "16-short", "changing-inputs"):
            cases = (
                requests()
                if case == "changing-inputs"
                else [common.workload(int(case.split("-")[0]), case.split("-")[1])]
            )
            for hook in hooks.values():
                engine.adapter.prepare = hook
                for fixture in cases:
                    engine.predict(**fixture)
            for round_id in range(11):
                names = list(hooks)
                indices = list(range(128 if case == "changing-inputs" else 150))
                rng.shuffle(names)
                rng.shuffle(indices)
                orders.append(
                    {
                        "case": case,
                        "round": round_id,
                        "variants": names,
                        "indices": indices,
                    }
                )
                for name in names:
                    engine.adapter.prepare = hooks[name]
                    samples = []
                    for i in indices:
                        fixture = cases[i % len(cases)]
                        start = time.perf_counter_ns()
                        engine.predict(**fixture)
                        samples.append((time.perf_counter_ns() - start) / 1e6)
                    rows.append(
                        {
                            "case": case,
                            "round": round_id,
                            "variant": name,
                            **common.stats(samples),
                        }
                    )
            print(f"Benchmarked {case}", flush=True)
        summary = {}
        for case in ("1-short", "1-long", "16-short", "changing-inputs"):
            subset = [row for row in rows if row["case"] == case]
            medians = {
                name: {r["round"]: r["p50_ms"] for r in subset if r["variant"] == name}
                for name in hooks
            }
            savings = [
                medians["retained"][i] - medians["direct-header"][i] for i in range(11)
            ]
            summary[case] = {
                "p50_ms": {
                    name: statistics.median(
                        [
                            s
                            for r in subset
                            if r["variant"] == name
                            for s in r["samples_ms"]
                        ]
                    )
                    for name in hooks
                },
                "paired_round_savings_ms": savings,
                "median_paired_savings_ms": statistics.median(savings),
                "faster_rounds": sum(value > 0 for value in savings),
            }
        report["timing"] = {"rows": rows, "orders": orders}
        report["summary"] = summary
        report["native_libraries_after"] = libraries()
        report["native_libraries_unchanged"] = (
            report["native_libraries_before"] == report["native_libraries_after"]
        )
        engine.adapter.prepare = original
    after = {path: digest(path) for path in before}
    report["source_after"] = after
    report["sources_unchanged"] = before == after
    Path("results/frontier/header-prepare-full.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "parity": {k: v for k, v in report["parity"].items() if k != "rows"},
                "summary": summary,
                "sources_unchanged": before == after,
                "native_libraries_unchanged": report["native_libraries_unchanged"],
            },
            indent=2,
        ),
        flush=True,
    )
    if not report["sources_unchanged"] or not report["native_libraries_unchanged"]:
        raise RuntimeError("Sources or native libraries changed during comparison")


if __name__ == "__main__":
    main()
