"""Validate and compare replay wrappers around one identical retained GPU graph."""

import hashlib
import json
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from experiments.native import common
from laya_blackwell.protocol import format_response
from laya_blackwell.workloads import workload

from .engine import FrontierEngine
from .holdout import requests
from .host_runtime import install


def main():
    torch.set_num_threads(4)
    output = Path("results/frontier/host-runtime.json")
    options = json.loads(Path("results/frontier/summary.json").read_text())[
        "recommended_constructor"
    ]
    report = {
        "metadata": common.metadata(),
        "constructor": options,
        "scope": "Same resident engine and GPU graphs; swap only run_prepared method before each block",
        "sources_sha256": {
            name: hashlib.sha256(
                Path(__file__).with_name(name).read_bytes()
            ).hexdigest()
            for name in ("host_runtime.py", "host_runtime_probe.py")
        },
    }
    with FrontierEngine(**options, max_graphs=16) as engine:
        prior = install(engine)
        hooks = {"retained": prior, "runtime": engine.adapter.run_prepared}
        parity = []
        fixtures = common.validation_requests() + requests()
        for i, fixture in enumerate(fixtures):
            p = engine.prepare(**fixture)
            outputs = {name: hook(p) for name, hook in hooks.items()}
            a, b = outputs.values()
            formatted = [
                format_response(
                    p,
                    value[0],
                    value[1],
                    engine.agent.temperature,
                    engine.agent.temperature_by_options,
                )
                for value in outputs.values()
            ]
            exact = (
                np.array_equal(a[0], b[0])
                and np.array_equal(a[1], b[1])
                and formatted[0] == formatted[1]
            )
            if not exact:
                raise RuntimeError(f"Host replay mismatch at request {i}")
            parity.append({"request": i, "decisions": len(p.items), "exact": exact})
        report["parity"] = {
            "requests": len(parity),
            "decisions": sum(r["decisions"] for r in parity),
            "all_exact": True,
            "rows": parity,
        }
        # Check ownership after staging is overwritten, plus concurrent use.
        prepared = [engine.prepare(**fixture) for fixture in requests()[:4]]
        expected = [prior(p)[:2] for p in prepared]
        saved = [tuple(x.copy() for x in value) for value in expected]
        with ThreadPoolExecutor(max_workers=4) as pool:
            actual = list(pool.map(hooks["runtime"], prepared * 4))
        report["concurrent_exact"] = all(
            np.array_equal(x[0], expected[i % 4][0])
            and np.array_equal(x[1], expected[i % 4][1])
            for i, x in enumerate(actual)
        )
        report["output_ownership_exact"] = all(
            np.array_equal(x[0], y[0]) and np.array_equal(x[1], y[1])
            for x, y in zip(expected, saved)
        )
        if not report["concurrent_exact"] or not report["output_ownership_exact"]:
            raise RuntimeError("Replay ownership/concurrency failed")
        # Force genuine cache misses through the candidate, preserving eviction.
        engine.adapter.graphs.clear()
        engine.adapter.max_graphs = 1
        checked_keys = []
        for fixture in (
            workload(1, "short"),
            workload(1, "long"),
            workload(1, "short"),
        ):
            p = engine.prepare(**fixture)
            result = hooks["runtime"](p)
            key = engine.base._graph_key(p)
            checked_keys.append(
                result[2]["graph_miss"]
                and len(engine.adapter.graphs) == 1
                and key in engine.adapter.graphs
            )
        if not all(checked_keys):
            raise RuntimeError("Graph miss/eviction behavior changed")
        report["cache_miss_eviction_checks"] = checked_keys
        engine.adapter.max_graphs = 16
        rows, order = [], []
        rng = random.Random(725099)
        for case in ("1-short", "changing-inputs", "1-long", "16-short"):
            fixtures = (
                requests()
                if case == "changing-inputs"
                else [workload(int(case.split("-")[0]), case.split("-")[1])]
            )
            for hook in hooks.values():
                engine.adapter.run_prepared = hook
                for fixture in fixtures:
                    engine.predict(**fixture)
            for round_id in range(9):
                names = list(hooks)
                rng.shuffle(names)
                indices = list(range(128 if len(fixtures) > 1 else 100))
                rng.shuffle(indices)
                order.append(
                    {
                        "case": case,
                        "round": round_id,
                        "variants": names,
                        "indices": indices,
                    }
                )
                for name in names:
                    engine.adapter.run_prepared = hooks[name]
                    samples = []
                    for index in indices:
                        start = time.perf_counter_ns()
                        engine.predict(**fixtures[index % len(fixtures)])
                        samples.append((time.perf_counter_ns() - start) / 1e6)
                    rows.append(
                        {
                            "case": case,
                            "variant": name,
                            "round": round_id,
                            **common.stats(samples),
                        }
                    )
            print("Completed", case, flush=True)
        report["timings"] = {"rows": rows, "order": order}
        report["summary"] = {}
        for case in ("1-short", "changing-inputs", "1-long", "16-short"):
            selected = [r for r in rows if r["case"] == case]
            by = {
                name: {
                    r["round"]: r["p50_ms"] for r in selected if r["variant"] == name
                }
                for name in hooks
            }
            report["summary"][case] = {
                "p50_ms": {
                    name: statistics.median(
                        [
                            v
                            for r in selected
                            if r["variant"] == name
                            for v in r["samples_ms"]
                        ]
                    )
                    for name in hooks
                },
                "paired_savings_ms": [
                    by["retained"][i] - by["runtime"][i] for i in range(9)
                ],
                "faster_rounds": sum(
                    by["runtime"][i] < by["retained"][i] for i in range(9)
                ),
            }
        engine.adapter.run_prepared = hooks["runtime"]
    try:
        hooks["runtime"](prepared[0])
    except RuntimeError as error:
        report["closed_guard"] = str(error) == "Engine is closed"
    else:
        raise RuntimeError("Closed adapter accepted inference")
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "parity": {k: v for k, v in report["parity"].items() if k != "rows"},
                "summary": report["summary"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
