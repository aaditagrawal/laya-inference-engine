"""Compare host preparation variants around the same retained GPU graph.

Run under the exclusive /tmp/laya-gpu-experiments.lock. Switching preparation
outside the timed call keeps graph, weights, allocation and formatting identical.
"""

import argparse
import hashlib
import json
import random
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from experiments.native import common
from laya_blackwell.protocol import format_response
from laya_blackwell.workloads import workload

from .engine import FrontierEngine
from .holdout import requests
from .host_prepare import install


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("results/frontier/host-full.json")
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    report = {
        "metadata": common.metadata(),
        "configuration": {
            "policy": "bf16-splitk-exact-short-compiled",
            "attention": "native",
            "fuse_reduce_norm": True,
            "token_tables": True,
            "fuse_mlp_geglu": True,
            "mlp_geglu_unpacked": True,
            "method": "One engine; switch bound prepare method before each measurement block",
        },
    }
    with FrontierEngine(
        policy="bf16-splitk-exact-short-compiled",
        attention="native",
        fuse_reduce_norm=True,
        token_tables=True,
        fuse_mlp_geglu=True,
        mlp_geglu_unpacked=True,
        max_graphs=16,
    ) as engine:
        hooks = {"baseline": engine.adapter.prepare}
        for mode in ("single", "batch", "template"):
            install(engine, mode)
            hooks[mode] = engine.adapter.prepare
        parity = []
        for index, fixture in enumerate(common.validation_requests() + requests()):
            outputs = {}
            for mode, hook in hooks.items():
                engine.adapter.prepare = hook
                prepared = engine.prepare(**fixture)
                logits, actions, _ = engine.run_prepared(prepared)
                formatted = format_response(
                    prepared,
                    logits,
                    actions,
                    engine.agent.temperature,
                    engine.agent.temperature_by_options,
                )
                outputs[mode] = (prepared, logits, actions, formatted)
            original = outputs["baseline"]
            exact = {
                mode: (
                    value[0] == original[0]
                    and np.array_equal(value[1], original[1])
                    and np.array_equal(value[2], original[2])
                    and value[3] == original[3]
                )
                for mode, value in outputs.items()
            }
            parity.append(
                {"request": index, "decisions": len(original[0].items), "exact": exact}
            )
            if not all(exact.values()):
                raise RuntimeError(f"Output parity failed on request {index}: {exact}")
            if index % 32 == 0:
                print(f"Verified request {index}", flush=True)
        report["parity"] = {
            "requests": len(parity),
            "decisions": sum(row["decisions"] for row in parity),
            "all_exact": True,
            "rows": parity,
        }
        rng = random.Random(945623)
        rows = []
        order = []
        for case in ("1-short", "1-long", "16-short", "changing-inputs"):
            fixtures = (
                requests()
                if case == "changing-inputs"
                else [workload(int(case.split("-")[0]), case.split("-")[1])]
            )
            for mode, hook in hooks.items():
                engine.adapter.prepare = hook
                for fixture in fixtures:
                    engine.predict(**fixture)
            for round_id in range(9):
                modes = list(hooks)
                rng.shuffle(modes)
                indices = list(range(128 if len(fixtures) > 1 else 100))
                rng.shuffle(indices)
                order.append(
                    {
                        "case": case,
                        "round": round_id,
                        "modes": modes,
                        "indices": indices,
                    }
                )
                for mode in modes:
                    engine.adapter.prepare = hooks[mode]
                    samples = []
                    for index in indices:
                        fixture = fixtures[index % len(fixtures)]
                        start = time.perf_counter_ns()
                        engine.predict(**fixture)
                        samples.append((time.perf_counter_ns() - start) / 1e6)
                    rows.append(
                        {
                            "case": case,
                            "mode": mode,
                            "round": round_id,
                            **common.stats(samples),
                        }
                    )
            print(f"Benchmarked {case}", flush=True)
        report["timing"] = {"rows": rows, "order": order}
        summary = {}
        for case in ("1-short", "1-long", "16-short", "changing-inputs"):
            selected = [row for row in rows if row["case"] == case]
            pooled = {
                mode: statistics.median(
                    [
                        sample
                        for row in selected
                        if row["mode"] == mode
                        for sample in row["samples_ms"]
                    ]
                )
                for mode in hooks
            }
            medians = {
                mode: {
                    row["round"]: row["p50_ms"]
                    for row in selected
                    if row["mode"] == mode
                }
                for mode in hooks
            }
            savings = {
                mode: [medians["baseline"][i] - medians[mode][i] for i in range(9)]
                for mode in hooks
                if mode != "baseline"
            }
            summary[case] = {
                "p50_ms": pooled,
                "paired_round_savings_ms": savings,
                "faster_rounds": {
                    mode: sum(value > 0 for value in values)
                    for mode, values in savings.items()
                },
            }
        report["summary"] = summary
        engine.adapter.prepare = hooks["baseline"]
    report["source_sha256"] = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(Path("experiments/frontier").glob("host_*.py"))
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "parity": {
                    key: value
                    for key, value in report["parity"].items()
                    if key != "rows"
                },
                "summary": summary,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
