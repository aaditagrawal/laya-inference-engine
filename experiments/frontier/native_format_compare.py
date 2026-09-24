"""Same-engine full-request comparison of native response formatting.

Requires the exclusive experiment lock. The baseline constructor is frozen in
the report; only this engine's bound adapter.predict method changes between blocks.
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
from .native_format import NativeFormatter, install


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("results/frontier/native-format-full.json")
    )
    args = parser.parse_args()
    probe = json.loads(Path("results/frontier/native-format.json").read_text())
    if (
        not probe["all_exact"]
        or probe["summary_ms"]["native"] >= probe["summary_ms"]["reference"]
    ):
        raise RuntimeError("A passing, faster CPU formatter probe is required")
    torch.set_num_threads(4)
    constructor = json.loads(Path("results/frontier/summary.json").read_text())[
        "recommended_constructor"
    ]
    # The retained constructor may already include the candidate formatter.
    constructor.pop("native_format", None)
    report = {
        "metadata": common.metadata(),
        "constructor": constructor,
        "method": "Same engine/graphs/weights; bound predict method swapped outside timing",
    }
    formatter = NativeFormatter()
    with FrontierEngine(**constructor, max_graphs=16) as engine:
        original = install(engine)
        candidate = engine.adapter.predict
        functions = {"reference": original, "native": candidate}
        parity = []
        for index, fixture in enumerate(common.validation_requests() + requests()):
            prepared = engine.prepare(**fixture)
            logits, actions, _ = engine.run_prepared(prepared)
            inputs = (
                prepared,
                logits,
                actions,
                engine.agent.temperature,
                engine.agent.temperature_by_options,
            )
            before = (logits.copy(), actions.copy())
            expected = format_response(*inputs)
            actual = formatter(*inputs)
            native_used = formatter.native(*inputs) is not None
            response_exact = json.dumps(expected) == json.dumps(actual)
            immutable = np.array_equal(logits, before[0]) and np.array_equal(
                actions, before[1]
            )
            public = {}
            for name, function in functions.items():
                engine.adapter.predict = function
                response = engine.predict(**fixture)
                metrics = response.pop("engine")
                public[name] = response
                if set(metrics) != {
                    "graph_miss",
                    "graph_build_ms",
                    "shape",
                    "backend",
                    "total_ms",
                }:
                    raise RuntimeError("Unexpected or changed engine metrics")
            full_exact = json.dumps(public["reference"]) == json.dumps(public["native"])
            parity.append(
                {
                    "request": index,
                    "decisions": len(prepared.ids),
                    "response_exact": response_exact,
                    "raw_unchanged": immutable,
                    "full_response_exact": full_exact,
                    "native_used": native_used,
                }
            )
            if not response_exact or not immutable or not full_exact:
                raise RuntimeError(f"Formatter parity failed on fixture {index}")
            if index % 32 == 0:
                print(f"Validated request {index}", flush=True)
        report["parity"] = {
            "all_exact": True,
            "requests": len(parity),
            "decisions": sum(row["decisions"] for row in parity),
            "native_requests": sum(row["native_used"] for row in parity),
            "rows": parity,
        }
        rng = random.Random(328935)
        timing, orders = [], []
        for case in ("1-short", "1-long", "16-short", "changing-inputs"):
            fixtures = (
                requests()
                if case == "changing-inputs"
                else [workload(int(case.split("-")[0]), case.split("-")[1])]
            )
            for function in functions.values():
                engine.adapter.predict = function
                for fixture in fixtures:
                    engine.predict(**fixture)
            for round_id in range(9):
                names = list(functions)
                rng.shuffle(names)
                indices = list(range(128 if len(fixtures) > 1 else 100))
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
                    engine.adapter.predict = functions[name]
                    samples = []
                    for index in indices:
                        fixture = fixtures[index % len(fixtures)]
                        start = time.perf_counter_ns()
                        engine.predict(**fixture)
                        samples.append((time.perf_counter_ns() - start) / 1e6)
                    timing.append(
                        {
                            "case": case,
                            "round": round_id,
                            "variant": name,
                            **common.stats(samples),
                        }
                    )
            print(f"Benchmarked {case}", flush=True)
        report["timing"] = {"rows": timing, "orders": orders}
        summary = {}
        for case in ("1-short", "1-long", "16-short", "changing-inputs"):
            selected = [row for row in timing if row["case"] == case]
            pooled = {
                name: statistics.median(
                    [
                        sample
                        for row in selected
                        if row["variant"] == name
                        for sample in row["samples_ms"]
                    ]
                )
                for name in functions
            }
            medians = {
                name: {
                    row["round"]: row["p50_ms"]
                    for row in selected
                    if row["variant"] == name
                }
                for name in functions
            }
            savings = [medians["reference"][i] - medians["native"][i] for i in range(9)]
            summary[case] = {
                "p50_ms": pooled,
                "paired_round_savings_ms": savings,
                "faster_rounds": sum(value > 0 for value in savings),
            }
        report["summary"] = summary
        engine.adapter.predict = original
    report["build"] = formatter.build
    report["source_sha256"] = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(Path("experiments/frontier").glob("native_format*"))
        if path.is_file()
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
                "summary": report["summary"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
