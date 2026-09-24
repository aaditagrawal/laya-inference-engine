"""Full parity and paired latency for the 32-row manual-indexing control.

Use the exclusive experiment lock. This control keeps the original 32 CTAs;
the higher-CTA 16/8-row variants were slower and are excluded here.
"""

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

from .attention_special_adapter import load as load_special
from .attention_special_probe import capture as capture_local
from .engine import FrontierEngine
from .global_attention_padding import capture as capture_global
from .holdout import requests
from .query_shard_adapter import install
from .query_shard_probe import additive, bit_mismatches, special


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    report = {
        "metadata": common.metadata(),
        "mechanism": "Bypass advance_to_block with constant/manual indexing; unchanged 32-row arithmetic and 32 CTAs",
    }
    constructor = json.loads(Path("results/frontier/summary.json").read_text())[
        "recommended_constructor"
    ]
    report["constructor"] = constructor
    fixtures = common.validation_requests() + requests()
    extras = [
        common.workload(count, length)
        for count, length in ((1, "short"), (1, "long"), (16, "short"))
    ]
    from .query_shard import attention, load

    load()
    load_special()
    # Check raw attention outputs for every eligible standard or benchmark input.
    raw_checks = []
    with FrontierEngine(policy="bf16-exact", max_graphs=2) as engine:
        for index, fixture in enumerate(fixtures + extras):
            prepared = engine.prepare(**fixture)
            key = engine.base._graph_key(prepared)
            if key[:2] != (1, 64):
                raw_checks.append(
                    {"request": index, "eligible": False, "graph_key": list(key)}
                )
                continue
            local = additive(
                capture_local(engine, fixture, allow_padding=True), unmasked=key[-1]
            )
            global_raw = capture_global(engine, fixture)
            global_group = [] if global_raw is None else additive(global_raw)
            mismatch = []
            for values in local + global_group:
                q, k, v, bias = values
                expected = special(q, k, v, 32, bias)
                actual = attention(q, k, v, 32, bias)
                mismatch.append(bit_mismatches(actual, expected))
            raw_checks.append(
                {
                    "request": index,
                    "eligible": True,
                    "local_calls": len(local),
                    "global_calls": len(global_group),
                    "mismatches": sum(mismatch),
                }
            )
            if sum(mismatch):
                raise RuntimeError(f"Raw attention parity failed at {index}")
    report["attention_gate"] = raw_checks
    print("All eligible raw attention inputs passed", flush=True)

    with (
        FrontierEngine(**constructor, max_graphs=16) as baseline,
        FrontierEngine(**constructor, max_graphs=16) as candidate,
    ):
        report["installation"] = install(candidate)
        engines = {"retained": baseline, "manual-32": candidate}
        parity = []
        for index, fixture in enumerate(fixtures + extras):
            outputs = {}
            for name, engine in engines.items():
                prepared = engine.prepare(**fixture)
                logits, actions, _ = engine.run_prepared(prepared)
                formatted = format_response(
                    prepared,
                    logits,
                    actions,
                    engine.agent.temperature,
                    engine.agent.temperature_by_options,
                )
                outputs[name] = (prepared, logits, actions, formatted)
            reference, actual = outputs.values()
            exact = (
                reference[0] == actual[0]
                and np.array_equal(reference[1], actual[1])
                and np.array_equal(reference[2], actual[2])
                and json.dumps(reference[3]) == json.dumps(actual[3])
            )
            parity.append(
                {"request": index, "decisions": len(reference[0].ids), "exact": exact}
            )
            if not exact:
                raise RuntimeError(f"Full model parity failed at {index}")
            if index % 32 == 0:
                print(f"Validated full request {index}", flush=True)
        report["parity"] = {
            "all_exact": True,
            "requests": len(parity),
            "decisions": sum(row["decisions"] for row in parity),
            "rows": parity,
        }
        report["fixed"] = common.benchmark_interleaved(
            engines,
            cases=[(1, "short"), (1, "long"), (16, "short")],
            rounds=11,
            repeats=150,
        )
        changing = requests()
        rows, order = [], []
        rng = random.Random(384253)
        for round_id in range(11):
            names = list(engines)
            indices = list(range(len(changing)))
            rng.shuffle(names)
            rng.shuffle(indices)
            order.append({"round": round_id, "variants": names, "indices": indices})
            for name in names:
                samples = []
                for index in indices:
                    start = time.perf_counter_ns()
                    engines[name].predict(**changing[index])
                    samples.append((time.perf_counter_ns() - start) / 1e6)
                rows.append(
                    {"variant": name, "round": round_id, **common.stats(samples)}
                )
        report["changing"] = {"rows": rows, "order": order}
        report["summary"] = {}
        for case in ("1-short", "1-long", "16-short", "changing"):
            selected = (
                rows
                if case == "changing"
                else [row for row in report["fixed"]["rows"] if row["case"] == case]
            )
            pooled = {
                name: statistics.median(
                    [
                        sample
                        for row in selected
                        if row["variant"] == name
                        for sample in row["samples_ms"]
                    ]
                )
                for name in engines
            }
            medians = {
                name: {
                    row["round"]: row["p50_ms"]
                    for row in selected
                    if row["variant"] == name
                }
                for name in engines
            }
            differences = [
                medians["retained"][i] - medians["manual-32"][i] for i in range(11)
            ]
            report["summary"][case] = {
                "p50_ms": pooled,
                "round_savings_ms": differences,
                "faster_rounds": sum(value > 0 for value in differences),
            }
    report["build"] = json.loads(
        Path(".research/frontier-query-shard/build.json").read_text()
    )
    report["source_sha256"] = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(Path("experiments/frontier").glob("query_shard*"))
        if path.is_file()
    }
    Path("results/frontier/query-shard-full.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
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
