"""Full retained-engine weight-placement parity and paired request timing."""

import argparse
import fcntl
import gc
import hashlib
import json
import random
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

from experiments.native import common
from laya_blackwell.protocol import format_response

from .engine import FrontierEngine
from .holdout import requests
from .weight_layout import WeightLayout, execution_order, granularity, original_model

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results/frontier/weight_layout-screen.json"


def hashes():
    paths = list(Path(__file__).parent.glob("weight_layout*.py"))
    paths += [
        ROOT / path
        for path in (
            "experiments/frontier/compression.py",
            "experiments/frontier/engine.py",
            "experiments/frontier/mlp_geglu.py",
            "experiments/frontier/host_runtime.py",
            "experiments/frontier/native_format.py",
            "experiments/frontier/holdout.py",
            "experiments/native/common.py",
            "src/laya_blackwell/model.py",
            "experiments/latency/serving/graph_adapter.py",
            "results/frontier/summary.json",
        )
    ]
    return {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
    }


def strip_metrics(response):
    return {name: value for name, value in response.items() if name != "engine"}


def same_bits(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def validate(baseline, candidate):
    fixtures = (
        common.validation_requests()
        + requests()
        + [
            common.workload(batch, length)
            for batch, length in ((1, "short"), (1, "long"), (16, "short"))
        ]
    )
    assert len(fixtures) == 197
    rows = []
    for index, fixture in enumerate(fixtures):
        expected_prepared = baseline.prepare(**fixture)
        actual_prepared = candidate.prepare(**fixture)
        expected = baseline.run_prepared(expected_prepared)
        actual = candidate.run_prepared(actual_prepared)
        expected_formatted = format_response(
            expected_prepared,
            expected[0],
            expected[1],
            baseline.agent.temperature,
            baseline.agent.temperature_by_options,
        )
        actual_formatted = format_response(
            actual_prepared,
            actual[0],
            actual[1],
            candidate.agent.temperature,
            candidate.agent.temperature_by_options,
        )
        expected_public = strip_metrics(baseline.predict(**fixture))
        actual_public = strip_metrics(candidate.predict(**fixture))
        row = {
            "request": index,
            "decisions": len(actual_prepared.ids),
            "graph_key": candidate.base._graph_key(actual_prepared),
            "prepared_exact": expected_prepared == actual_prepared,
            "logits_bit_exact": same_bits(actual[0], expected[0]),
            "actions_bit_exact": same_bits(actual[1], expected[1]),
            "reference_formatter_exact": expected_formatted == actual_formatted,
            "public_response_exact": expected_public == actual_public,
        }
        row["exact"] = all(row[name] for name in row if name.endswith("exact"))
        rows.append(row)
        if not row["exact"]:
            raise RuntimeError(f"Weight placement changed request {index}: {row}")
        if index % 32 == 0:
            print("Parity", index, flush=True)
    return {
        "requests": len(rows),
        "decisions": sum(row["decisions"] for row in rows),
        "all_exact": True,
        "rows": rows,
    }


def summarize(rows):
    result = {}
    for case in dict.fromkeys(row["case"] for row in rows):
        selected = [row for row in rows if row["case"] == case]
        by = {
            name: {
                row["round"]: row["p50_ms"]
                for row in selected
                if row["variant"] == name
            }
            for name in ("original", "candidate")
        }
        savings = [by["original"][i] - by["candidate"][i] for i in range(9)]
        result[case] = {
            "pooled_p50_ms": {
                name: statistics.median(
                    value
                    for row in selected
                    if row["variant"] == name
                    for value in row["samples_ms"]
                )
                for name in by
            },
            "paired_savings_ms": savings,
            "median_paired_savings_ms": statistics.median(savings),
            "faster_rounds": sum(value > 0 for value in savings),
        }
    return result


def measure(baseline, candidate):
    engines = {"original": baseline, "candidate": candidate}
    fixed = common.benchmark_interleaved(
        engines, cases=[(1, "short")], rounds=9, repeats=150, warmups=5
    )
    fixtures = requests()
    for engine in engines.values():
        for fixture in fixtures:
            engine.predict(**fixture)
    rng = random.Random(984652)
    changing, order = [], []
    for round_id in range(9):
        indices, names = list(range(len(fixtures))), list(engines)
        rng.shuffle(indices)
        rng.shuffle(names)
        order.append({"round": round_id, "indices": indices, "variants": names})
        for name in names:
            samples = []
            for index in indices:
                started = time.perf_counter_ns()
                engines[name].predict(**fixtures[index])
                samples.append((time.perf_counter_ns() - started) / 1e6)
            changing.append(
                {
                    "variant": name,
                    "case": "changing-holdout",
                    "round": round_id,
                    **common.stats(samples),
                }
            )
    combined = fixed["rows"] + changing
    summary = summarize(combined)
    # Secondary shapes are only timed when the short results look useful.
    useful = any(
        value["median_paired_savings_ms"] > 0.005 and value["faster_rounds"] >= 7
        for value in summary.values()
    )
    secondary = None
    if useful:
        secondary = common.benchmark_interleaved(
            engines, cases=[(1, "long"), (16, "short")], rounds=9, repeats=50, warmups=5
        )
        combined += secondary["rows"]
        summary = summarize(combined)
    return {
        "fixed": fixed,
        "changing": {"rows": changing, "order": order},
        "secondary": secondary,
        "secondary_triggered": useful,
        "summary": summary,
    }


def evaluate(baseline, candidate, row, save):
    row["parity"] = validate(baseline, candidate)
    print("All 197 requests exact; beginning paired timing", row["kind"], flush=True)
    save()
    # Both models and their graphs stay resident throughout each paired series.
    row["timings"] = measure(baseline, candidate)
    print(
        json.dumps({"kind": row["kind"], "summary": row["timings"]["summary"]}),
        flush=True,
    )
    save()


def run(args):
    torch.set_num_threads(4)
    constructor = json.loads((ROOT / "results/frontier/summary.json").read_text())[
        "recommended_constructor"
    ]
    report = {
        "metadata": common.metadata(),
        "constructor": constructor,
        "granularity": granularity(),
        "source_before": hashes(),
        "rows": [],
        "scope": "Original retained engine remains resident; candidates replace only frozen BF16 parameter storage before any graph capture. No HTTP, math changes, answer caching, or allocator/context setting changes.",
    }

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    baseline = FrontierEngine(**constructor, max_graphs=16)
    try:
        report["original_parameters"] = [
            {
                "name": name,
                "pointer": parameter.data_ptr(),
                "bytes": parameter.nbytes,
                "shape": list(parameter.shape),
            }
            for name, parameter in execution_order(original_model(baseline))
        ]
        save()
        for kind in args.layouts:
            row = {"kind": kind}
            report["rows"].append(row)
            layout = WeightLayout()
            candidate = FrontierEngine(**constructor, max_graphs=16)
            try:
                row["installation"] = layout.install(candidate, kind)
                print(
                    "Installed",
                    kind,
                    "bytes",
                    row["installation"]["allocation"],
                    flush=True,
                )
                save()
                evaluate(baseline, candidate, row, save)
            finally:
                candidate.close()
                del candidate
                if args.process_owned:
                    layout.retain_until_process_exit()
                else:
                    layout.close()
                row["cleanup"] = layout.report["cleanup"]
                del layout
                gc.collect()
                torch.cuda.empty_cache()
                save()
            current = [
                parameter.data_ptr()
                for _, parameter in execution_order(original_model(baseline))
            ]
            assert current == [row["pointer"] for row in report["original_parameters"]]
            # Dynamo cleanup does not destroy the baseline's owned replay graphs.
            baseline.predict(**common.workload(1, "short"))
        report["source_after"] = hashes()
        assert report["source_before"] == report["source_after"]
    finally:
        baseline.close()
        del baseline
        torch._dynamo.reset()
        gc.collect()
    save()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--layouts",
        nargs="+",
        choices=("torch-bank", "vmm-bank", "vmm-2m-aligned"),
        default=("torch-bank", "vmm-bank", "vmm-2m-aligned"),
    )
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--process-owned", action="store_true")
    args = parser.parse_args()
    if args.process_owned and len(args.layouts) != 1:
        parser.error("Process-owned mode requires exactly one candidate per process")
    if not args.process_owned:
        # The dispatcher holds no lock while waiting for workers. Each worker
        # owns exactly one candidate bank and acquires the experiment lock.
        for kind in args.layouts:
            destination = args.output.with_name(args.output.stem + "-" + kind + ".json")
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "experiments.frontier.weight_layout_probe",
                    "--layouts",
                    kind,
                    "--process-owned",
                    "--output",
                    str(destination),
                ],
                check=True,
            )
        return
    with Path("/tmp/laya-gpu-experiments.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        run(args)


if __name__ == "__main__":
    main()
