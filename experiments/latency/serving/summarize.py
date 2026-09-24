"""Aggregate completed serving benchmark JSON; failed runs are excluded."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    groups = defaultdict(list)
    for path in args.inputs:
        report = json.loads(path.read_text())
        if "cuda_bytes_after_engine_close" not in report or report.get("failure"):
            raise ValueError(f"Refusing incomplete/failed benchmark: {path}")
        for row in report["rows"]:
            groups[row["variant"], row["profile"], row["concurrency"]].append(row)
    rows = []
    for (variant, profile, concurrency), group in sorted(groups.items()):
        samples = [s for r in group for s in r["samples"]]
        latency = [s["latency_ms"] for s in samples]
        elapsed = sum(r["elapsed_s"] for r in group)
        row = {
            "variant": variant,
            "profile": profile,
            "concurrency": concurrency,
            "rounds": len(group),
            "requests": len(samples),
            "elapsed_s": elapsed,
            "requests_per_second": len(samples) / elapsed,
            "decisions_per_second": sum(s["questions"] for s in samples) / elapsed,
            "p50_ms": float(np.median(latency)),
            "p95_ms": float(np.percentile(latency, 95)),
            "graph_misses": sum(r["graph_misses"] for r in group),
            "exact_answers_all": all(
                r["response_check"]["exact_answers_all"] for r in group
            ),
            "decision_disagreements": sum(
                r["response_check"]["decisions"] - r["response_check"]["agreement"]
                for r in group
            ),
            "max_probability_error": max(
                r["response_check"]["max_probability_error"] for r in group
            ),
            "max_action_probability_error": max(
                r["response_check"]["max_action_probability_error"] for r in group
            ),
        }
        if profile == "mixed":
            row["by_request_class"] = {}
            for index, name in enumerate(("1-short", "16-short", "1-long", "16-long")):
                values = [s["latency_ms"] for s in samples if s["request"] % 4 == index]
                row["by_request_class"][name] = {
                    "requests": len(values),
                    "p50_ms": float(np.median(values)),
                    "p95_ms": float(np.percentile(values, 95)),
                }
        rows.append(row)
    baseline = {
        (r["profile"], r["concurrency"]): r
        for r in rows
        if r["variant"] == "serial-current"
    }
    for row in rows:
        ref = baseline.get((row["profile"], row["concurrency"]))
        if ref:
            row["throughput_ratio_vs_serial"] = (
                row["requests_per_second"] / ref["requests_per_second"]
            )
            row["p95_ratio_vs_serial"] = row["p95_ms"] / ref["p95_ms"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps({"inputs": [str(p) for p in args.inputs], "rows": rows}, indent=2)
        + "\n"
    )


if __name__ == "__main__":
    main()
