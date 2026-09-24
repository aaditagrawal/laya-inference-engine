"""Summarize saved raw samples without rerunning GPU experiments."""

import json
from pathlib import Path

import numpy as np


def main():
    root = Path("results/latency-optimizations/fusion")
    summary = {
        "method": "Median of all full-predict samples per engine/case, three randomized blocks. Final best uses 90 samples; no startup or HTTP.",
        "comparisons": [],
    }
    for filename in ["combined.json", "combined-confirmation.json", "best.json"]:
        report = json.loads((root / filename).read_text())
        rows = report["benchmark"]["rows"]
        for case in dict.fromkeys(row["case"] for row in rows):
            values = {
                name: [
                    value
                    for row in rows
                    if row["case"] == case and row["variant"] == name
                    for value in row["samples_ms"]
                ]
                for name in ["native-window", "fusion"]
            }
            base, fused = (
                float(np.median(values[name])) for name in ["native-window", "fusion"]
            )
            summary["comparisons"].append(
                {
                    "report": filename,
                    "case": case,
                    "samples_per_variant": len(values["fusion"]),
                    "baseline_p50_ms": base,
                    "fusion_p50_ms": fused,
                    "speedup": base / fused,
                    "latency_reduction_pct": 100 * (1 - fused / base),
                }
            )
    best = json.loads((root / "best.json").read_text())
    summary["best_configuration"] = best["configuration"]
    summary["validation"] = {
        key: val for key, val in best["validation"].items() if key != "cases"
    }
    summary["benchmark_probe_decisions"] = sum(
        row["decisions"] for row in best["benchmark_parity"]
    )
    summary["benchmark_probes_exact"] = all(
        row["exact_logits_and_actions"] for row in best["benchmark_parity"]
    )
    summary["baseline_probe_exact"] = (
        best["baseline_reference_probe"]["exact_logits_all"]
        and best["baseline_reference_probe"]["exact_actions_all"]
    )
    summary["extra_packed_weight_bytes"] = best["fusion"]["packed_weight_bytes"]
    summary["conclusion"] = (
        "Useful for larger batches. Small-shape microkernel wins did not give stable full-request wins, so best.json retains the existing projections at <=512 rows and QKV at <=2048 rows."
    )
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        json.dumps(
            [row for row in summary["comparisons"] if row["report"] == "best.json"],
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
