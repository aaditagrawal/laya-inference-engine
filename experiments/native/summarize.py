"""Summarize raw local reports without mixing samples from different runs."""

import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/native-optimizations"


def summarize_interleaved(path):
    raw = json.loads(path.read_text())
    grouped = {}
    for row in raw.get("timings", {}).get("rows", []):
        grouped.setdefault((row["case"], row["variant"]), []).append(row)
    rows = []
    for (case, variant), blocks in grouped.items():
        samples = [value for block in blocks for value in block["samples_ms"]]
        baseline = grouped[(case, "baseline")]
        base_median = np.median(
            [value for block in baseline for value in block["samples_ms"]]
        )
        median = float(np.median(samples))
        rows.append(
            {
                "case": case,
                "variant": variant,
                "p50_ms": median,
                "p95_ms": float(np.percentile(samples, 95)),
                "mean_ms": float(np.mean(samples)),
                "decisions_per_second": blocks[0]["questions"]
                * 1000
                / float(np.mean(samples)),
                "latency_reduction_percent": float(100 * (1 - median / base_median)),
                "sample_count": len(samples),
                "rounds": len(blocks),
                "round_p50_ms": [block["p50_ms"] for block in blocks],
            }
        )
    validation = raw.get("validation", {})
    return {
        "file": path.name,
        "status": raw.get("status"),
        "rows": rows,
        "validation": {
            key: value for key, value in validation.items() if key != "cases"
        },
        "metadata": raw.get("metadata"),
        "first_shape_ms": raw.get("first_shape_ms"),
    }


def main():
    reports = []
    for path in sorted(RESULTS.glob("*.json")):
        if path.name in {"summary.json", "manifest.json"}:
            continue
        raw = json.loads(path.read_text())
        if "timings" in raw:
            reports.append(summarize_interleaved(path))
    negative = []
    for path in sorted((RESULTS / "precision").glob("fp8-*.json")):
        raw = json.loads(path.read_text())
        validation = raw.get("validation", {})
        negative.append(
            {
                "variant": raw["variant"],
                "p50_ms": {row["case"]: row["p50_ms"] for row in raw.get("rows", [])},
                "agreement": validation.get("agreement"),
                "decisions": validation.get("decisions"),
                "max_probability_error": validation.get("max_probability_error"),
                "accepted": False,
                "file": str(path.relative_to(RESULTS)),
            }
        )
    summary = {
        "comparisons": reports,
        "selective_fp8": negative,
        "scope": "Same-device software experiments on RTX 5070 Ti SM120. Synthetic implementation parity, not labeled model accuracy. No hardware-generation attribution.",
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    manifest = {}
    for directory in (
        ROOT / "experiments/native",
        ROOT / "src/laya_blackwell",
        RESULTS,
    ):
        for path in sorted(directory.rglob("*")):
            if (
                path.is_file()
                and path.suffix in {".py", ".cpp", ".cu", ".json", ".npz", ".md"}
                and path.name != "manifest.json"
            ):
                manifest[str(path.relative_to(ROOT))] = hashlib.sha256(
                    path.read_bytes()
                ).hexdigest()
    (RESULTS / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for report in reports:
        print(
            report["file"],
            [
                (row["case"], row["variant"], round(row["p50_ms"], 3))
                for row in report["rows"]
            ],
        )


if __name__ == "__main__":
    main()
