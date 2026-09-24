"""Collect final local experiment results without initializing CUDA."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/latency-optimizations"


def read(path):
    return json.loads(path.read_text())


def timings(report):
    result = []
    for case in dict.fromkeys(row["case"] for row in report["rows"]):
        values = {}
        for variant in dict.fromkeys(row["variant"] for row in report["rows"]):
            rows = [
                r
                for r in report["rows"]
                if r["case"] == case and r["variant"] == variant
            ]
            samples = [sample for row in rows for sample in row["samples_ms"]]
            values[variant] = {
                "p50_ms": float(np.median(samples)),
                "p95_ms": float(np.percentile(samples, 95)),
                "mean_ms": float(np.mean(samples)),
                "samples": len(samples),
                "rounds": len(rows),
            }
        result.append({"case": case, "variants": values})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serving", type=Path, required=True)
    parser.add_argument("--startup", type=Path, required=True)
    parser.add_argument("--integration", type=Path, required=True)
    args = parser.parse_args()
    for field in ("serving", "startup", "integration"):
        setattr(args, field, getattr(args, field).resolve())
    fusion = read(RESULTS / "fusion/best.json")
    fusion_summary = read(RESULTS / "fusion/summary.json")
    padding = read(RESULTS / "padding/batch-exact-confirmed.json")
    sequence = read(RESULTS / "padding/sequence-32.json")
    churn = read(RESULTS / "padding/cache-churn-fixed.json")
    assert churn["status"] == "complete"
    graph_check = read(RESULTS / "serving/graph-churn-fixed.json")
    assert graph_check["status"] == "passed"
    integration = read(args.integration)
    assert integration["status"] == "passed"
    serving = read(args.serving)
    startup = read(args.startup)
    assert serving["rows"] and startup["rows"]
    assert all(row["exact_answers_all"] for row in serving["rows"])
    assert all(row["all_exact"] for row in startup["rows"])
    summary = {
        "metadata": fusion["metadata"],
        "scope": "Second-round software changes on the same RTX 5070 Ti, compared with the previous native-window experimental engine. No HTTP or model-quality evaluation.",
        "fusion": {
            "source": "fusion/best.json",
            "rows": timings(fusion["benchmark"]),
            "validation": fusion_summary["validation"],
            "probe_decisions": fusion_summary["benchmark_probe_decisions"],
            "probe_exact": fusion_summary["benchmark_probes_exact"],
            "packed_weight_bytes": fusion_summary["extra_packed_weight_bytes"],
        },
        "serving": {
            "source": str(args.serving.relative_to(ROOT)),
            **serving,
        },
        "startup": {
            "source": str(args.startup.relative_to(ROOT)),
            **startup,
        },
        "integration": {
            "source": str(args.integration.relative_to(ROOT)),
            **integration,
        },
        "padding": {
            "source": "padding/batch-exact-confirmed.json",
            "rows": timings(padding["timings"]),
            "initial_validation": {
                key: value
                for key, value in padding["validation"].items()
                if key != "cases"
            },
            "expanded_validation": read(RESULTS / "padding/irregular-validation.json")[
                "summary"
            ],
            "compatible_validation": read(
                RESULTS / "padding/compatible-validation.json"
            )["summary"],
            "sequence_source": "padding/sequence-32.json",
            "sequence_rows": timings(sequence["timings"]),
            "sequence_validation": {
                key: value
                for key, value in sequence["validation"].items()
                if key != "cases"
            },
            "cache_source": "padding/cache-churn-fixed.json",
            "cache": {
                name: {key: value for key, value in row.items() if key != "samples_ms"}
                for name, row in churn["summary"].items()
            },
            "accepted_as_default": False,
        },
        "graph_lifetime": {
            "source": "serving/graph-churn-fixed.json",
            "status": graph_check["status"],
            "cycles": graph_check["cycles"],
            "checks": len(graph_check["rows"]),
            "all_exact": all(
                row["exact_logits_and_actions"] for row in graph_check["rows"]
            ),
            "cuda_bytes_after_close": graph_check["cuda_bytes_after_close"],
        },
        "microbatch": read(RESULTS / "serving/microbatch-summary.json"),
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    paths = list(RESULTS.rglob("*.json")) + list(RESULTS.rglob("*.ptx"))
    paths += list((ROOT / "experiments/latency").rglob("*.py"))
    paths += list((ROOT / "experiments/latency").rglob("*.json"))
    paths += list((ROOT / "experiments/native").rglob("*.py"))
    paths += list((ROOT / "src/laya_blackwell").rglob("*.py"))
    manifest = []
    for path in sorted(set(paths)):
        if path.name == "manifest.json" and path.parent == RESULTS:
            continue
        manifest.append(
            {
                "path": str(path.relative_to(ROOT)),
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        )
    (RESULTS / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(RESULTS / "summary.json")


if __name__ == "__main__":
    main()
