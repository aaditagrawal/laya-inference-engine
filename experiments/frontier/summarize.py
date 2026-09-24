"""Summarize measured samples and parity gates without selecting a lucky run."""

import hashlib
import json
import statistics
import subprocess
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/frontier"


def main():
    holdouts = {}
    for path in sorted(RESULTS.glob("holdout*.json")):
        holdouts.update(json.loads(path.read_text())["variants"])
    summary = {
        "created_utc": datetime.now(UTC).isoformat(),
        "target": "Under 1 ms full warm single-question request latency, without an answer cache",
        "target_achieved": False,
        "rows": [],
    }
    for path in sorted(RESULTS.glob("full-*.json")):
        data = json.loads(path.read_text())
        policy = data["policy"]
        validation = data["validation"]
        extra = holdouts.get(policy)
        timings = {}
        for case in ("1-short", "1-long", "16-short"):
            timings[case] = {}
            for name in ("native", policy):
                samples = [
                    sample
                    for row in data["timings"]["rows"]
                    if row["case"] == case and row["variant"] == name
                    for sample in row["samples_ms"]
                ]
                timings[case][name] = {
                    "p50_ms": statistics.median(samples),
                    "mean_ms": statistics.mean(samples),
                    "samples": len(samples),
                }
            timings[case]["latency_reduction_percent"] = 100 * (
                1 - timings[case][policy]["p50_ms"] / timings[case]["native"]["p50_ms"]
            )
        original_passed = (
            validation["passed_current_engine"] and validation["passed_actions"]
        )
        formatter_validation = None
        if policy.endswith("-native-format"):
            cpu = json.loads((RESULTS / "native-format.json").read_text())
            public = json.loads((RESULTS / "native-format-full.json").read_text())
            sources_match = cpu["source_sha256"] == public["source_sha256"] and all(
                hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest
                for name, digest in public["source_sha256"].items()
                if name.endswith(
                    (
                        "/native_format.cpp",
                        "/native_format.py",
                        "/native_format_build.py",
                    )
                )
            )
            formatter_validation = {
                "all_passed": cpu["all_exact"]
                and public["parity"]["all_exact"]
                and sources_match
                and cpu["build"]["library_sha256"] == public["build"]["library_sha256"],
                "cpu_cases": cpu["checks"],
                "public_response_requests": public["parity"]["requests"],
                "source": "results/frontier/native-format-full.json",
            }
        accepted = (
            original_passed
            and validation["exact_logits_all"]
            and validation["exact_actions_all"]
            and extra is not None
            and extra["all_passed"]
            and extra["all_exact"]
            and (formatter_validation is None or formatter_validation["all_passed"])
        )
        summary["rows"].append(
            {
                "policy": policy,
                "source": str(path.relative_to(ROOT)),
                "timings": timings,
                "original_gate_passed": original_passed,
                "original_decision_agreement": validation["agreement"],
                "original_decisions": validation["decisions"],
                "original_max_probability_error": validation["max_probability_error"],
                "original_exact": validation["exact_logits_all"]
                and validation["exact_actions_all"],
                "holdout": None
                if extra is None
                else {
                    key: extra[key]
                    for key in (
                        "all_passed",
                        "all_exact",
                        "agreement",
                        "max_probability_error",
                        "optimized_shape_requests",
                    )
                },
                "accepted_after_both_suites": bool(accepted),
                "formatter_validation": formatter_validation,
            }
        )
    winners = [row for row in summary["rows"] if row["accepted_after_both_suites"]]
    summary["best_measured_accepted_ms"] = min(
        row["timings"]["1-short"][row["policy"]]["p50_ms"] for row in winners
    )
    summary["target_achieved"] = summary["best_measured_accepted_ms"] < 1
    recommended = min(
        (row for row in winners if "short-compiled" in row["policy"]),
        key=lambda row: row["timings"]["1-short"][row["policy"]]["p50_ms"],
    )["policy"]
    summary["recommended_variant"] = recommended
    native_format = recommended.endswith("-native-format")
    without_format = recommended.removesuffix("-native-format")
    host_runtime = without_format.endswith("-host-runtime")
    without_runtime = without_format.removesuffix("-host-runtime")
    global_attention = without_runtime.endswith("-global-attention")
    without_global = without_runtime.removesuffix("-global-attention")
    attention_special = without_global.endswith("-attention-special")
    without_special = without_global.removesuffix("-attention-special")
    without_host, _, host_prepare = without_special.partition("-host-")
    head_kernels = without_host.endswith("-head-kernels")
    without_head = without_host.removesuffix("-head-kernels")
    mlp_geglu_unpacked = without_head.endswith("-mlp-geglu-unpacked")
    with_mlp = (
        without_head.removesuffix("-unpacked") if mlp_geglu_unpacked else without_head
    )
    fuse_mlp_geglu = with_mlp.endswith("-mlp-geglu")
    without_mlp = with_mlp.removesuffix("-mlp-geglu")
    token_tables = without_mlp.endswith("-token-tables")
    without_tables = without_mlp.removesuffix("-token-tables")
    fuse_reduce_norm = without_tables.endswith("-reduce-norm")
    policy, _, attention = without_tables.removesuffix("-reduce-norm").partition(
        "-attn-"
    )
    summary["recommended_constructor"] = {
        "policy": policy,
        "attention": attention or None,
        "fuse_reduce_norm": fuse_reduce_norm,
        "token_tables": token_tables,
        "fuse_mlp_geglu": fuse_mlp_geglu,
        "mlp_geglu_unpacked": mlp_geglu_unpacked,
        "head_kernels": head_kernels,
        "host_prepare": host_prepare or None,
        "attention_special": attention_special,
        "global_attention": global_attention,
        "host_runtime": host_runtime,
        "native_format": native_format,
    }
    (RESULTS / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    files = sorted(
        path for path in (ROOT / "experiments/frontier").iterdir() if path.is_file()
    )
    files += sorted(RESULTS.glob("*.json"))
    files = [path for path in files if path.name != "manifest.json"]
    manifest = {
        "created_utc": datetime.now(UTC).isoformat(),
        "base_repository_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_state": "Hash inventory of the optimization archive. The base commit identifies checkout ancestry; file hashes identify the archived contents.",
        "files_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in files
        },
    }
    (RESULTS / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {key: value for key, value in summary.items() if key != "rows"}, indent=2
        )
    )


if __name__ == "__main__":
    main()
