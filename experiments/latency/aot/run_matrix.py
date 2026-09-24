"""Run independent deployment processes, locking each GPU phase separately."""

import argparse
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cache-root", required=True, type=Path)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--modes", nargs="+", default=["aot", "native", "compile"])
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for run in range(args.runs):
        order = args.modes
        order = order[run % len(order) :] + order[: run % len(order)]
        for mode in order:
            states = ["fresh", "reused"] if mode == "compile" else ["fresh"]
            for state in states:
                name = f"request-{mode}-{state}-{run}"
                cache = args.cache_root / f"{mode}-{run}"
                env = dict(os.environ)
                env["TORCHINDUCTOR_CACHE_DIR"] = str((cache / "inductor").resolve())
                env["TRITON_CACHE_DIR"] = str((cache / "triton").resolve())
                destination = args.output / f"{name}.json"
                command = [
                    "flock",
                    "/tmp/laya-gpu-experiments.lock",
                    "timeout",
                    "180",
                    sys.executable,
                    "-m",
                    "experiments.latency.aot.request_startup",
                    "--mode",
                    mode,
                    "--cache-state",
                    state,
                    "--package",
                    str(args.package),
                    "--output",
                    str(destination),
                ]
                print(f"Queued {name}", flush=True)
                with (args.output / f"{name}.log").open("w") as log:
                    subprocess.run(
                        command,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
                report = json.loads(destination.read_text())
                if report["status"] != "complete":
                    raise RuntimeError(f"{name} failed: {report.get('error')}")
                rows.append(
                    {
                        "mode": mode,
                        "cache_state": state,
                        "run": run,
                        "file": destination.name,
                        "entry_to_first_response_ms": report[
                            "entry_to_first_response_ms"
                        ],
                        "warm_predict_p50_ms": report["warm_predict_p50_ms"],
                        "exact_outputs": report["exact_outputs"],
                        "prepared_inputs_exact": report["prepared_inputs_exact"],
                    }
                )
                print(json.dumps(rows[-1]), flush=True)
                (args.output / "matrix-progress.json").write_text(
                    json.dumps(rows, indent=2) + "\n"
                )
    summary = []
    for mode, state in dict.fromkeys((r["mode"], r["cache_state"]) for r in rows):
        subset = [r for r in rows if r["mode"] == mode and r["cache_state"] == state]
        startup = [r["entry_to_first_response_ms"] for r in subset]
        warm = [r["warm_predict_p50_ms"] for r in subset]
        summary.append(
            {
                "mode": mode,
                "cache_state": state,
                "processes": len(subset),
                "startup_median_ms": statistics.median(startup),
                "startup_min_ms": min(startup),
                "startup_max_ms": max(startup),
                "warm_predict_median_ms": statistics.median(warm),
                "all_exact": all(
                    all(r["exact_outputs"]) and r["prepared_inputs_exact"]
                    for r in subset
                ),
            }
        )
    (args.output / "matrix-summary.json").write_text(
        json.dumps({"rows": summary, "processes": rows}, indent=2) + "\n"
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
