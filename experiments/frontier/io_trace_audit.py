"""Audit captured transfer durations and adjacent gaps in an existing trace.

No new GPU execution. Instrumentation changes scheduling, so these observations
are neither uninstrumented latency nor a bound on mapped-memory improvements.
"""

import argparse
import collections
import hashlib
import json
import statistics
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("results/frontier/io-trace-audit.json")
    )
    args = parser.parse_args()
    raw = args.trace.read_bytes()
    groups = collections.defaultdict(list)
    for event in json.loads(raw)["traceEvents"]:
        if event.get("cat") in {"kernel", "gpu_memcpy", "gpu_memset"}:
            fields = event.get("args", {})
            if "graph id" in fields and "correlation" in fields:
                key = (fields["device"], fields["graph id"], fields["correlation"])
                groups[key].append(event)
    rows = []
    for key, events in sorted(groups.items()):
        events.sort(key=lambda event: event["ts"])
        copies = []
        for index, event in enumerate(events):
            if event["cat"] != "gpu_memcpy":
                continue
            previous = events[index - 1] if index else None
            following = events[index + 1] if index + 1 < len(events) else None
            copies.append(
                {
                    "name": event["name"],
                    "bytes": event["args"]["bytes"],
                    "duration_us": event["dur"],
                    "previous_gap_us": None
                    if previous is None
                    else event["ts"] - previous["ts"] - previous["dur"],
                    "following_gap_us": None
                    if following is None
                    else following["ts"] - event["ts"] - event["dur"],
                }
            )
        if copies:
            rows.append(
                {
                    "device_graph_correlation": list(key),
                    "kernels": sum(event["cat"] == "kernel" for event in events),
                    "memsets": sum(event["cat"] == "gpu_memset" for event in events),
                    "copies": copies,
                    "copy_duration_us": sum(row["duration_us"] for row in copies),
                    "device_span_us": max(e["ts"] + e["dur"] for e in events)
                    - events[0]["ts"],
                }
            )
    if not rows:
        raise ValueError("Trace has no graph-correlated GPU transfers")
    report = {
        "scope": "Read-only instrumented trace audit; no new GPU timing or execution",
        "limitation": "Transfer durations and gaps are instrumented observations, not an additive request timing or a bound on mapped-memory gains.",
        "trace": str(args.trace),
        "trace_sha256": hashlib.sha256(raw).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "graph_replays": len(rows),
        "mean_copy_duration_us": statistics.mean(r["copy_duration_us"] for r in rows),
        "median_copy_duration_us": statistics.median(
            r["copy_duration_us"] for r in rows
        ),
        "rows": rows,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2))


if __name__ == "__main__":
    main()
