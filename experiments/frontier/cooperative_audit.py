"""Read-only trace audit of exposed projection/normalization boundary gaps."""

import argparse
import collections
import hashlib
import itertools
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--replays", type=int, default=10)
    parser.add_argument(
        "--output", type=Path, default=Path("results/frontier/cooperative-audit.json")
    )
    args = parser.parse_args()
    if args.replays < 1:
        parser.error("--replays must be positive")
    raw = args.trace.read_bytes()
    events = sorted(
        (e for e in json.loads(raw)["traceEvents"] if e.get("cat") == "kernel"),
        key=lambda e: e["ts"],
    )
    norm = "void norm_kernel_vector<true, false, false, false>"

    def name(event):
        return "residual_norm" if event["name"].startswith(norm) else event["name"]

    selected = {
        ("_matmul", "residual_norm"),
        ("residual_norm", "_project"),
        ("residual_norm", "_tma"),
    }
    pairs = collections.defaultdict(list)
    for previous, following in itertools.pairwise(events):
        key = (name(previous), name(following))
        if key in selected:
            pairs[key].append(following["ts"] - previous["ts"] - previous["dur"])
    rows = [
        {
            "previous": previous,
            "following": following,
            "count": len(gaps),
            "gap_us_per_replay": sum(gaps) / args.replays,
            "overlapping_pairs": sum(gap < 0 for gap in gaps),
        }
        for (previous, following), gaps in sorted(pairs.items())
    ]
    report = {
        "scope": "Instrumented trace audit; no new timing or GPU execution",
        "method": "Sort kernel events by timestamp and sum next.start minus previous.end for the recorded adjacent stage pairs.",
        "decision": "Do not implement a cooperative rewrite aimed only at eliminating these small exposed boundary gaps.",
        "limitation": "Observed trace gaps are not a universal bound on fusion gains. Normalization arithmetic and intermediate storage remain necessary with the existing distributed projection tiles.",
        "trace": str(args.trace),
        "trace_sha256": hashlib.sha256(raw).hexdigest(),
        "replays": args.replays,
        "kernel_events": len(events),
        "pairs": rows,
        "selected_gap_us_per_replay": sum(r["gap_us_per_replay"] for r in rows),
        "residual_norm_count_per_replay": sum(
            name(e) == "residual_norm" for e in events
        )
        / args.replays,
        "residual_norm_us_per_replay": sum(
            e["dur"] for e in events if name(e) == "residual_norm"
        )
        / args.replays,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
