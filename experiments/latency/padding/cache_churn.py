"""Measure the cost of a shape working set larger than an eight-graph cache."""

import argparse
import json
import random
import time
import traceback
from pathlib import Path

import torch

from experiments.native import common

from .engine import DenseEngine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stable-host", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    report = {
        "metadata": common.metadata(),
        "max_graphs": 8,
        "requests_per_cycle": 10,
        "stable_host": args.stable_host,
        "method": "Counts 1..10 short, in order. One complete untimed cycle then two timed cycles per block. Three randomized blocks. Includes graph recapture, excludes initial model load and initial warmup.",
    }
    rows = report["rows"] = []
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    with (
        DenseEngine(
            model=common.model_path(), shape_policy="stock", max_graphs=8
        ) as baseline,
        DenseEngine(
            model=common.model_path(), shape_policy="batch-exact", max_graphs=8
        ) as candidate,
    ):
        engines = {"baseline": baseline, "candidate": candidate}
        if args.stable_host:
            from experiments.latency.serving.graph_adapter import replace_adapter

            for engine in engines.values():
                replace_adapter(engine)
        requests = [common.workload(count, "short") for count in range(1, 11)]
        for engine in engines.values():
            for request in requests:
                engine.predict(**request)
        rng = random.Random(982451653)
        for block in range(3):
            names = list(engines)
            rng.shuffle(names)
            for name in names:
                for cycle in range(2):
                    for count, request in enumerate(requests, 1):
                        report["current"] = {
                            "variant": name,
                            "block": block,
                            "cycle": cycle,
                            "questions": count,
                        }
                        save()
                        start = time.perf_counter()
                        try:
                            response = engines[name].predict(**request)
                        except BaseException:
                            report["status"] = "failed"
                            report["traceback"] = traceback.format_exc()
                            save()
                            raise
                        rows.append(
                            {
                                "variant": name,
                                "block": block,
                                "cycle": cycle,
                                "questions": count,
                                "elapsed_ms": (time.perf_counter() - start) * 1000,
                                **response["engine"],
                            }
                        )
                        save()
                print(
                    name,
                    block,
                    sum(r["graph_miss"] for r in rows if r["variant"] == name),
                    flush=True,
                )
    report["summary"] = {}
    for name in ("baseline", "candidate"):
        selected = [r for r in rows if r["variant"] == name]
        report["summary"][name] = {
            **common.stats([r["elapsed_ms"] for r in selected]),
            "graph_misses": sum(r["graph_miss"] for r in selected),
            "requests": len(selected),
        }
    report["status"] = "complete"
    save()


if __name__ == "__main__":
    main()
