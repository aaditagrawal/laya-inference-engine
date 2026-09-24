"""Compare exact fusion, head and host combinations in one isolated process."""

import hashlib
import json
import random
import statistics
import time
from contextlib import ExitStack
from pathlib import Path

import torch

from experiments.native import common

from .engine import FrontierEngine
from .holdout import requests


def main():
    torch.set_num_threads(4)
    settings = {
        "policy": "bf16-splitk-exact-short-compiled",
        "attention": "native",
        "fuse_reduce_norm": True,
        "token_tables": True,
        "fuse_mlp_geglu": True,
        "mlp_geglu_unpacked": True,
        "max_graphs": 6,
    }
    variants = {
        "mlp": {},
        "mlp-head": {"head_kernels": True},
        "mlp-host": {"host_prepare": "batch"},
        "mlp-head-host": {"head_kernels": True, "host_prepare": "batch"},
    }
    path = Path("results/frontier/stack-comparison.json")
    report = {
        "metadata": common.metadata(),
        "settings": settings,
        "variants": variants,
        "scope": "Full warm requests. Exclusive benchmark lock; sibling CPU benchmarks finished before this run. Each variant resident. No prepared-request or answer cache.",
        "source_sha256": {
            name: hashlib.sha256(
                Path(__file__).with_name(name).read_bytes()
            ).hexdigest()
            for name in [
                "stack_compare.py",
                "engine.py",
                "mlp_geglu.py",
                "head_install.py",
                "head_gemm.py",
                "host_prepare.py",
            ]
        },
    }
    with ExitStack() as stack:
        engines = {
            name: stack.enter_context(FrontierEngine(**settings, **options))
            for name, options in variants.items()
        }
        fixtures = requests()
        parity = {name: [] for name in engines if name != "mlp"}
        for request in fixtures:
            prepared = engines["mlp"].prepare(**request)
            reference = engines["mlp"].run_prepared(prepared)
            for name, details in parity.items():
                candidate = engines[name]
                prepared = candidate.prepare(**request)
                details.append(
                    common.compare_outputs(
                        candidate.run_prepared(prepared),
                        reference,
                        prepared,
                        candidate.agent,
                    )
                )
        report["holdout_vs_mlp"] = {
            name: {
                "all_exact": all(row["exact_logits_and_actions"] for row in rows),
                "details": rows,
            }
            for name, rows in parity.items()
        }
        path.write_text(json.dumps(report, indent=2) + "\n")
        report["fixed"] = common.benchmark_interleaved(
            engines, cases=[(1, "short")], rounds=9, repeats=100
        )
        report["changing_inputs"] = {"order": [], "rows": []}
        rng = random.Random(195407)
        for round_id in range(9):
            indices = list(range(len(fixtures)))
            names = list(engines)
            rng.shuffle(indices)
            rng.shuffle(names)
            report["changing_inputs"]["order"].append(
                {"variants": names, "indices": indices}
            )
            for name in names:
                samples = []
                for index in indices:
                    start = time.perf_counter()
                    engines[name].predict(**fixtures[index])
                    samples.append((time.perf_counter() - start) * 1000)
                report["changing_inputs"]["rows"].append(
                    {"variant": name, "round": round_id, **common.stats(samples)}
                )
        report["summary"] = {}
        for label, key in [("fixed", "fixed"), ("changing", "changing_inputs")]:
            rows = report[key]["rows"]
            medians = {
                name: statistics.median(
                    [
                        v
                        for row in rows
                        if row["variant"] == name
                        for v in row["samples_ms"]
                    ]
                )
                for name in engines
            }
            by_round = {
                name: {r["round"]: r["p50_ms"] for r in rows if r["variant"] == name}
                for name in engines
            }
            report["summary"][label] = {
                "median_ms": medians,
                "faster_rounds_vs_mlp": {
                    name: sum(by_round[name][i] < by_round["mlp"][i] for i in range(9))
                    for name in engines
                    if name != "mlp"
                },
                "head_added_to_host_faster_rounds": sum(
                    by_round["mlp-head-host"][i] < by_round["mlp-host"][i]
                    for i in range(9)
                ),
            }
        path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
