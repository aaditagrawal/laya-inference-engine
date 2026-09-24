"""Full-request parity and paired timings for optional head kernels."""

import hashlib
import json
import random
import time
from pathlib import Path

import torch

from experiments.native import common

from .engine import FrontierEngine
from .head_install import install
from .holdout import requests


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    settings = {
        "policy": "bf16-splitk-exact-short-compiled",
        "attention": "native",
        "fuse_reduce_norm": True,
        "token_tables": True,
        "max_graphs": 6,
    }
    report = {
        "metadata": common.metadata(),
        "settings": settings,
        "source_sha256": {
            name: hashlib.sha256(
                Path(__file__).with_name(name).read_bytes()
            ).hexdigest()
            for name in ["head_gemm.py", "head_install.py", "head_compare.py"]
        },
    }
    path = Path("results/frontier/head-full.json")
    with (
        FrontierEngine(**settings) as baseline,
        FrontierEngine(**settings) as candidate,
    ):
        report["selection"] = install(candidate.base.model.original)
        details = []
        fixtures = requests()
        for i, request in enumerate(fixtures):
            prepared = candidate.prepare(**request)
            actual = candidate.run_prepared(prepared)
            expected = baseline.run_prepared(baseline.prepare(**request))
            details.append(
                {
                    "index": i,
                    **common.compare_outputs(
                        actual, expected, prepared, candidate.agent
                    ),
                }
            )
        report["holdout"] = {
            "details": details,
            "all_exact": all(row["exact_logits_and_actions"] for row in details),
            "all_passed": all(row["passed"] for row in details),
        }
        path.write_text(json.dumps(report, indent=2) + "\n")
        print("HOLDOUT", report["holdout"]["all_exact"], flush=True)
        report["fixed"] = common.benchmark_interleaved(
            {"retained": baseline, "head-kernels": candidate},
            cases=[(1, "short")],
            rounds=9,
            repeats=100,
        )
        rng = random.Random(87241)
        report["changing"] = []
        for round_id in range(9):
            indices = list(range(len(fixtures)))
            rng.shuffle(indices)
            order = ["retained", "head-kernels"]
            rng.shuffle(order)
            for name in order:
                engine = baseline if name == "retained" else candidate
                samples = []
                for index in indices:
                    start = time.perf_counter()
                    engine.predict(**fixtures[index])
                    samples.append((time.perf_counter() - start) * 1000)
                report["changing"].append(
                    {"variant": name, "round": round_id, **common.stats(samples)}
                )
        if report["holdout"]["all_exact"]:
            report["original_validation"] = common.validate(candidate)
        path.write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "all_exact": report["holdout"]["all_exact"],
                    "fixed_summary": report["fixed"].get("summary"),
                    "selection": list(report["selection"]),
                }
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
