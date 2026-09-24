"""Parity check for the full packed-QKV variant screened out on performance."""

import json
from pathlib import Path

import torch

from experiments.native import common

from .engine import FrontierEngine
from .holdout import requests


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    options = {
        "policy": "bf16-splitk-exact-short-compiled",
        "attention": "native",
        "fuse_reduce_norm": True,
        "token_tables": True,
        "max_graphs": 4,
    }
    report = {"metadata": common.metadata(), "details": []}
    with (
        FrontierEngine(**options) as baseline,
        FrontierEngine(**options, packed_qkv=True) as candidate,
    ):
        report["selection"] = candidate.selection["packed_qkv"]
        for request in requests():
            prepared = candidate.prepare(**request)
            actual = candidate.run_prepared(prepared)
            reference = baseline.run_prepared(baseline.prepare(**request))
            report["details"].append(
                common.compare_outputs(actual, reference, prepared, candidate.agent)
            )
    report["all_exact"] = all(r["exact_logits_and_actions"] for r in report["details"])
    Path("results/frontier/packed-qkv-parity.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(
        "Requests", len(report["details"]), "all exact", report["all_exact"], flush=True
    )


if __name__ == "__main__":
    main()
