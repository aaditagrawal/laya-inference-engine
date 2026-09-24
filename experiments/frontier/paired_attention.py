"""Paired confirmation of a small attention win against the previous best."""

import argparse
import json
import random
import statistics
import time
from pathlib import Path

import torch

from experiments.native import common

from .engine import FrontierEngine
from .holdout import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-policy", default="bf16-exact-short-compiled")
    parser.add_argument("--baseline-attention", choices=["native"])
    parser.add_argument("--candidate-policy", default="bf16-exact-short-compiled")
    parser.add_argument("--short-only", action="store_true")
    parser.add_argument("--candidate-fuse-reduce-norm", action="store_true")
    parser.add_argument("--baseline-fuse-reduce-norm", action="store_true")
    parser.add_argument("--candidate-token-tables", action="store_true")
    parser.add_argument("--baseline-token-tables", action="store_true")
    parser.add_argument("--candidate-packed-qkv", action="store_true")
    parser.add_argument("--candidate-fuse-mlp-geglu", action="store_true")
    parser.add_argument("--baseline-fuse-mlp-geglu", action="store_true")
    parser.add_argument("--candidate-mlp-geglu-unpacked", action="store_true")
    parser.add_argument("--baseline-mlp-geglu-unpacked", action="store_true")
    parser.add_argument("--candidate-head-kernels", action="store_true")
    parser.add_argument(
        "--attention-special",
        "--candidate-attention-special",
        dest="candidate_attention_special",
        action="store_true",
    )
    parser.add_argument("--baseline-attention-special", action="store_true")
    parser.add_argument("--baseline-global-attention", action="store_true")
    parser.add_argument("--candidate-global-attention", action="store_true")
    parser.add_argument("--baseline-host-runtime", action="store_true")
    parser.add_argument("--candidate-host-runtime", action="store_true")
    parser.add_argument(
        "--baseline-host-prepare", choices=["single", "batch", "template"]
    )
    parser.add_argument("--baseline-head-kernels", action="store_true")
    parser.add_argument(
        "--candidate-host-prepare", choices=["single", "batch", "template"]
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/frontier/paired-native-attention.json"),
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    report = {
        "metadata": common.metadata(),
        "configuration": {
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
    }
    with (
        FrontierEngine(
            policy=args.baseline_policy,
            attention=args.baseline_attention,
            fuse_reduce_norm=args.baseline_fuse_reduce_norm,
            token_tables=args.baseline_token_tables,
            fuse_mlp_geglu=args.baseline_fuse_mlp_geglu,
            mlp_geglu_unpacked=args.baseline_mlp_geglu_unpacked,
            head_kernels=args.baseline_head_kernels,
            host_prepare=args.baseline_host_prepare,
            attention_special=args.baseline_attention_special,
            global_attention=args.baseline_global_attention,
            host_runtime=args.baseline_host_runtime,
            max_graphs=6,
        ) as prior,
        FrontierEngine(
            policy=args.candidate_policy,
            attention="native",
            fuse_reduce_norm=args.candidate_fuse_reduce_norm,
            token_tables=args.candidate_token_tables,
            packed_qkv=args.candidate_packed_qkv,
            fuse_mlp_geglu=args.candidate_fuse_mlp_geglu,
            mlp_geglu_unpacked=args.candidate_mlp_geglu_unpacked,
            head_kernels=args.candidate_head_kernels,
            host_prepare=args.candidate_host_prepare,
            attention_special=args.candidate_attention_special,
            global_attention=args.candidate_global_attention,
            host_runtime=args.candidate_host_runtime,
            max_graphs=6,
        ) as candidate,
    ):
        engines = {"previous-best": prior, "native-attention": candidate}
        report["fixed"] = common.benchmark_interleaved(
            engines,
            cases=[(1, "short")]
            if args.short_only
            else [(1, "short"), (1, "long"), (16, "short")],
            rounds=9,
            repeats=100,
        )
        fixtures = requests()
        report["changing_inputs"] = {"order": [], "rows": []}
        rng = random.Random(45211)
        for round_id in range(9):
            indices = list(range(len(fixtures)))
            rng.shuffle(indices)
            names = list(engines)
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
        for case in (
            ["1-short", "changing-inputs"]
            if args.short_only
            else ["1-short", "1-long", "16-short", "changing-inputs"]
        ):
            rows = (
                report["changing_inputs"]["rows"]
                if case == "changing-inputs"
                else [r for r in report["fixed"]["rows"] if r["case"] == case]
            )
            pooled = {
                name: statistics.median(
                    [v for r in rows if r["variant"] == name for v in r["samples_ms"]]
                )
                for name in engines
            }
            by_round = {
                name: {r["round"]: r["p50_ms"] for r in rows if r["variant"] == name}
                for name in engines
            }
            diffs = [
                by_round["previous-best"][i] - by_round["native-attention"][i]
                for i in range(9)
            ]
            report["summary"][case] = {
                "median_ms": pooled,
                "paired_round_savings_ms": diffs,
                "faster_rounds": sum(v > 0 for v in diffs),
            }
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
