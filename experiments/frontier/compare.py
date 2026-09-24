"""Full-request confirmation, including original fixtures and changed inputs."""

import argparse
import json
from pathlib import Path

import torch

from experiments.latency.engine import V2Engine
from experiments.native import common

from .engine import FrontierEngine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", default="bf16-exact")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--attention", choices=["triton", "cudnn", "native"])
    parser.add_argument("--fuse-reduce-norm", action="store_true")
    parser.add_argument("--token-tables", action="store_true")
    parser.add_argument("--fuse-mlp-geglu", action="store_true")
    parser.add_argument("--mlp-geglu-unpacked", action="store_true")
    parser.add_argument("--head-kernels", action="store_true")
    parser.add_argument("--attention-special", action="store_true")
    parser.add_argument("--global-attention", action="store_true")
    parser.add_argument("--host-runtime", action="store_true")
    parser.add_argument("--native-format", action="store_true")
    parser.add_argument("--host-prepare", choices=["single", "batch", "template"])
    args = parser.parse_args()
    torch.set_num_threads(4)
    label = args.policy + ("-attn-" + args.attention if args.attention else "")
    if args.fuse_reduce_norm:
        label += "-reduce-norm"
    if args.token_tables:
        label += "-token-tables"
    if args.fuse_mlp_geglu:
        label += "-mlp-geglu"
        if args.mlp_geglu_unpacked:
            label += "-unpacked"
    if args.head_kernels:
        label += "-head-kernels"
    if args.host_prepare:
        label += "-host-" + args.host_prepare
    if args.attention_special:
        label += "-attention-special"
    if args.global_attention:
        label += "-global-attention"
    if args.host_runtime:
        label += "-host-runtime"
    if args.native_format:
        label += "-native-format"
    path = Path(f"results/frontier/full-{label}.json")
    report = {"metadata": common.metadata(), "policy": label}
    with (
        V2Engine(optimization="native", max_graphs=6) as baseline,
        FrontierEngine(
            policy=args.policy,
            attention=args.attention,
            fuse_reduce_norm=args.fuse_reduce_norm,
            token_tables=args.token_tables,
            fuse_mlp_geglu=args.fuse_mlp_geglu,
            mlp_geglu_unpacked=args.mlp_geglu_unpacked,
            head_kernels=args.head_kernels,
            host_prepare=args.host_prepare,
            attention_special=args.attention_special,
            global_attention=args.global_attention,
            host_runtime=args.host_runtime,
            native_format=args.native_format,
            max_graphs=6,
        ) as candidate,
    ):
        report["selection"] = candidate.selection
        report["timings"] = common.benchmark_interleaved(
            {"native": baseline, label: candidate},
            cases=[(1, "short"), (1, "long"), (16, "short")],
            rounds=5,
            repeats=50,
        )
        report["benchmark_parity"] = []
        for batch, length in [(1, "short"), (1, "long"), (16, "short")]:
            request = common.workload(batch, length)
            prepared = candidate.prepare(**request)
            a = candidate.run_prepared(prepared)
            b = baseline.run_prepared(baseline.prepare(**request))
            report["benchmark_parity"].append(
                {
                    "case": f"{batch}-{length}",
                    **common.compare_outputs(a, b, prepared, candidate.agent),
                }
            )
        path.write_text(json.dumps(report, indent=2) + "\n")
        if args.validate:
            report["validation"] = common.validate(candidate)
            report["baseline_validation"] = common.validate(baseline)
            path.write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                {
                    key: value
                    for key, value in report.items()
                    if key not in {"timings", "selection"}
                },
                indent=2,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
