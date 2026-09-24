"""Parity and kernel-only timing with retained real attention bias."""

import json
from pathlib import Path

import torch

from experiments.native import common

from .attention_special_adapter import load
from .attention_special_probe import capture
from .engine import FrontierEngine
from .holdout import requests
from .native_attention import load as load_retained
from .tune import timing


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    load()
    load_retained()
    report = {
        "metadata": common.metadata(),
        "scope": "18 local attention kernels per graph; bias constructed before capture; 9 real requests including padding",
        "build": json.loads(
            Path(".research/frontier-attention-special/build.json").read_text()
        ),
        "rows": [],
    }
    with FrontierEngine(policy="bf16-exact", max_graphs=2) as engine:
        fixtures = [common.workload(1, "short"), *requests()[::16]]
        groups = []
        for request in fixtures:
            tensors = capture(engine, request, allow_padding=True)
            groups.append(
                [
                    (
                        q,
                        k,
                        v,
                        torch.zeros_like(mask, dtype=q.dtype).masked_fill_(
                            ~mask, float("-inf")
                        ),
                    )
                    for q, k, v, mask in tensors
                ]
            )
        report["requests"] = fixtures
        expected = [
            [
                torch.ops.laya_frontier_attention.forward(q, k, v, bias, 32)
                for q, k, v, bias in group
            ]
            for group in groups
        ]
        first = groups[0]

        def baseline():
            return [
                torch.ops.laya_frontier_attention.forward(q, k, v, bias, 32)
                for q, k, v, bias in first
            ]

        report["baseline_ms"], report["baseline_samples_ms"] = timing(
            baseline, repeats=100, rounds=7
        )
        for tile in [32, 64]:
            for special in [False, True]:
                config = (tile, True, special)
                mismatch = []
                for group, ref in zip(groups, expected):
                    actual = [
                        torch.ops.laya_frontier_attention_special.forward(
                            q, k, v, *config, bias
                        )
                        for q, k, v, bias in group
                    ]
                    mismatch.append(
                        sum(int((a != b).sum()) for a, b in zip(actual, ref))
                    )

                def candidate(config=config):
                    return [
                        torch.ops.laya_frontier_attention_special.forward(
                            q, k, v, *config, bias
                        )
                        for q, k, v, bias in first
                    ]

                elapsed, samples = timing(candidate, repeats=100, rounds=7)
                row = {
                    "configuration": config,
                    "request_mismatches": mismatch,
                    "mismatches": sum(mismatch),
                    "ms": elapsed,
                    "samples_ms": samples,
                    "speedup": report["baseline_ms"] / elapsed,
                }
                report["rows"].append(row)
                print(row, flush=True)
                Path("results/frontier/attention-special-padding.json").write_text(
                    json.dumps(report, indent=2) + "\n"
                )


if __name__ == "__main__":
    main()
