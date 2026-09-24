"""Unseen synthetic requests and changing-input latency, never answer caching.

These fixtures check implementation parity, not labeled model quality.
"""

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import torch

from experiments.latency.engine import V2Engine
from experiments.native import common

from .engine import FrontierEngine


def requests():
    domains = [
        (
            ["refund", "delivery", "warranty", "other"],
            "My parcel arrived broken",
            "the card was charged twice",
        ),
        (
            ["login", "network", "storage", "other"],
            "The password reset failed",
            "the office router is offline",
        ),
        (
            ["booking", "cancellation", "baggage", "other"],
            "My suitcase is missing",
            "I want to cancel the flight",
        ),
        (
            ["invoice", "contract", "support", "other"],
            "The annual price seems wrong",
            "our signed contract expired",
        ),
        (
            ["urgent", "normal", "resolved", "unknown"],
            "Production has stopped",
            "the system now works again",
        ),
        (
            ["food", "drinks", "service", "other"],
            "The soup arrived cold",
            "we waited an hour for water",
        ),
        (
            ["bug", "feature", "question", "other"],
            "The export button crashes",
            "I need a weekly report option",
        ),
        (
            ["positive", "negative", "neutral", "mixed"],
            "The update is wonderful",
            "the new interface is confusing",
        ),
    ]
    result = []
    for labels, first, second in domains:
        states = [
            first + ".",
            second + ".",
            first + ", but that is already fixed.",
            "Yesterday: " + first + ". Today: " + second + ".",
            "Ignore the earlier report. " + second + ".",
            "Someone wrote '" + first + "'. I disagree.",
            "This is a test case. " + first + ".",
            "Please explain the options. I have no request yet.",
            first + ". Can this wait until Monday?",
            first + ". Please help immediately.",
            second + ". I only need documentation.",
            "Thank you, that is resolved. " + first + " was last week.",
            "I am unsure whether this applies: " + second + ".",
            "Case #2309, café team: " + first + ".",
            "[MASK] [SEP] " + second + ".",
            "",
        ]
        for index, state in enumerate(states):
            result.append(
                {
                    "state": state,
                    "questions": {
                        "route": {
                            "type": "choice",
                            "instructions": [
                                "Classify this request.",
                                "Choose the best category.",
                                "What is the main issue?",
                            ][index % 3],
                            "criteria": {label: label for label in labels},
                        }
                    },
                }
            )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policies",
        nargs="+",
        default=["bf16-exact", "bf16-fast", "bf16-exact-compiled"],
    )
    parser.add_argument(
        "--output", type=Path, default=Path("results/frontier/holdout.json")
    )
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
    fixtures = requests()
    report = {
        "metadata": common.metadata(),
        "requests": fixtures,
        "request_sha256": hashlib.sha256(
            json.dumps(fixtures, sort_keys=True).encode()
        ).hexdigest(),
        "variants": {},
    }
    path = args.output
    with V2Engine(optimization="native", max_graphs=6) as baseline:
        for policy in args.policies:
            with FrontierEngine(
                policy=policy,
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
            ) as candidate:
                policy = candidate.policy
                details = []
                for i, request in enumerate(fixtures):
                    prepared = candidate.prepare(**request)
                    actual = candidate.run_prepared(prepared)
                    reference = baseline.run_prepared(baseline.prepare(**request))
                    details.append(
                        {
                            "index": i,
                            "shape": candidate.base._shape(prepared),
                            **common.compare_outputs(
                                actual, reference, prepared, candidate.agent
                            ),
                        }
                    )
                # Warm both shape sets before timing. Each full predict retokenizes.
                samples = {"native": [], policy: []}
                order = []
                rng = random.Random(39019)
                for _ in range(5):
                    indices = list(range(len(fixtures)))
                    rng.shuffle(indices)
                    names = list(samples)
                    rng.shuffle(names)
                    order.append({"variants": names, "indices": indices})
                    for name in names:
                        engine = baseline if name == "native" else candidate
                        for index in indices:
                            start = time.perf_counter()
                            engine.predict(**fixtures[index])
                            samples[name].append((time.perf_counter() - start) * 1000)
                row = {
                    "details": details,
                    "all_passed": all(d["passed"] for d in details),
                    "all_exact": all(d["exact_logits_and_actions"] for d in details),
                    "agreement": sum(d["agreement"] for d in details),
                    "max_probability_error": max(
                        d["max_probability_error"] for d in details
                    ),
                    "optimized_shape_requests": sum(
                        d["shape"][0] * d["shape"][1] == 64 for d in details
                    ),
                    "timings": {
                        name: common.stats(values) for name, values in samples.items()
                    },
                    "order": order,
                }
                report["variants"][policy] = row
                path.write_text(json.dumps(report, indent=2) + "\n")
                print(
                    policy,
                    {
                        k: v
                        for k, v in row.items()
                        if k not in {"details", "timings", "order"}
                    },
                    {name: values["p50_ms"] for name, values in row["timings"].items()},
                    flush=True,
                )


if __name__ == "__main__":
    main()
