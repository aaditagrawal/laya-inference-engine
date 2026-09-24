"""Screen exact Torch FlashAttention specializations on real Laya activations."""

import argparse
import json
import random
from pathlib import Path

import torch
from torch.nn import functional as F

from experiments.native import common

from .engine import FrontierEngine
from .global_attention_adapter import CONFIGURATIONS, attention, load
from .holdout import requests
from .tune import timing


@torch.inference_mode()
def capture(engine, request):
    prepared = engine.prepare(**request)
    key = engine.base._graph_key(prepared)
    if key != (1, 64, 4, True):
        return None
    host = engine.base._allocate(key[:3], host=True)
    engine.base._fill(host, prepared)
    inputs = {k: v.to(engine.base.device) for k, v in host.items()}
    inputs["global_attention_unmasked"] = True
    original = F.scaled_dot_product_attention
    tensors = []

    def intercept(q, k, v, attn_mask=None, **kwargs):
        if q.shape == (1, 16, 64, 64) and attn_mask is None:
            tensors.append((q, k, v))
        return original(q, k, v, attn_mask=attn_mask, **kwargs)

    F.scaled_dot_product_attention = intercept
    try:
        engine.base._forward(inputs)
    finally:
        F.scaled_dot_product_attention = original
    if len(tensors) != 10:
        raise RuntimeError(
            f"Expected 10 global attention calls, captured {len(tensors)}"
        )
    return tensors


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/frontier/global-attention-screen.json"),
    )
    parser.add_argument("--configs", type=int, nargs="+", default=list(range(10)))
    args = parser.parse_args()
    torch.set_num_threads(4)
    load()
    build = json.loads(
        Path(".research/frontier-global-attention/build.json").read_text()
    )
    report = {
        "metadata": common.metadata(),
        "build": build,
        "configuration_fields": ["query_tile", "key_tile", "warps", "fixed_dimensions"],
        "configurations": CONFIGURATIONS,
        "scope": "10 unmasked global attention calls; exact real activations from original and holdout fixtures eligible for 64-token unmasked specialization",
        "rows": [],
    }
    configurations = list(args.configs)
    random.Random(2518).shuffle(configurations)
    with FrontierEngine(policy="bf16-exact", max_graphs=2) as engine:
        groups = []
        sources = []
        for suite, fixtures in [
            ("benchmark", [common.workload(1, "short")]),
            ("original", common.validation_requests()),
            ("holdout", requests()),
        ]:
            for index, request in enumerate(fixtures):
                group = capture(engine, request)
                if group is not None:
                    groups.append(group)
                    sources.append({"suite": suite, "index": index, "request": request})
        for source in json.loads(
            Path("results/frontier/attention-special-screen.json").read_text()
        )["requests"][1:]:
            groups.append(capture(engine, source))
            sources.append({"suite": "extra-fully-occupied", "request": source})
        report["eligible_requests"] = sources
        expected = [
            [F.scaled_dot_product_attention(q, k, v) for q, k, v in group]
            for group in groups
        ]
        first = groups[0]

        def baseline():
            return [F.scaled_dot_product_attention(q, k, v) for q, k, v in first]

        report["baseline_ms"], report["baseline_samples_ms"] = timing(
            baseline, repeats=100, rounds=7
        )
        for config in configurations:
            mismatch = []
            max_errors = []
            for group, ref in zip(groups, expected):
                actual = [attention(q, k, v, config) for q, k, v in group]
                mismatch.append(sum(int((a != b).sum()) for a, b in zip(actual, ref)))
                max_errors.append(
                    max(
                        float((a.float() - b.float()).abs().max())
                        for a, b in zip(actual, ref)
                    )
                )

            def candidate(config=config):
                return [attention(q, k, v, config) for q, k, v in first]

            ms, samples = timing(candidate, repeats=100, rounds=7)
            row = {
                "configuration": config,
                "mismatches": sum(mismatch),
                "request_mismatches": mismatch,
                "max_abs_error": max(max_errors),
                "ms": ms,
                "samples_ms": samples,
                "speedup": report["baseline_ms"] / ms,
            }
            report["rows"].append(row)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(row, flush=True)


if __name__ == "__main__":
    main()
