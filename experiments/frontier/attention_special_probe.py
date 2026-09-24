"""Screen fixed-shape local attention on actual and changing model activations."""

import itertools
import json
import random
from pathlib import Path

import torch
from torch.nn import functional as F

from experiments.native import common

from .attention_special_adapter import attention, load
from .engine import FrontierEngine
from .holdout import requests
from .native_attention import attention as retained_attention
from .tune import timing


@torch.inference_mode()
def capture(engine, request, *, allow_padding=False):
    prepared = engine.prepare(**request)
    key = engine.base._graph_key(prepared)
    host = engine.base._allocate(key[:3], host=True)
    engine.base._fill(host, prepared)
    inputs = {k: v.to(engine.base.device) for k, v in host.items()}
    inputs["global_attention_unmasked"] = key[-1]
    tensors = []
    original = F.scaled_dot_product_attention

    def intercept(q, k, v, attn_mask=None, **kwargs):
        if (
            q.shape == (1, 16, 64, 64)
            and attn_mask is not None
            and attn_mask.shape[-2] == 64
        ):
            if not allow_padding and not bool(attn_mask.all()):
                raise RuntimeError("Probe expected fully occupied local attention")
            tensors.append((q, k, v, attn_mask))
        return original(q, k, v, attn_mask=attn_mask, **kwargs)

    F.scaled_dot_product_attention = intercept
    try:
        engine.base._forward(inputs)
    finally:
        F.scaled_dot_product_attention = original
    if len(tensors) != 18:
        raise AssertionError(f"Expected 18 local attention calls, got {len(tensors)}")
    return tensors


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    load()
    fixtures = [common.workload(1, "short"), *requests()[::16]]
    report = {
        "metadata": common.metadata(),
        "scope": "18 local attention calls per graph; 9 actual requests for elementwise parity",
        "build": json.loads(
            Path(".research/frontier-attention-special/build.json").read_text()
        ),
        "rows": [],
    }
    configurations = list(itertools.product((32, 64), (False, True), (False, True)))
    random.Random(925).shuffle(configurations)
    with FrontierEngine(policy="bf16-exact", max_graphs=2) as engine:
        for fixture in fixtures[1:]:
            for _ in range(64):
                key = engine.base._graph_key(engine.prepare(**fixture))
                if key == (1, 64, 4, True):
                    break
                if key[1] > 64:
                    raise RuntimeError("Could not make a 64-token changing request")
                fixture["state"] += " a"
            else:
                raise RuntimeError("Could not fill local attention fixture")
        report["requests"] = fixtures
        groups = [capture(engine, request) for request in fixtures]
        refs = [
            [retained_attention(q, k, v, mask) for q, k, v, mask in group]
            for group in groups
        ]
        group = groups[0]

        def baseline():
            return [retained_attention(q, k, v, mask) for q, k, v, mask in group]

        report["baseline_ms"], report["baseline_samples_ms"] = timing(
            baseline, repeats=100, rounds=7
        )
        for config in configurations:
            row = {"configuration": config, "request_mismatches": []}
            for tensors, expected in zip(groups, refs):
                outputs = [attention(q, k, v, config) for q, k, v, _ in tensors]
                row["request_mismatches"].append(
                    sum(int((a != b).sum()) for a, b in zip(outputs, expected))
                )
            row["mismatches"] = sum(row["request_mismatches"])

            def candidate(config=config):
                return [attention(q, k, v, config) for q, k, v, _ in group]

            row["ms"], row["samples_ms"] = timing(candidate, repeats=100, rounds=7)
            row["speedup"] = report["baseline_ms"] / row["ms"]
            report["rows"].append(row)
            print(row, flush=True)
            Path("results/frontier/attention-special-screen.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )


if __name__ == "__main__":
    main()
