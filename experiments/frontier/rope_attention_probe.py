"""Exact real-activation screen for fused rotary and pinned attention math."""

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from experiments.native import common
from laya_blackwell.kernels import rope_qkv

from .attention_special_adapter import attention as local_attention
from .attention_special_adapter import load as load_local
from .engine import FrontierEngine
from .global_attention_adapter import attention as global_attention
from .global_attention_adapter import load as load_global
from .holdout import requests
from .rope_attention_adapter import attention, load
from .tune import timing


@torch.inference_mode()
def capture(engine, request):
    prepared = engine.prepare(**request)
    key = engine.base._graph_key(prepared)
    if key[:2] != (1, 64):
        return None
    host = engine.base._allocate(key[:3], host=True)
    engine.base._fill(host, prepared)
    inputs = {k: v.to(engine.base.device) for k, v in host.items()}
    inputs["global_attention_unmasked"] = key[-1]
    model = engine.base.model
    namespace = model.forward.__func__.__globals__
    original_rope = namespace["rope_qkv"]
    original_attention = F.scaled_dot_product_attention
    records = []

    def intercept_rope(qkv, cos, sin, **kwargs):
        assert kwargs["fp32"]
        records.append([qkv, cos, sin])
        return original_rope(qkv, cos, sin, **kwargs)

    def intercept_attention(q, k, v, attn_mask=None, **kwargs):
        if records and len(records[-1]) == 3:
            index = len(records) - 1
            records[-1].extend([attn_mask, model.global_layers[index]])
        return original_attention(q, k, v, attn_mask=attn_mask, **kwargs)

    namespace["rope_qkv"] = intercept_rope
    F.scaled_dot_product_attention = intercept_attention
    try:
        engine.base._forward(inputs)
    finally:
        namespace["rope_qkv"] = original_rope
        F.scaled_dot_product_attention = original_attention
    assert len(records) == 28 and all(len(r) == 5 for r in records)
    # Fully occupied local windows need no bias, exactly as retained adapter.
    if key[-1]:
        for record in records:
            record[3] = None
    return records


def baseline(record):
    raw, cos, sin, mask, is_global = record
    q, k, v = rope_qkv(raw, cos, sin, fp32=True).permute(0, 3, 2, 1, 4).unbind(2)
    if is_global and mask is None:
        return global_attention(q, k, v, 8)
    return local_attention(q, k, v, (32, True, True), mask)


def candidate(record, local_config, global_config):
    raw, cos, sin, mask, is_global = record
    config = global_config if is_global and mask is None else local_config
    return attention(raw, cos, sin, config, mask)


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/frontier/rope_attention-screen.json"),
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--local-configs", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--global-configs", type=int, nargs="+", default=[2, 3, 4])
    args = parser.parse_args()
    torch.set_num_threads(4)
    load()
    load_local()
    load_global()
    report = {
        "metadata": common.metadata(),
        "build": json.loads(
            Path(".research/frontier-rope-attention/build.json").read_text()
        ),
        "scope": "28 encoder layers: separate exact FP32 RoPE + retained attention versus shared-staging fusion",
        "sources": [],
        "rows": [],
    }
    groups = []
    with FrontierEngine(policy="bf16-exact", max_graphs=2) as engine:
        suites = [
            ("benchmark", [common.workload(1, "short")]),
            ("original", common.validation_requests()),
            ("holdout", requests()),
        ]
        if args.quick:
            suites = [suites[0], ("holdout", requests()[:2])]
        else:
            extras = json.loads(
                Path("results/frontier/attention-special-screen.json").read_text()
            )["requests"][1:]
            suites.append(("extra-fully-occupied", extras))
        for suite, fixtures in suites:
            for index, request in enumerate(fixtures):
                group = capture(engine, request)
                if group is None:
                    continue
                groups.append(group)
                report["sources"].append(
                    {"suite": suite, "index": index, "request": request}
                )
        print(
            f"Captured {len(groups)} requests / {len(groups) * 28} layers", flush=True
        )
        expected = [[baseline(record) for record in group] for group in groups]
        for lc in args.local_configs:
            for gc in args.global_configs:
                mismatches, max_error = [], 0.0
                for group, references in zip(groups, expected):
                    count = 0
                    for record, reference in zip(group, references):
                        actual = candidate(record, lc, gc)
                        count += int((actual != reference).sum())
                        max_error = max(
                            max_error,
                            float((actual.float() - reference.float()).abs().max()),
                        )
                    mismatches.append(count)
                row = {
                    "local_config": lc,
                    "global_config": gc,
                    "mismatches": sum(mismatches),
                    "request_mismatches": mismatches,
                    "max_abs_error": max_error,
                    "timings": {},
                }
                padded = next(
                    group
                    for group, source in zip(groups, report["sources"])
                    if source["suite"] == "holdout"
                )
                for name, group in [("full", groups[0]), ("padded", padded)]:
                    base_ms, base_samples = timing(
                        lambda group=group: [baseline(record) for record in group],
                        repeats=100,
                        rounds=7,
                    )
                    new_ms, new_samples = timing(
                        lambda group=group, lc=lc, gc=gc: [
                            candidate(record, lc, gc) for record in group
                        ],
                        repeats=100,
                        rounds=7,
                    )
                    row["timings"][name] = {
                        "baseline_ms": base_ms,
                        "fused_ms": new_ms,
                        "speedup": base_ms / new_ms,
                        "baseline_samples_ms": base_samples,
                        "fused_samples_ms": new_samples,
                    }
                report["rows"].append(row)
                report["python_sha256"] = {
                    path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                    for path in [
                        Path(__file__),
                        Path(__file__).with_name("rope_attention_adapter.py"),
                    ]
                }
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print(row, flush=True)


if __name__ == "__main__":
    main()
