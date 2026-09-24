"""Compare attention kernels on actual Laya Q/K/V tensors from a full request."""

import argparse
import json
from pathlib import Path

import torch
from torch.nn import functional as F

from experiments.native import common

from .attention import cudnn_attention, triton_attention
from .engine import FrontierEngine
from .tune import timing


@torch.inference_mode()
def capture_inputs(engine):
    prepared = engine.prepare(**common.workload(1, "short"))
    slot, _ = engine.adapter._slot(prepared)
    tensors = []
    original = F.scaled_dot_product_attention

    def capture(q, k, v, attn_mask=None, **kwargs):
        if q.shape[-2:] == (64, 64) and len(tensors) < 28:
            # Retain actual views: cloning changes QKV strides and memory traffic.
            tensors.append((q, k, v, attn_mask))
        return original(q, k, v, attn_mask=attn_mask, **kwargs)

    F.scaled_dot_product_attention = capture
    try:
        engine.base._forward(slot.inputs)
    finally:
        F.scaled_dot_product_attention = original
    if len(tensors) != 28:
        raise AssertionError(f"Expected 28 encoder layers, captured {len(tensors)}")
    return tensors


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path("results/frontier/attention.json")
    )
    parser.add_argument("--semantics", action="store_true")
    parser.add_argument("--native", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    path = args.output
    report = {
        "metadata": common.metadata(),
        "scope": "CUDA Graph of 28 encoder attention calls, captured from actual model activations",
        "rows": [],
    }
    with FrontierEngine(policy="bf16-exact", max_graphs=1) as engine:
        tensors = capture_inputs(engine)

        def baseline():
            return [
                F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
                for q, k, v, mask in tensors
            ]

        expected = baseline()
        base_ms, samples = timing(baseline)
        report["baseline_ms"], report["baseline_samples_ms"] = base_ms, samples
        configurations = [
            (bm, normal, exp, warps)
            for bm, warps in [(16, 4), (32, 4), (64, 4), (64, 8)]
            for normal in (False, True)
            for exp in (0, 1)
        ]
        if args.semantics:
            configurations = [
                (bm, False, exp, 4) for bm in (16, 32, 64) for exp in (2, 3, 4, 5)
            ]
        if args.native:
            from .native_attention import attention as native_attention

            configurations = ["native32", "native64"]
        for config in ["cudnn", *configurations]:
            row = {"configuration": config}
            try:

                def candidate(config=config):
                    if config == "cudnn":
                        return [
                            cudnn_attention(q, k, v, mask) for q, k, v, mask in tensors
                        ]
                    if config in ("native32", "native64"):
                        return [
                            native_attention(q, k, v, mask, query_tile=int(config[-2:]))
                            if mask is not None and mask.shape[-2] == 64
                            else F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
                            for q, k, v, mask in tensors
                        ]
                    return [
                        triton_attention(q, k, v, mask, config=config)
                        for q, k, v, mask in tensors
                    ]

                actual = candidate()
                row["mismatches"] = sum(
                    int((a != b).sum()) for a, b in zip(actual, expected)
                )
                row["local_mismatches"] = sum(
                    int((a != b).sum())
                    for a, b, t in zip(actual, expected, tensors)
                    if t[-1] is not None and t[-1].shape[-2] == 64
                )
                row["layer_mismatches"] = [
                    int((a != b).sum()) for a, b in zip(actual, expected)
                ]
                row["max_error"] = max(
                    float((a.float() - b.float()).abs().max())
                    for a, b in zip(actual, expected)
                )
                row["ms"], row["samples_ms"] = timing(candidate)
                row["speedup"] = base_ms / row["ms"]
            except Exception as error:  # noqa: BLE001 - record experimental failures
                row["error"] = str(error)
            report["rows"].append(row)
            path.write_text(json.dumps(report, indent=2) + "\n")
            print(row, flush=True)


if __name__ == "__main__":
    main()
