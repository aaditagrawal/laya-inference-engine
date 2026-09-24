"""Screen projection epilogues against the exact round-one implementation."""

import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from experiments.native.common import metadata
from experiments.native.compiler.padded_rope import rope_qkv_padded
from experiments.native.engine import ExperimentalEngine
from experiments.native.kernels.candidates import provenance, triton_geglu_corrected
from laya_blackwell.kernels import rope_qkv

from .kernels import (
    COMPILED,
    QKV_CONFIGS,
    WI_CONFIGS,
    pack_qkv,
    pack_wi,
    qkv_rope,
    wi_geglu,
)


def timing(fn, repeats=50, rounds=5):
    for _ in range(3):
        _output = fn()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _output = fn()
    samples = []
    for _ in range(rounds):
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        start.record()
        for _ in range(repeats):
            graph.replay()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / repeats)
    return sorted(samples)[len(samples) // 2], samples


def errors(actual, expected):
    if isinstance(actual, tuple):
        pairs = list(zip(actual, expected))
    else:
        pairs = [(actual, expected)]
    return {
        "mismatches": sum(int((a != b).sum()) for a, b in pairs),
        "values": sum(a.numel() for a, _ in pairs),
        "max_abs": max(float((a.float() - b.float()).abs().max()) for a, b in pairs),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--length", type=int, default=64)
    ap.add_argument("--lut", action="store_true")
    ap.add_argument("--shortlist", action="store_true")
    ap.add_argument("--operation", choices=["wi", "qkv", "both"], default="both")
    args = ap.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(1701)
    path = Path(
        f"results/latency-optimizations/fusion/micro-{args.batch}x{args.length}-{args.operation}{'-shortlist' if args.shortlist else ''}{'-lut' if args.lut else ''}.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "metadata": metadata(),
        "input": vars(args),
        "rows": [],
        "method": "Actual checkpoint layer-zero weights, random BF16 normalized input, CUDA Graph replay events, serial candidates with seeded shuffle. Weights can be warm in L2; full-model measurements required.",
    }
    with ExperimentalEngine() as engine, torch.inference_mode():
        model = engine.base.model
        layer = model.net.encoder.layers[0]
        x = torch.randn(
            args.batch, args.length, 1024, device="cuda", dtype=torch.bfloat16
        )
        jobs = []
        if args.operation in {"wi", "both"}:
            linear = layer.mlp.Wi
            packed_w, packed_b = pack_wi(linear.weight, linear.bias)
            reference = triton_geglu_corrected(linear(x))
            jobs.append(
                (
                    "wi",
                    "baseline",
                    None,
                    lambda: triton_geglu_corrected(
                        F.linear(x, linear.weight, linear.bias)
                    ),
                    reference,
                )
            )
            for config in (
                WI_CONFIGS[:1] + WI_CONFIGS[3:5] + WI_CONFIGS[-1:]
                if args.shortlist
                else WI_CONFIGS
            ):
                for packed in [True] if args.shortlist else [False, True]:
                    # The sparse exact-erf epilogue is the first screen. LUT is a follow-up only if competitive.
                    w, bias = (
                        (packed_w, packed_b) if packed else (linear.weight, linear.bias)
                    )
                    fn = lambda w=w, bias=bias, config=config, packed=packed: wi_geglu(
                        x, w, bias, config, packed=packed, lut=args.lut
                    )
                    jobs.append(
                        ("wi", "packed" if packed else "dual", config, fn, reference)
                    )
        if args.operation in {"qkv", "both"}:
            linear_q = layer.attn.Wqkv
            wq, bq = pack_qkv(linear_q.weight, linear_q.bias)
            cos, sin = model.global_cos, model.global_sin
            padding = 64 if args.length > 64 else 0

            def base_qkv():
                projected = F.linear(x, linear_q.weight, linear_q.bias).view(
                    args.batch, args.length, 3, 16, 64
                )
                if padding:
                    return rope_qkv_padded(
                        projected, cos, sin, fp32=True, window=padding
                    )
                return (
                    rope_qkv(projected, cos, sin, fp32=True)
                    .permute(0, 3, 2, 1, 4)
                    .unbind(2)
                )

            reference_q = base_qkv()
            jobs.append(("qkv", "baseline", None, base_qkv, reference_q))
            for config in QKV_CONFIGS[:3] if args.shortlist else QKV_CONFIGS:
                for packed in [False] if args.shortlist else [False, True]:
                    w, bias = (wq, bq) if packed else (linear_q.weight, linear_q.bias)
                    fn = lambda w=w, bias=bias, config=config, packed=packed: qkv_rope(
                        x, w, bias, cos, sin, config, packed=packed, padding=padding
                    )
                    jobs.append(
                        (
                            "qkv",
                            "packed" if packed else "normal",
                            config,
                            fn,
                            reference_q,
                        )
                    )
        random.Random(1409).shuffle(jobs)
        for operation, variant, config, fn, reference in jobs:
            row = {"operation": operation, "variant": variant, "config": config}
            start = time.perf_counter()
            try:
                out = fn()
                torch.cuda.synchronize()
                row["compile_and_first_ms"] = (time.perf_counter() - start) * 1000
                row.update(errors(out, reference))
                row["median_us"], row["samples_us"] = timing(fn)
            except Exception as exc:  # noqa: BLE001 - record experimental compiler failures
                row["error"] = f"{type(exc).__name__}: {exc}"
            report["rows"].append(row)
            report["kernel_provenance"] = provenance()
            report["compiled_kernels"] = {
                str(key): {
                    "registers": obj.n_regs,
                    "spills": obj.n_spills,
                    "shared_bytes": obj.metadata.shared,
                    "mma_instructions": obj.asm["ptx"].count("mma.sync"),
                    "ldmatrix_instructions": obj.asm["ptx"].count("ldmatrix"),
                    "cp_async_instructions": obj.asm["ptx"].count("cp.async"),
                }
                for key, obj in COMPILED.items()
            }
            path.write_text(json.dumps(report, indent=2))
            print(json.dumps(row), flush=True)
    print(path, flush=True)


if __name__ == "__main__":
    main()
