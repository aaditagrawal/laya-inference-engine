"""Reproducible serial latency benchmark against the unmodified Laya SDK."""

import argparse
import gc
import json
import subprocess
import time
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
from laya import Agent

from .engine import MODEL_ID, REVISION, BlackwellEngine, hardware_info, model_path
from .workloads import workload

BACKENDS = (
    "upstream",
    "upstream-fast",
    "upstream-compile",
    "eager",
    "fused",
    "fp8",
    "fast",
)


def load_backend(backend, path, device="cuda:0"):
    """Keep upstream defaults explicit and reject unavailable fast-path fallbacks."""
    if backend not in BACKENDS:
        raise ValueError(f"Unknown backend: {backend}")
    if backend == "fast":
        from .fast import FastEngine

        if Path(path).resolve() != Path(model_path()).resolve():
            raise ValueError("Fast mode benchmarks require the pinned Laya checkpoint")
        return FastEngine(device=device)
    if backend.startswith("upstream"):
        agent = Agent(path, device=device, compile=backend == "upstream-compile")
        if agent.device.type != "cuda":
            raise RuntimeError("Upstream fell back to CPU; invalid GPU baseline")
        if backend == "upstream-fast":
            agent.accelerate(strict=True)
        return agent
    return BlackwellEngine(path, device=device, backend=backend)


def summarize(samples):
    return {
        "mean_ms": float(np.mean(samples)),
        "p50_ms": float(np.percentile(samples, 50)),
        "p95_ms": float(np.percentile(samples, 95)),
        "min_ms": float(min(samples)),
    }


def run(iterations=50, warmup=5, backends=("upstream", "fused"), cases=None, threads=4):
    if iterations < 2 or warmup < 1:
        raise ValueError("Use at least 2 iterations and 1 warmup")
    if not backends or any(b not in BACKENDS for b in backends):
        raise ValueError(f"Choose backends from {BACKENDS}")
    if threads < 1:
        raise ValueError("threads must be positive")
    torch.set_num_threads(threads)
    cases = cases or [
        (1, "short"),
        (4, "short"),
        (16, "short"),
        (1, "medium"),
        (4, "medium"),
        (1, "long"),
        (16, "long"),
    ]
    device = "cuda:0"
    report = {
        "created_utc": datetime.now(UTC).isoformat(),
        "hardware": hardware_info(device),
        "packages": {
            p: version(p)
            for p in (
                "torch",
                "triton",
                "transformers",
                "laya",
                "huggingface-hub",
                "tokenizers",
            )
        },
        "model": MODEL_ID,
        "revision": REVISION,
        "iterations": iterations,
        "warmup": warmup,
        "torch_threads": torch.get_num_threads(),
        "method": "Serial in-process requests; wall time includes tokenization, transfers, inference and formatting. No HTTP. No answer cache. Cold shape setup excluded from warm samples and reported separately.",
        "first_request_note": "CUDA is already initialized; local model and Triton caches may exist. Backends run sequentially. first_request_ms is first use of this engine/shape, not a clean-machine cold start.",
        "timing_contract": {
            "both_backends_same_device": True,
            "both_include_tokenization_transfers_forward_formatting": True,
            "both_exclude_model_load_download_and_warmup": True,
            "http_included": False,
            "upstream": "Agent defaults: fast=False, compile=False",
            "upstream-fast": "Agent.accelerate(strict=True), TileLang and CUDA graphs",
            "upstream-compile": "Agent(compile=True), torch.compile defaults",
        },
        "rows": [],
    }
    if "upstream-fast" in backends:
        report["packages"]["tilelang"] = version("tilelang")
    report["nvidia_smi"] = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,pstate,temperature.gpu,memory.used,utilization.gpu",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    path = model_path()
    for backend in backends:
        start = time.perf_counter()
        engine = load_backend(backend, path, device)
        torch.cuda.synchronize(device)
        load_ms = (time.perf_counter() - start) * 1000
        for batch, length in cases:
            request = workload(batch, length)
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            response = engine.predict(**request)
            if backend.startswith("upstream") and engine.device.type != "cuda":
                raise RuntimeError(
                    "Upstream fell back to CPU; this is not a valid GPU baseline"
                )
            first_request_ms = (time.perf_counter() - start) * 1000
            for _ in range(warmup):
                engine.predict(**request)
            samples = []
            for _ in range(iterations):
                # predict copies the result to CPU, so each sample completes GPU work.
                start = time.perf_counter()
                response = engine.predict(**request)
                samples.append((time.perf_counter() - start) * 1000)
            if backend.startswith("upstream") and engine.device.type != "cuda":
                raise RuntimeError("Upstream fell back to CPU during measurement")
            metrics = summarize(samples)
            metrics.update(
                {
                    "backend": backend,
                    "questions": batch,
                    "state_length": length,
                    "input_tokens": response["usage"]["input_tokens"],
                    "decisions_per_second": batch * 1000 / metrics["mean_ms"],
                    "input_tokens_per_second": response["usage"]["input_tokens"]
                    * 1000
                    / metrics["mean_ms"],
                    "first_request_ms": first_request_ms,
                    "load_ms": load_ms,
                    "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                    "samples_ms": samples,
                }
            )
            if backend in {"fused", "fp8"}:
                prepared = engine.prepare(**request)
                slot = engine.graphs[engine._graph_key(prepared)]
                # Separate device-only graph timing, with no tokenization or transfer.
                begin, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                begin.record()
                for _ in range(iterations):
                    slot.graph.replay()
                end.record()
                end.synchronize()
                metrics["graph_device_ms"] = begin.elapsed_time(end) / iterations
                metrics["shape"] = response["engine"]["shape"]
                del slot
            report["rows"].append(metrics)
            print(
                f"{backend:12} q={batch:2} {length:6} p50={metrics['p50_ms']:.3f}ms p95={metrics['p95_ms']:.3f}ms decisions/s={metrics['decisions_per_second']:.1f}",
                flush=True,
            )
        if not backend.startswith("upstream"):
            engine.close()
        del engine
        gc.collect()
        torch.cuda.empty_cache()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument(
        "--backends", nargs="+", choices=BACKENDS, default=["upstream", "fused"]
    )
    parser.add_argument("--output", default="results/benchmark.json")
    args = parser.parse_args()
    report = run(args.iterations, args.warmup, args.backends, threads=args.threads)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(path)


if __name__ == "__main__":
    main()
