"""Run one comparison profile in an isolated process using shared JSON requests.

The non-Blackwell override is confined to this benchmark. Production hardware
checks remain unchanged. External CPU/GPU/hybrid code is supplied with --source.
"""

import argparse
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import sys
import time

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

PROFILES = {
    "cpu": ("cpu", {"precision": "mlp-bf16", "threads": 32, "max_batch_tokens": 4096}),
    "cpu-pool": ("cpu", {"precision": "mlp-bf16", "threads": 8, "workers": 12}),
    "gpu": ("gpu", {"dtype": "fp16", "graphs": True, "capture_warmups": 1}),
    "gpu-mixed-bf16": ("gpu", {"dtype": "mixed-bf16", "graphs": True, "capture_warmups": 1}),
    "gpu-compiled": ("gpu", {"dtype": "fp16", "graphs": True, "capture_warmups": 1, "compile_model": True}),
    "gpu-compiled-mixed-bf16": ("gpu", {"dtype": "mixed-bf16", "graphs": True, "capture_warmups": 1, "compile_model": True}),
    "hybrid": ("hybrid", {"split": "scorer", "cpu_threads": 4, "gpu_dtype": "fp16", "cpu_dtype": "fp32", "graphs": True}),
    "hybrid-compiled": ("hybrid", {"split": "scorer", "cpu_threads": 1, "gpu_dtype": "fp16", "cpu_dtype": "fp32", "graphs": True, "compile_model": True}),
    "blackwell": ("blackwell", {}),
    "reference": ("reference", {}),
}


def percentile(samples, p):
    values = sorted(samples)
    position = (len(values) - 1) * p
    lo = int(position)
    return values[lo] + (values[min(lo + 1, len(values) - 1)] - values[lo]) * (position - lo)


def hardware():
    info = {"host": platform.node(), "python": platform.python_version()}
    for line in Path("/proc/cpuinfo").read_text().splitlines():
        if line.startswith("model name"):
            info["cpu"] = line.split(":", 1)[1].strip()
            break
    info["gpu_telemetry"] = subprocess.check_output([
        "nvidia-smi", "--query-gpu=name,driver_version,memory.used,utilization.gpu,temperature.gpu",
        "--format=csv,noheader"], text=True).strip()
    return info


def graph_count(engine, mode):
    if mode == "blackwell":
        return len(engine.graphs)
    if mode == "gpu":
        return len(engine.agent.model.graphs)
    if mode == "hybrid":
        return len(engine.agent.model.gpu_stage.graphs)
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--profile", choices=PROFILES, required=True)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--allow-non-blackwell", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.repeats < 2 or args.warmups < 1:
        parser.error("Use at least 2 repeats and 1 warmup")
    sys.path.insert(0, str(args.source.resolve()))
    import torch
    cases = json.loads(args.requests.read_text())
    mode, options = PROFILES[args.profile]
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "profile": args.profile,
        "hardware": hardware(), "options": options, "repeats": args.repeats, "warmups": args.warmups,
        "packages": {p: version(p) for p in ("torch", "transformers", "laya", "tokenizers", "triton")},
        "cuda_runtime": torch.version.cuda,
        "method": "Serial in-process predict. Tokenization, transfers, inference and response formatting included. Model loading, first call, warmup, HTTP and SSH transport excluded equally. Separate process per profile. No answer cache. CPU microbatching and precision follow the recorded profile.",
        "non_blackwell_benchmark_override": False, "cases": [], "status": "running",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    engine = None
    save()
    try:
        started = time.perf_counter()
        if mode == "blackwell":
            import laya_blackwell.engine as module
            if args.allow_non_blackwell:
                def benchmark_hardware(device="cuda:0"):
                    p = torch.cuda.get_device_properties(device)
                    if p.major < 8:
                        raise RuntimeError("Benchmark requires BF16-capable CUDA hardware")
                    return {"name": p.name, "compute_capability": f"{p.major}.{p.minor}",
                            "sm_count": p.multi_processor_count, "vram_bytes": p.total_memory,
                            "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
                            "cuda_architecture": f"sm_{p.major}{p.minor}"}
                module.hardware_info = benchmark_hardware
                report["non_blackwell_benchmark_override"] = True
            engine = module.BlackwellEngine(str(args.model))
        elif mode == "reference":
            from laya import Agent
            torch.set_num_threads(8)
            engine = Agent(str(args.model), device="cpu")
        else:
            from laya_engine import load_backend
            engine = load_backend(mode, str(args.model), **options)
        report["model_load_ms"] = (time.perf_counter() - started) * 1000
        report["torch_threads"] = torch.get_num_threads()
        revision = args.model / "REVISION"
        report["model_revision"] = revision.read_text().strip() if revision.exists() else None
        if mode in {"gpu", "blackwell"} and engine.agent.device.type != "cuda":
            raise RuntimeError("Unexpected CPU fallback")
        if mode == "hybrid":
            split = engine.agent.model
            assert next(split.gpu_stage.parameters()).device.type == "cuda"
            assert next(split.scorer.parameters()).device.type == "cpu"
        for case in cases:
            request = case["request"]
            digest = hashlib.sha256(json.dumps(request, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            assert digest == case["request_sha256"]
            start = time.perf_counter()
            with torch.inference_mode():
                response = engine.predict(**request)
            first_ms = (time.perf_counter() - start) * 1000
            samples = []
            if mode != "reference":
                for _ in range(args.warmups):
                    engine.predict(**request)
                for _ in range(args.repeats):
                    start = time.perf_counter()
                    response = engine.predict(**request)
                    samples.append((time.perf_counter() - start) * 1000)
            count = graph_count(engine, mode)
            if count is not None and count < len(report["cases"]) + 1:
                raise RuntimeError("Timed shape did not create its own CUDA graph")
            row = {"name": case["name"], "questions": case["questions"],
                   "request_sha256": digest, "input_tokens": response["usage"]["input_tokens"],
                   "first_call_ms": first_ms, "graph_count": count,
                   "answers": response["answers"], "samples_ms": samples}
            if samples:
                row.update(p50_ms=percentile(samples, .5), p95_ms=percentile(samples, .95),
                           mean_ms=statistics.mean(samples),
                           decisions_per_second=case["questions"] * 1000 / statistics.mean(samples))
            if mode == "blackwell":
                prepared = engine.prepare(**request)
                slot = engine.graphs[engine._graph_key(prepared)]
                begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                begin.record()
                for _ in range(args.repeats):
                    slot.graph.replay()
                end.record()
                end.synchronize()
                row["graph_device_ms"] = begin.elapsed_time(end) / args.repeats
                row["padded_shape"] = list(engine._shape(prepared))
            report["cases"].append(row)
            print(args.profile, case["name"], row.get("p50_ms", "reference"), flush=True)
            save()
        report["configuration"] = engine.describe() if hasattr(engine, "describe") else getattr(engine, "hardware", {"device": "cpu", "dtype": "fp32"})
        report["status"] = "complete"
    except Exception as exc:
        report["status"] = "failed"
        report["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if engine is not None and hasattr(engine, "close"):
            engine.close()
        report["hardware_after"] = hardware()
        save()


if __name__ == "__main__":
    main()
