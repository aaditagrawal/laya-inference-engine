"""Measure a fresh process through its first answer, then warm request latency.

Run under the shared experiment lock. Model files and native extensions must
already exist. Set separate TORCHINDUCTOR_CACHE_DIR and TRITON_CACHE_DIR paths
for a fresh compiler-cache measurement; this command never deletes caches.

Invoke with ``uv run --no-sync python -m experiments.native.startup``.
"""

import time

ENTRY_TIME = time.perf_counter()

import argparse
import json
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["native-window", "compiled"], required=True)
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--cache-state", choices=["fresh", "reused"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=50)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    cache_paths = {
        key: os.environ.get(key)
        for key in ["TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"]
    }
    cache_files_before = {
        key: sum(1 for p in Path(value).rglob("*") if p.is_file())
        if value and Path(value).exists()
        else 0
        for key, value in cache_paths.items()
    }
    if args.cache_state == "fresh" and (
        not all(cache_paths.values()) or any(cache_files_before.values())
    ):
        parser.error("Fresh requires both explicit cache paths to be empty or absent")
    script_directory = Path(__file__).resolve().parent
    # Direct script execution must not shadow Hugging Face's installed kernels
    # package with our experiments/native/kernels directory.
    sys.path[:] = [
        entry for entry in sys.path if Path(entry).resolve() != script_directory
    ]
    sys.path.insert(0, str(script_directory.parents[1]))
    import numpy as np
    import torch

    from experiments.native.compiler import pin_libdevice
    from experiments.native.engine import ExperimentalEngine
    from laya_blackwell.workloads import workload

    imports_ms = (time.perf_counter() - ENTRY_TIME) * 1000
    torch.set_num_threads(4)
    library = pin_libdevice()
    load_start = time.perf_counter()
    engine = ExperimentalEngine(mode=args.mode, kernel=args.kernel)
    load_ms = (time.perf_counter() - load_start) * 1000
    try:
        request = workload(1, "short")
        first_start = time.perf_counter()
        response = engine.predict(**request)
        first_ms = (time.perf_counter() - first_start) * 1000
        total_ms = (time.perf_counter() - ENTRY_TIME) * 1000
        for _ in range(10):
            engine.predict(**request)
        samples = []
        for _ in range(args.repeats):
            started = time.perf_counter()
            engine.predict(**request)
            samples.append((time.perf_counter() - started) * 1000)
        report = {
            "mode": args.mode,
            "kernel": args.kernel,
            "cache_state": args.cache_state,
            "scope": "Fresh Python process; existing model files and prebuilt native extensions. No model download, HTTP or SSH. Cache state refers only to the explicit compiler directories.",
            "imports_ms": imports_ms,
            "model_and_adapter_setup_ms": load_ms,
            "first_predict_ms": first_ms,
            "python_entry_to_first_response_ms": total_ms,
            "warm_p50_ms": float(np.median(samples)),
            "samples_ms": samples,
            "response": response,
            "libdevice": library,
            "cache_environment": cache_paths,
            "cache_files_before": cache_files_before,
            "hardware": engine.hardware,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                {
                    key: value
                    for key, value in report.items()
                    if key not in ["samples_ms", "response"]
                }
            ),
            flush=True,
        )
    finally:
        engine.close()


if __name__ == "__main__":
    main()
