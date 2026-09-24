"""Fresh-process model-only deployment: AOT package, native, or torch.compile."""

import time

ENTRY_TIME = time.perf_counter()

import argparse
import json
import os
import statistics
import traceback
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["aot", "native", "compile"], required=True)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--cache-state", choices=["fresh", "reused"], required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "mode": args.mode,
        "cache_state": args.cache_state,
        "scope": "Fresh Python process, fixed model inputs; excludes tokenization, HTTP, model download, and package creation. Includes imports, registration, weight loading, first inference, and graph capture as separate stages.",
        "package": str(args.package.resolve()),
        "status": "importing",
    }
    report["cache_environment"] = {
        key: os.environ.get(key)
        for key in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR")
    }
    report["cache_files_before"] = {
        key: sum(p.is_file() for p in Path(value).rglob("*"))
        if value and Path(value).exists()
        else 0
        for key, value in report["cache_environment"].items()
    }
    if args.cache_state == "fresh" and (
        not all(report["cache_environment"].values())
        or any(report["cache_files_before"].values())
    ):
        parser.error("Fresh requires explicit empty or absent cache directories")
    engine = None

    def checkpoint():
        report["elapsed_seconds"] = time.perf_counter() - ENTRY_TIME
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    try:
        import torch

        from experiments.native.compiler import (
            pin_libdevice,
            preserve_ops,  # noqa: F401
        )
        from experiments.native.kernels.candidates import build_vector

        report["imports_ms"] = (time.perf_counter() - ENTRY_TIME) * 1000
        torch.set_num_threads(4)
        report["libdevice"] = pin_libdevice()
        report["torch"] = torch.__version__
        setup_start = time.perf_counter()
        build_vector()
        payload = torch.load(args.package.with_suffix(".inputs.pt"), weights_only=True)
        inputs = tuple(x.cuda() for x in payload["inputs"])
        expected = payload["expected"]
        if args.mode == "aot":
            model = torch._inductor.aoti_load_package(
                str(args.package.resolve()), run_single_threaded=True
            )
        else:
            from experiments.native.compiler import (
                install_padded_rope_window,
                install_precise_compile,
            )
            from experiments.native.kernels import install
            from laya_blackwell.engine import BlackwellEngine

            from .build import FixedShapeModel

            engine = BlackwellEngine(max_graphs=1)
            install(engine, "cuda_vector_norm_triton_geglu_corrected")
            install_padded_rope_window(engine)
            if args.mode == "compile":
                install_precise_compile(engine)
            # The fixture stores a fully occupied batch of 64-token inputs.
            unmasked = bool(inputs[1].all().item())
            model = FixedShapeModel(engine.model, unmasked)
        torch.cuda.synchronize()
        report["model_setup_ms"] = (time.perf_counter() - setup_start) * 1000
        report["status"] = "first_inference"
        checkpoint()
        with torch.inference_mode():
            first_start = time.perf_counter()
            actual = model(*inputs)
            torch.cuda.synchronize()
            report["first_inference_ms"] = (time.perf_counter() - first_start) * 1000
            report["entry_to_first_output_ms"] = (
                time.perf_counter() - ENTRY_TIME
            ) * 1000
            actual_cpu = tuple(x.cpu() for x in actual)
            report["exact_outputs"] = [
                torch.equal(a, b) for a, b in zip(actual_cpu, expected)
            ]
            report["max_output_error"] = [
                (a.float() - b.float()).abs().max().item()
                for a, b in zip(actual_cpu, expected)
            ]
            report["status"] = "capturing_graph"
            checkpoint()
            capture_start = time.perf_counter()
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    actual = model(*inputs)
            torch.cuda.current_stream().wait_stream(stream)
            torch.cuda.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                outputs = model(*inputs)
            torch.cuda.synchronize()
            report["graph_setup_ms"] = (time.perf_counter() - capture_start) * 1000
            report["entry_to_graph_ready_ms"] = (
                time.perf_counter() - ENTRY_TIME
            ) * 1000
            samples = []
            for _ in range(10):
                graph.replay()
            torch.cuda.synchronize()
            for _ in range(args.repeats):
                before = time.perf_counter()
                graph.replay()
                tuple(x.float().cpu() for x in outputs)
                samples.append((time.perf_counter() - before) * 1000)
            begin, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            begin.record()
            for _ in range(args.repeats):
                graph.replay()
            end.record()
            end.synchronize()
            report["graph_device_ms"] = begin.elapsed_time(end) / args.repeats
            report["warm_graph_with_outputs_p50_ms"] = statistics.median(samples)
            report["warm_samples_ms"] = samples
            report["status"] = "complete"
    except Exception as error:  # noqa: BLE001 - persist a bounded experiment failure.
        report["failed_stage"] = report["status"]
        report["status"] = "failed"
        report["error"] = str(error)
        report["traceback"] = traceback.format_exc()
        print(report["traceback"], flush=True)
    finally:
        if engine is not None:
            engine.close()
        checkpoint()
        print(
            json.dumps({k: v for k, v in report.items() if k != "warm_samples_ms"}),
            flush=True,
        )


if __name__ == "__main__":
    main()
