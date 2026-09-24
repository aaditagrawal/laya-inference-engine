"""Profile the current exact engine before choosing the next optimization."""

import argparse
import collections
import json
from pathlib import Path
from time import perf_counter

import torch

from experiments.latency.engine import V2Engine
from experiments.native import common
from laya_blackwell.protocol import format_response
from laya_blackwell.workloads import workload


def timed(call, repeats=200):
    for _ in range(20):
        call()
    samples = []
    for _ in range(repeats):
        start = perf_counter()
        call()
        samples.append((perf_counter() - start) * 1000)
    return common.stats(samples)


@torch.inference_mode()
def run(
    output,
    policy="native",
    attention=None,
    fuse_reduce_norm=False,
    token_tables=False,
    fuse_mlp_geglu=False,
    mlp_geglu_unpacked=False,
    head_kernels=False,
    host_prepare=None,
    attention_special=False,
    global_attention=False,
    host_runtime=False,
    native_format=False,
):
    if (
        attention_special or global_attention or host_runtime or native_format
    ) and policy == "native":
        raise ValueError("Optional frontier adapters require a FrontierEngine policy")
    torch.set_num_threads(4)
    report = {
        "metadata": common.metadata(),
        "scope": "Profiling, not an optimization comparison",
        "policy": policy,
        "attention": attention,
        "fuse_reduce_norm": fuse_reduce_norm,
        "token_tables": token_tables,
        "fuse_mlp_geglu": fuse_mlp_geglu,
        "mlp_geglu_unpacked": mlp_geglu_unpacked,
        "head_kernels": head_kernels,
        "host_prepare": host_prepare,
        "attention_special": attention_special,
        "global_attention": global_attention,
        "host_runtime": host_runtime,
        "native_format": native_format,
    }
    if policy == "native":
        engine = V2Engine(
            optimization="native", model=common.model_path(), max_graphs=1
        )
    else:
        from .engine import FrontierEngine

        engine = FrontierEngine(
            policy=policy,
            attention=attention,
            fuse_reduce_norm=fuse_reduce_norm,
            token_tables=token_tables,
            fuse_mlp_geglu=fuse_mlp_geglu,
            mlp_geglu_unpacked=mlp_geglu_unpacked,
            head_kernels=head_kernels,
            host_prepare=host_prepare,
            attention_special=attention_special,
            global_attention=global_attention,
            host_runtime=host_runtime,
            native_format=native_format,
            model=common.model_path(),
            max_graphs=1,
        )
    with engine:
        formatter = format_response
        if native_format:
            from .native_format import NativeFormatter

            formatter = NativeFormatter()
        request = workload(1, "short")
        prepared = engine.prepare(**request)
        engine.predict(**request)
        slot, _ = engine.adapter._slot(prepared)
        logits, actions, _ = engine.run_prepared(prepared)
        report["shape"] = engine.base._shape(prepared)
        report["input_tokens"] = prepared.input_tokens
        report["stages"] = {
            "full_predict": timed(lambda: engine.predict(**request)),
            "prepare_only": timed(lambda: engine.prepare(**request)),
            "run_prepared": timed(lambda: engine.run_prepared(prepared)),
            "native_pack_replay_sync": timed(lambda: slot.native.run(prepared.items)),
            "format_only": timed(
                lambda: formatter(
                    prepared,
                    logits,
                    actions,
                    engine.agent.temperature,
                    engine.agent.temperature_by_options,
                )
            ),
        }
        start, end = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        gpu_samples = []
        for _ in range(100):
            start.record()
            slot.graph.replay()
            end.record()
            end.synchronize()
            gpu_samples.append(start.elapsed_time(end))
        report["gpu_graph_with_transfers"] = common.stats(gpu_samples)
        linear = []
        exponent_histogram = torch.zeros(256, device=engine.device, dtype=torch.int64)
        for name, module in engine.base.model.named_modules():
            if isinstance(module, torch.nn.Linear):
                weight = module.weight
                exponents = (
                    (weight.view(torch.int16).to(torch.int32) >> 7) & 255
                ).flatten()
                exponent_histogram += torch.bincount(exponents, minlength=256)
                linear.append(
                    {
                        "name": name,
                        "shape": list(weight.shape),
                        "bytes": weight.numel() * weight.element_size(),
                    }
                )
        report["linear_weights"] = linear
        report["linear_weight_bytes"] = sum(row["bytes"] for row in linear)
        report["all_parameter_bytes"] = sum(
            p.numel() * p.element_size() for p in engine.base.model.parameters()
        )
        report["bf16_weight_exponent_histogram"] = exponent_histogram.cpu().tolist()
        report["theoretical_linear_weight_read_ms_at_896_GBs"] = (
            report["linear_weight_bytes"] / 896e9 * 1000
        )
        # Replay profiling records actual kernels but instrumentation can change timings.
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profiler:
            for _ in range(10):
                slot.graph.replay()
            torch.cuda.synchronize()
        label = policy + ("-attn-" + attention if attention else "")
        if fuse_reduce_norm:
            label += "-reduce-norm"
        if token_tables:
            label += "-token-tables"
        if fuse_mlp_geglu:
            label += "-mlp-geglu"
            if mlp_geglu_unpacked:
                label += "-unpacked"
        if head_kernels:
            label += "-head-kernels"
        if host_prepare:
            label += "-host-" + host_prepare
        if attention_special:
            label += "-attention-special"
        if global_attention:
            label += "-global-attention"
        if host_runtime:
            label += "-host-runtime"
        if native_format:
            label += "-native-format"
        trace_path = Path(f".research/frontier-profile-{label}.trace.json")
        profiler.export_chrome_trace(str(trace_path))
        events = json.loads(trace_path.read_text())["traceEvents"]
        totals = collections.defaultdict(lambda: {"calls": 0, "total_us": 0.0})
        for event in events:
            if event.get("cat") == "kernel":
                row = totals[event["name"]]
                row["calls"] += 1
                row["total_us"] += event.get("dur", 0)
        report["profiled_kernels"] = sorted(
            [{"name": name, **value} for name, value in totals.items()],
            key=lambda row: row["total_us"],
            reverse=True,
        )
        report["profiler_replays"] = 10
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "shape": report["shape"],
                "stages_ms": {
                    key: value["p50_ms"] for key, value in report["stages"].items()
                },
                "gpu_ms": report["gpu_graph_with_transfers"]["p50_ms"],
                "linear_GB": report["linear_weight_bytes"] / 1e9,
                "top_kernels": report["profiled_kernels"][:8],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path("results/frontier/profile.json")
    )
    parser.add_argument("--policy", default="native")
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
    run(
        args.output,
        args.policy,
        args.attention,
        args.fuse_reduce_norm,
        args.token_tables,
        args.fuse_mlp_geglu,
        args.mlp_geglu_unpacked,
        args.head_kernels,
        args.host_prepare,
        args.attention_special,
        args.global_attention,
        args.host_runtime,
        args.native_format,
    )
