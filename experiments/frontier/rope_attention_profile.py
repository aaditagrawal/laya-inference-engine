"""Kernel-level timing of the fusion control and candidate."""

import json
from pathlib import Path

import torch

from experiments.native import common

from .attention_special_adapter import load as load_local
from .engine import FrontierEngine
from .global_attention_adapter import load as load_global
from .rope_attention_adapter import load
from .rope_attention_probe import baseline, candidate, capture


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    load()
    load_local()
    load_global()
    report = {
        "build": json.loads(
            Path(".research/frontier-rope-attention/build.json").read_text()
        ),
        "variants": {},
    }
    with FrontierEngine(policy="bf16-exact") as engine:
        records = capture(engine, common.workload(1, "short"))
        for name, call in [
            ("baseline", lambda: [baseline(r) for r in records]),
            ("cutlass64_flash32", lambda: [candidate(r, 1, 3) for r in records]),
        ]:
            for _ in range(3):
                call()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                outputs = call()
            torch.cuda.synchronize()
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as prof:
                for _ in range(50):
                    graph.replay()
                torch.cuda.synchronize()
            rows = [
                {
                    "name": event.key,
                    "count": event.count / 50,
                    "cuda_us": event.device_time_total / 50,
                }
                for event in prof.key_averages()
                if event.device_type == torch.autograd.DeviceType.CUDA
            ]
            report["variants"][name] = rows
            print(name, rows, flush=True)
            assert len(outputs) == 28
    Path("results/frontier/rope_attention-profile.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
