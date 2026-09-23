"""Record actual CUDA kernels executed by one warmed request."""
import argparse
from collections import Counter
import json
from pathlib import Path

import torch

from .engine import BlackwellEngine
from .workloads import workload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("fused", "fp8"), default="fused")
    parser.add_argument("--questions", type=int, default=1)
    parser.add_argument("--length", choices=("short", "medium", "long"), default="short")
    parser.add_argument("--output", default="results/profile")
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    request = workload(args.questions, args.length)
    with BlackwellEngine(backend=args.backend) as engine:
        engine.warmup(**request)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                               torch.profiler.ProfilerActivity.CUDA]) as profile:
            engine.predict(**request)
            torch.cuda.synchronize()
        trace_path = str(output) + ".trace.json"
        profile.export_chrome_trace(trace_path)
        events = json.loads(Path(trace_path).read_text())["traceEvents"]
        count, durations = Counter(), Counter()
        for event in events:
            if event.get("cat") == "kernel":
                count[event["name"]] += 1
                durations[event["name"]] += event.get("dur", 0.)
        result = {"hardware": engine.hardware, "backend": args.backend,
                  "questions": args.questions, "state_length": args.length,
                  "note": "Profiler-instrumented kernel durations in microseconds, not latency benchmark results. Kernel names identify execution paths, not a complete SASS instruction audit.",
                  "kernels": [{"name": name, "count": count[name], "total_us": duration}
                              for name, duration in durations.most_common()]}
        Path(str(output) + ".json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
