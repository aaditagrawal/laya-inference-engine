"""Measure the CUDA graph replay benefit without changing model or buffers."""

import argparse
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import random
import time

import numpy as np

from laya_blackwell.benchmark import load_backend, summarize
from laya_blackwell.engine import BlackwellEngine, hardware_info, model_path, REVISION
from laya_blackwell.workloads import workload


class PythonReplay:
    """Benchmark-only substitute for replay; run_prepared retains its usual lock."""

    def __init__(self, engine, slot):
        self.engine, self.slot = engine, slot

    def replay(self):
        self.slot.outputs = self.engine._forward(self.slot.inputs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--output", default="results/ablation.json")
    args = parser.parse_args()
    if args.iterations < 2 or args.rounds < 2:
        parser.error("Use at least 2 iterations and 2 rounds")
    rng = random.Random(42)
    path = model_path()
    stock = load_backend("upstream", path)
    report = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "hardware": hardware_info(), "revision": REVISION,
        "packages": {p: version(p) for p in ("torch", "triton", "transformers", "laya")},
        "iterations_per_round": args.iterations, "rounds": args.rounds, "seed": 42,
        "method": "Serial in-process predict calls; 5 warmups per block. Randomized mode order within each round. All modes on the same GPU; both models resident. No HTTP, startup or answer cache. Uncaptured mode preserves optimized model, shapes, request preparation and reusable input buffers; it replaces graph replay with the same Python forward.",
        "limitation": "Graph replay difference includes Python dispatch, allocation and kernel launch scheduling. It is not a Blackwell-exclusive gain. Other optimizations are not individually isolated. No cross-GPU hardware attribution.",
        "rows": [], "block_order": [],
    }
    with BlackwellEngine(path) as engine:
        for batch, length in ((1, "short"), (16, "short"), (16, "long")):
            request = workload(batch, length)
            item = engine.prepare(**request)
            captured_logits, captured_act, _ = engine.run_prepared(item)
            slot = engine.graphs[engine._graph_key(item)]
            graph, outputs = slot.graph, slot.outputs
            modes = ("upstream", "fused_without_graph", "fused")
            samples = {mode: [] for mode in modes}
            try:
                slot.graph = PythonReplay(engine, slot)
                logits, act, _ = engine.run_prepared(item)
                np.testing.assert_array_equal(logits, captured_logits)
                np.testing.assert_array_equal(act, captured_act)
                # Check changed input as well, so static output reuse cannot pass.
                changed = engine.prepare("Everything works now. Thank you.", request["questions"])
                if engine._graph_key(changed) == engine._graph_key(item):
                    eager_changed, _, _ = engine.run_prepared(changed)
                    slot.graph, slot.outputs = graph, outputs
                    replay_changed, _, _ = engine.run_prepared(changed)
                    np.testing.assert_array_equal(eager_changed, replay_changed)
                for round_id in range(args.rounds):
                    order = list(modes)
                    rng.shuffle(order)
                    report["block_order"].append({"questions": batch, "length": length,
                                                  "round": round_id, "modes": order})
                    for mode in order:
                        slot.graph, slot.outputs = graph, outputs
                        if mode == "fused_without_graph":
                            slot.graph = PythonReplay(engine, slot)
                        active = stock if mode == "upstream" else engine
                        for _ in range(5):
                            active.predict(**request)
                        for _ in range(args.iterations):
                            start = time.perf_counter()
                            response = active.predict(**request)
                            samples[mode].append((time.perf_counter() - start) * 1000)
                        if active.device.type != "cuda":
                            raise RuntimeError("CPU fallback invalidates the benchmark")
                for mode in modes:
                    row = {"backend": mode, "questions": batch, "state_length": length,
                           "input_tokens": response["usage"]["input_tokens"],
                           **summarize(samples[mode]), "samples_ms": samples[mode]}
                    report["rows"].append(row)
                    print(f"{mode}: q={batch} {length} p50={row['p50_ms']:.3f} ms", flush=True)
            finally:
                slot.graph, slot.outputs = graph, outputs
    report["graph_toggle_outputs_exact"] = True
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
