"""Lossless weight decoding overlapped with exact GEMM on a second stream."""

import json
import random
from pathlib import Path

import torch
import triton

from experiments.native import common

from .engine import FrontierEngine
from .float13 import _unpack, pack
from .matmul import matmul
from .tma import compiled_matmul
from .tune import timing


class DecodedBank:
    def __init__(self, x, weights, gemm, block, parallel):
        self.x, self.weights, self.gemm = x, weights, gemm
        self.block, self.parallel = block, parallel
        self.n = weights[0].shape[0]
        self.k = weights[0].shape[1] * 32 // 13
        self.buffers = [
            torch.empty((self.n, self.k), dtype=torch.bfloat16, device=x.device)
            for _ in range(2)
        ]
        self.stream = torch.cuda.Stream()
        self.ready = [torch.cuda.Event() for _ in range(2)]
        self.consumed = [torch.cuda.Event() for _ in range(2)]

    def __call__(self):
        main = torch.cuda.current_stream()
        self.stream.wait_stream(main)
        for event in self.consumed:
            event.record(main)
        output = []
        for index, packed in enumerate(self.weights):
            slot = index % 2
            stream = self.stream if self.parallel else main
            stream.wait_event(self.consumed[slot])
            with torch.cuda.stream(stream):
                _unpack[(triton.cdiv(self.n * self.k, self.block),)](
                    packed, self.buffers[slot], self.n, self.k, self.block
                )
                self.ready[slot].record()
            main.wait_event(self.ready[slot])
            output.append(self.gemm(self.x, self.buffers[slot]))
            self.consumed[slot].record(main)
        main.wait_stream(self.stream)
        return output


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    torch.manual_seed(1843)
    report = {
        "metadata": common.metadata(),
        "method": "28 distinct layer weights; exact decoding to two reusable buffers; serial and overlapped decode",
        "rows": [],
    }
    path = Path("results/frontier/matmul-decode-pipeline.json")
    with FrontierEngine(policy="bf16-splitk-exact", max_graphs=1) as engine:
        for field, choice in engine.selection.items():
            group, attr = field.split(".")
            originals = [
                getattr(getattr(layer, group), attr).weight
                for layer in engine.base.model.net.encoder.layers
            ]
            weights = [pack(weight) for weight in originals]
            config = tuple(choice["config"])
            partial_bf16 = choice["variant"] == "bf16-partial"
            is_tma = choice["variant"] == "tma"

            def gemm(x, w, config=config, partial_bf16=partial_bf16, is_tma=is_tma):
                return (
                    compiled_matmul(x, w, [int(v) for v in config])
                    if is_tma
                    else matmul(x, w, config, partial_bf16=partial_bf16)
                )

            x = torch.randn(
                (1, 64, originals[0].shape[1]), device="cuda", dtype=torch.bfloat16
            )
            reference = [gemm(x, w) for w in originals]
            runners = {
                "bf16": lambda originals=originals, x=x, gemm=gemm: [
                    gemm(x, w) for w in originals
                ]
            }
            for block in (256, 1024, 4096):
                for parallel in (False, True):
                    runners[
                        f"decode-{block}-{'parallel' if parallel else 'serial'}"
                    ] = DecodedBank(x, weights, gemm, block, parallel)
            rows = {}
            for label, runner in runners.items():
                actual = runner()
                rows[label] = {
                    "field": field,
                    "variant": label,
                    "mismatches": sum(
                        int((a != b).sum()) for a, b in zip(actual, reference)
                    ),
                    "samples_ms": [],
                }
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    captured = runner()
                for _ in range(3):
                    graph.replay()
                torch.cuda.synchronize()
                rows[label]["graph_mismatches"] = sum(
                    int((a != b).sum()) for a, b in zip(captured, reference)
                )
                graph.reset()
            rng = random.Random(8392)
            for _ in range(5):
                labels = list(runners)
                rng.shuffle(labels)
                for label in labels:
                    _, samples = timing(runners[label], repeats=20, rounds=1)
                    rows[label]["samples_ms"].extend(samples)
            for row in rows.values():
                row["ms"] = common.stats(row["samples_ms"])["p50_ms"]
            baseline = rows["bf16"]["ms"]
            for row in rows.values():
                row["speedup"] = baseline / row["ms"]
                report["rows"].append(row)
                print(json.dumps(row), flush=True)
            path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
