"""Exhaustive GELU proof and paired same-output native epilogue control."""

import ctypes
import fcntl
import json
import random
import statistics
from pathlib import Path

import torch

from experiments.native import common
from experiments.native.kernels.candidates import gelu_lut

from .cutlass_geglu_build import ROOT, digest
from .cutlass_geglu_erf_domain import DIRECTORY
from .cutlass_geglu_probe import exact
from .engine import FrontierEngine
from .mlp_geglu import pack, project


class Operation:
    def __init__(self, library, x, w, table, mode):
        self.library, self.x, self.w, self.table, self.mode = library, x, w, table, mode
        self.output = torch.empty((1, 64, 2624), device=x.device, dtype=x.dtype)
        info = (ctypes.c_int * 5)()
        error = library.cutlass_geglu_erf(None, None, None, None, mode, None, info)
        if error:
            raise RuntimeError(f"Resource query failed: {error}")
        self.resources = dict(
            zip(
                [
                    "registers",
                    "shared_bytes",
                    "threads",
                    "active_blocks_per_sm",
                    "local_bytes",
                ],
                info,
            )
        )

    def __call__(self):
        error = self.library.cutlass_geglu_erf(
            self.x.data_ptr(),
            self.w.data_ptr(),
            self.table.data_ptr(),
            self.output.data_ptr(),
            self.mode,
            torch.cuda.current_stream().cuda_stream,
            None,
        )
        if error:
            raise RuntimeError(f"Native launch failed: {error}")
        return self.output


@torch.inference_mode()
def run():
    torch.set_num_threads(4)
    report = {
        "metadata": common.metadata(),
        "scope": "Best native BM32 BN64 BK64 two-stage tile, matched LUT versus corrected erf; same fused output bytes, 28 weights, no full-model measurement",
        "build": json.loads((DIRECTORY / "build.json").read_text()),
        "source_sha256": {
            str(Path(__file__).relative_to(ROOT)): digest(Path(__file__))
        },
        "parity": [],
        "rounds": [],
    }
    library = ctypes.CDLL(str(DIRECTORY / "cutlass_geglu_erf.so"))
    p, i = ctypes.c_void_p, ctypes.c_int
    library.cutlass_geglu_erf.argtypes = [p, p, p, p, i, p, ctypes.POINTER(i)]
    library.cutlass_geglu_corrected_domain.argtypes = [p, p]
    table = gelu_lut()
    actual = torch.empty(65536, device="cuda", dtype=torch.int16)
    error = library.cutlass_geglu_corrected_domain(
        actual.data_ptr(), torch.cuda.current_stream().cuda_stream
    )
    if error:
        raise RuntimeError(f"Corrected domain launch failed: {error}")
    report["corrected_domain_mismatches"] = int(
        (actual != table.view(torch.int16)).sum()
    )
    if report["corrected_domain_mismatches"]:
        raise RuntimeError("Corrected GELU domain is not bit-exact")
    report["corrected_domain_patterns"] = 65536
    choices = json.loads(
        (ROOT / "results/frontier/mlp-geglu-unpacked.json").read_text()
    )["rows"]
    config = max(
        [r for r in choices if r.get("mismatches") == 0 and r.get("speedup", 0) > 1.03],
        key=lambda r: r["speedup"],
    )["config"]
    report["retained_config"] = config
    with FrontierEngine(policy="bf16-splitk-exact", max_graphs=1) as engine:
        modules = [layer.mlp.Wi for layer in engine.base.model.net.encoder.layers]
        weights = [pack(module.weight) for module in modules]
        inputs = [
            torch.empty((1, 64, 1024), device="cuda", dtype=torch.bfloat16)
            for _ in modules
        ]
        ops = {
            name: [
                Operation(library, x, w, table, mode) for x, w in zip(inputs, weights)
            ]
            for name, mode in [("native_lut", 0), ("native_erf", 1)]
        }
        report["resources"] = {name: items[0].resources for name, items in ops.items()}

        def retained():
            return [
                project(x, module.weight, config) for x, module in zip(inputs, modules)
            ]

        def native_lut():
            return [op() for op in ops["native_lut"]]

        def native_erf():
            return [op() for op in ops["native_erf"]]

        calls = {
            "retained": retained,
            "native_lut": native_lut,
            "native_erf": native_erf,
        }
        for seed in [20260928, 20260929, 20260930]:
            torch.manual_seed(seed)
            for x in inputs:
                x.copy_(torch.randn_like(x))
            expected = retained()
            row = {
                "seed": seed,
                "native_lut": exact(native_lut(), expected),
                "native_erf": exact(native_erf(), expected),
            }
            report["parity"].append(row)
            if row["native_lut"]["mismatches"] or row["native_erf"]["mismatches"]:
                raise RuntimeError(f"Native epilogue mismatch: {row}")
        graphs, outputs = {}, {}
        for name, call in calls.items():
            for _ in range(2):
                call()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                outputs[name] = call()
            graphs[name] = graph
        for round_id in range(9):
            order = list(calls)
            random.Random(20260928 + round_id).shuffle(order)
            row = {"round": round_id, "order": order}
            for name in order:
                graphs[name].replay()
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                start.record()
                for _ in range(40):
                    graphs[name].replay()
                end.record()
                end.synchronize()
                row[name + "_ms"] = start.elapsed_time(end) / 40
            report["rounds"].append(row)
            print(json.dumps(row), flush=True)
        for name in calls:
            report[name + "_median_ms"] = statistics.median(
                r[name + "_ms"] for r in report["rounds"]
            )
        report["native_erf_wins_vs_lut"] = sum(
            r["native_erf_ms"] < r["native_lut_ms"] for r in report["rounds"]
        )
        report["native_erf_wins_vs_retained"] = sum(
            r["native_erf_ms"] < r["retained_ms"] for r in report["rounds"]
        )
        for graph in graphs.values():
            graph.reset()
        del outputs
    destination = ROOT / "results/frontier/cutlass-geglu-erf.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(destination, flush=True)


def main():
    with Path("/tmp/laya-gpu-experiments.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        run()


if __name__ == "__main__":
    main()
