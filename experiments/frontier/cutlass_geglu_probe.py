"""Check and time 28 distinct real MLPWi weights using native CUTLASS."""

import argparse
import fcntl
import json
import random
from pathlib import Path
from time import perf_counter

import torch

from experiments.native import common
from experiments.native.kernels.candidates import gelu_lut, triton_geglu_corrected

from .cutlass_geglu import TILES, Operation, load
from .cutlass_geglu_build import DIRECTORY, ROOT, SOURCE, digest
from .engine import FrontierEngine
from .mlp_geglu import pack, project
from .tune import timing


def exact(actual, expected):
    errors = [
        int((a.view(torch.int16) != b.view(torch.int16)).sum())
        for a, b in zip(actual, expected)
    ]
    return {
        "mismatches": sum(errors),
        "per_matrix_mismatches": errors,
        "max_abs_error": max(
            float((a.float() - b.float()).abs().max()) for a, b in zip(actual, expected)
        ),
    }


@torch.inference_mode()
def run(args):
    torch.set_num_threads(4)
    torch.manual_seed(20260928)
    load()
    table = gelu_lut()
    choices = json.loads(
        (ROOT / "results/frontier/mlp-geglu-unpacked.json").read_text()
    )["rows"]
    config = max(
        [r for r in choices if r.get("mismatches") == 0 and r.get("speedup", 0) > 1.03],
        key=lambda r: r["speedup"],
    )["config"]
    report = {
        "metadata": common.metadata(),
        "native_build": json.loads((DIRECTORY / "build.json").read_text()),
        "scope": "Isolated real-weight bank. Native CUTLASS mainloop plus true GEGLU visitor; no full 5248-column temporary in fused calls.",
        "source_sha256": {
            str(p.relative_to(ROOT)): digest(p)
            for p in [
                SOURCE,
                Path(__file__),
                ROOT / "experiments/frontier/cutlass_geglu.py",
                ROOT / "experiments/frontier/cutlass_geglu_build.py",
            ]
        },
        "seed": 20260928,
        "retained_fused_config": config,
        "rows": [],
        "lut_bytes": table.nbytes,
        "roundings": "FP32 accumulator -> BF16 activation/gate -> exact BF16 GELU LUT -> FP32 multiplication -> BF16 output",
    }
    rounds, repeats = (3, 12) if args.smoke else (5, 30)
    with FrontierEngine(policy="bf16-splitk-exact", max_graphs=1) as engine:
        modules = [layer.mlp.Wi for layer in engine.base.model.net.encoder.layers]
        if args.smoke:
            modules = modules[:1]
        report["matrix_count"] = len(modules)
        report["retained_selection"] = engine.selection
        inputs = [
            torch.randn((1, 64, 1024), device="cuda", dtype=torch.bfloat16)
            for _ in modules
        ]
        torch.cuda.synchronize()
        start = perf_counter()
        weights = [pack(module.weight) for module in modules]
        torch.cuda.synchronize()
        report["packing_setup_wall_ms"] = (perf_counter() - start) * 1000
        report["additional_packed_weight_bytes"] = sum(w.nbytes for w in weights)
        report["packed_weight_bits_roundtrip_exact"] = True

        def raw_baseline():
            return [module(x) for x, module in zip(inputs, modules)]

        def fused_baseline():
            return [
                project(x, module.weight, config) for x, module in zip(inputs, modules)
            ]

        expected_raw = raw_baseline()
        expected = [triton_geglu_corrected(y) for y in expected_raw]
        report["reference_fused_parity"] = exact(fused_baseline(), expected)
        assert report["reference_fused_parity"]["mismatches"] == 0
        report["baseline_raw_ms"], report["baseline_raw_samples_ms"] = timing(
            raw_baseline, repeats, rounds
        )
        report["baseline_fused_ms"], report["baseline_fused_samples_ms"] = timing(
            fused_baseline, repeats, rounds
        )
        print(
            "BASELINE",
            report["baseline_raw_ms"],
            report["baseline_fused_ms"],
            flush=True,
        )
        configs = list(range(len(TILES)))
        random.Random(20260928).shuffle(configs)
        configs.remove(0)
        configs.insert(0, 0)
        for tile in configs:
            row = {"tile": tile, "config": TILES[tile]}
            raw = [Operation(x, w, table, tile, False) for x, w in zip(inputs, weights)]
            fused = [
                Operation(x, w, table, tile, True) for x, w in zip(inputs, weights)
            ]

            def run_raw(raw=raw):
                return [op() for op in raw]

            def run_fused(fused=fused):
                return [op() for op in fused]

            def raw_restored(run_raw=run_raw):
                return [
                    out.view(1, 64, 2624, 2).transpose(2, 3).reshape(1, 64, 5248)
                    for out in run_raw()
                ]

            row["raw_resources"], row["fused_resources"] = (
                raw[0].resources,
                fused[0].resources,
            )
            row["raw_parity"] = exact(raw_restored(), expected_raw)
            row["fused_parity"] = exact(run_fused(), expected)
            if (
                row["raw_parity"]["mismatches"]
                == row["fused_parity"]["mismatches"]
                == 0
            ):
                row["raw_ms"], row["raw_samples_ms"] = timing(run_raw, repeats, rounds)
                row["fused_ms"], row["fused_samples_ms"] = timing(
                    run_fused, repeats, rounds
                )
                row["raw_speedup"] = report["baseline_raw_ms"] / row["raw_ms"]
                row["fused_speedup"] = report["baseline_fused_ms"] / row["fused_ms"]
            report["rows"].append(row)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(row), flush=True)
            if tile == 0 and row["raw_parity"]["mismatches"]:
                report["stopped"] = (
                    "Initial native GEMM arithmetic failed strict parity; no larger sweep performed"
                )
                break
        report["baseline_raw_after_ms"], _ = timing(raw_baseline, repeats, rounds)
        report["baseline_fused_after_ms"], _ = timing(fused_baseline, repeats, rounds)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path("results/frontier/cutlass-geglu.json")
    )
    args = parser.parse_args()
    with Path("/tmp/laya-gpu-experiments.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        run(args)


if __name__ == "__main__":
    main()
