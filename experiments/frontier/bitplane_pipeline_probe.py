"""Screen async register-decoded bitplane GEMMs on all 28 real MLPWi matrices."""

import argparse
import hashlib
import itertools
import json
import random
from pathlib import Path

import torch

from experiments.native import common

from .bitplane_pipeline import TILES, Operation, load, pack
from .bitplane_pipeline_build import DIRECTORY, SOURCE
from .compression import Allocation
from .engine import FrontierEngine
from .tune import timing


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path("results/frontier/bitplane-pipeline.json")
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(20260925)
    load()
    report = {
        "metadata": common.metadata(),
        "seed": 20260925,
        "source_sha256": hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        "native_build": json.loads((DIRECTORY / "build.json").read_text()),
        "method": "28 distinct MLPWi weights with independent random BF16 activations; exact zigzag bitplanes packed in MMA fragment lane order; 2/3-stage cp.async pipeline; warp bit transpose decodes directly into MMA registers; normal and hardware-compressible VMM allocations; graph replay; no temporary decode kernel.",
        "rows": [],
        "packing": [],
    }
    with FrontierEngine(policy="bf16-splitk-exact", max_graphs=1) as engine:
        modules = [layer.mlp.Wi for layer in engine.base.model.net.encoder.layers]
        if args.smoke:
            modules = modules[:1]
        report["matrix_count"] = len(modules)
        report["baseline_selection"] = engine.selection
        n, k = modules[0].weight.shape
        inputs = [
            torch.randn(1, 64, k, device="cuda", dtype=torch.bfloat16) for _ in modules
        ]
        expected = [m(x) for m, x in zip(modules, inputs)]

        def baseline():
            return [m(x) for m, x in zip(modules, inputs)]

        base, samples = timing(baseline, repeats=30, rounds=5)
        report["baseline"] = {"ms": base, "samples_ms": samples}
        for bn in [16, 32, 64]:
            packed = [pack(m.weight, bn) for m in modules]
            report["packing"].append({"bn": bn, "all_weights_bit_exact": True})
            owners = []
            operations = []
            try:
                compressed = []
                for p in packed:
                    owner = Allocation(p.shape, dtype=torch.uint32, compressed=True)
                    owners.append(owner)
                    compressed.append(owner.copy(p))
                torch.cuda.synchronize()
                if not all(torch.equal(a, b) for a, b in zip(packed, compressed)):
                    raise RuntimeError("VMM copied bitplanes changed bits")
                report["packing"][-1]["allocations"] = [o.report() for o in owners]
                configs = [
                    (tile, stages, compressed_flag, paired)
                    for tile, stages, compressed_flag, paired in itertools.product(
                        TILES, [2, 3], [False, True], [False, True]
                    )
                    if TILES[tile][1] == bn
                ]
                random.Random(20260925).shuffle(configs)
                for tile, stages, compressed_flag, paired in configs:
                    row = {
                        "tile": tile,
                        "bm": TILES[tile][0],
                        "bn": bn,
                        "stages": stages,
                        "compressed": compressed_flag,
                        "paired_decode": paired,
                    }
                    operations = []
                    try:
                        weights = compressed if compressed_flag else packed
                        operations = [
                            Operation(x, w, n, k, tile, stages, paired)
                            for x, w in zip(inputs, weights)
                        ]

                        def run(operations=operations):
                            return [op() for op in operations]

                        actual = run()
                        row["mismatches"] = sum(
                            int((a.view(torch.int16) != b.view(torch.int16)).sum())
                            for a, b in zip(actual, expected)
                        )
                        row["max_abs_error"] = max(
                            float((a.float() - b.float()).abs().max())
                            for a, b in zip(actual, expected)
                        )
                        row["ms"], row["samples_ms"] = timing(run, repeats=30, rounds=5)
                        row["speedup_retained"] = base / row["ms"]
                    except Exception as error:  # noqa: BLE001 - report failed probes
                        row["error"] = str(error)
                    report["rows"].append(row)
                    args.output.write_text(json.dumps(report, indent=2) + "\n")
                    print(json.dumps(row), flush=True)
            finally:
                torch.cuda.synchronize()
                operations = []
                weights = None
                run = None
                compressed = []
                for owner in owners:
                    owner.close()
            del packed
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
