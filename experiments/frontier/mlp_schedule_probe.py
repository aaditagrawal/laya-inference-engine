"""Compare direct-weight fused MLP schedules across all 28 checkpoint matrices."""

import hashlib
import itertools
import json
import random
from pathlib import Path

import torch

from experiments.native import common

from .engine import FrontierEngine
from .mlp_geglu import project as retained
from .mlp_schedule import COMPILED, project
from .tune import timing


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    torch.manual_seed(733811)
    path = Path("results/frontier/mlp-schedule.json")
    report = {
        "metadata": common.metadata(),
        "scope": "28 distinct MLPWi matrices and independent random activations; fused BF16 GEMM+exact GEGLU only",
        "source_sha256": {
            name: hashlib.sha256(
                Path(__file__).with_name(name).read_bytes()
            ).hexdigest()
            for name in ("mlp_schedule.py", "mlp_schedule_probe.py")
        },
        "retained_config": [32, 64, 64, 4, 3, 2, False],
        "rows": [],
    }
    geometries = [
        (16, 32, 32),
        (16, 64, 32),
        (16, 64, 64),
        (32, 32, 32),
        (32, 32, 64),
        (32, 64, 32),
        (32, 64, 64),
        (32, 64, 128),
        (32, 128, 64),
        (64, 32, 32),
        (64, 32, 64),
        (64, 64, 32),
        (64, 64, 64),
        (64, 64, 128),
        (64, 128, 64),
    ]
    configs = [
        (*geo, warps, stages, dual, 0)
        for geo, warps, stages, dual in itertools.product(
            geometries, (2, 4), (2, 3, 4), (False, True)
        )
    ]
    random.Random(4190).shuffle(configs)
    with FrontierEngine(policy="bf16-splitk-exact", max_graphs=1) as engine:
        weights = [
            layer.mlp.Wi.weight for layer in engine.base.model.net.encoder.layers
        ]
        inputs = [
            torch.randn(1, 64, 1024, device="cuda", dtype=torch.bfloat16)
            for _ in weights
        ]

        def baseline():
            return [
                retained(x, w, report["retained_config"])
                for x, w in zip(inputs, weights)
            ]

        reference = baseline()

        def measure(config):
            row = {"config": config}
            try:

                def run():
                    return [project(x, w, config) for x, w in zip(inputs, weights)]

                actual = run()
                row["mismatches"] = sum(
                    int((a != b).sum()) for a, b in zip(actual, reference)
                )
                row["max_abs_error"] = max(
                    float((a.float() - b.float()).abs().max())
                    for a, b in zip(actual, reference)
                )
                row["ms"], row["samples_ms"] = timing(run, repeats=16, rounds=3)
                row["baseline_ms"], row["baseline_samples_ms"] = timing(
                    baseline, repeats=16, rounds=3
                )
                row["speedup"] = row["baseline_ms"] / row["ms"]
                kernel = COMPILED[tuple(config)]
                row["registers"], row["shared_bytes"] = (
                    kernel.n_regs,
                    kernel.metadata.shared,
                )
            except Exception as error:  # noqa: BLE001 - preserve compiler/resource failures.
                row["error"] = str(error)[:1800]
            report["rows"].append(row)
            path.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(row), flush=True)

        for config in configs:
            measure(config)
        # Cache controls are adaptive, applied to the fastest exact schedules
        # and the retained geometry, without changing arithmetic.
        top = []
        for dual in (False, True):
            exact = [
                r
                for r in report["rows"]
                if r.get("mismatches") == 0 and r["config"][5] == dual
            ]
            top.extend(r["config"] for r in sorted(exact, key=lambda r: r["ms"])[:3])
        top.append((32, 64, 64, 4, 3, False, 0))
        for config, cache in itertools.product(sorted(set(map(tuple, top))), (1, 2)):
            measure((*config[:-1], cache))


if __name__ == "__main__":
    main()
