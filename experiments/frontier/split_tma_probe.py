"""Compare exact split-K TMA kernels against the retained ordinary-load kernel."""

import hashlib
import itertools
import json
import random
import statistics
from pathlib import Path

import torch

from experiments.native import common

from .engine import FrontierEngine
from .matmul import matmul
from .split_tma import COMPILED, partial
from .tune import timing

ROOT = Path(__file__).resolve().parents[2]


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    torch.manual_seed(470713)
    path = ROOT / "results/frontier/split-tma.json"
    report = {
        "metadata": common.metadata(),
        "scope": "Isolated BF16 split partials for all 28 distinct MLP output weights. Exact partition boundaries preserved across tile sizes. Downstream fused normalization excluded from both variants.",
        "sources_sha256": {
            str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__), Path(__file__).with_name("split_tma.py")]
        },
        "rows": [],
    }
    rng = random.Random(470713)

    def save():
        path.write_text(json.dumps(report, indent=2) + "\n")

    with FrontierEngine(policy="bf16-splitk-exact", max_graphs=1) as engine:
        config = tuple(engine.selection["mlp.Wo"]["config"])
        report["retained_configuration"] = config
        weights = [
            layer.mlp.Wo.weight for layer in engine.base.model.net.encoder.layers
        ]
        inputs = [
            torch.randn(1, 64, 2624, device="cuda", dtype=torch.bfloat16)
            for _ in weights
        ]

        def baseline():
            return [
                matmul(x, w, config, partial_bf16=True, return_partials=True)
                for x, w in zip(inputs, weights)
            ]

        expected = baseline()
        configs = [
            (bm, bn, bk, warps, stages, both)
            for (bm, bn), bk, warps, stages, both in itertools.product(
                [(16, 32), (16, 64), (32, 32), (32, 64), (64, 32), (64, 64)],
                [64, 128],
                [2, 4],
                [3, 4],
                [False, True],
            )
        ]
        rng.shuffle(configs)
        for candidate in configs:
            row = {"config": candidate}
            try:

                def run(candidate=candidate):
                    return [partial(x, w, candidate) for x, w in zip(inputs, weights)]

                actual = run()
                row["mismatches"] = sum(
                    int((a.view(torch.int16) != b.view(torch.int16)).sum())
                    for a, b in zip(actual, expected)
                )
                row["max_abs_error"] = max(
                    float((a.float() - b.float()).abs().max())
                    for a, b in zip(actual, expected)
                )
                kernel = COMPILED[candidate]
                row["registers"] = kernel.n_regs
                row["shared_bytes"] = kernel.metadata.shared
                row["ptx_sha256"] = hashlib.sha256(
                    kernel.asm["ptx"].encode()
                ).hexdigest()
                row["tma_instruction"] = "cp.async.bulk.tensor.2d" in kernel.asm["ptx"]
                if not row["tma_instruction"]:
                    raise RuntimeError("Expected native tensor loads in generated PTX")
                order = ["candidate", "retained"]
                rng.shuffle(order)
                row["timing_order"] = order
                for name in order:
                    call = run if name == "candidate" else baseline
                    row[name + "_ms"], row[name + "_samples_ms"] = timing(
                        call, repeats=20, rounds=3
                    )
                row["speedup"] = row["retained_ms"] / row["candidate_ms"]
            except Exception as error:  # noqa: BLE001 - keep failed configurations.
                row["error"] = str(error)[:3000]
            report["rows"].append(row)
            save()
            print(json.dumps(row), flush=True)
        eligible = [
            r for r in report["rows"] if r.get("mismatches") == 0 and "error" not in r
        ]
        if eligible:
            best = max(eligible, key=lambda r: r["speedup"])["config"]
            confirmation = {"config": best, "rows": []}
            report["confirmation"] = confirmation
            for round_id in range(9):
                names = ["candidate", "retained"]
                rng.shuffle(names)
                for name in names:
                    call = (
                        (lambda: [partial(x, w, best) for x, w in zip(inputs, weights)])
                        if name == "candidate"
                        else baseline
                    )
                    ms, samples = timing(call, repeats=60, rounds=1)
                    confirmation["rows"].append(
                        {
                            "round": round_id,
                            "variant": name,
                            "ms": ms,
                            "samples_ms": samples,
                        }
                    )
            confirmation["median_ms"] = {
                name: statistics.median(
                    v
                    for r in confirmation["rows"]
                    if r["variant"] == name
                    for v in r["samples_ms"]
                )
                for name in ["candidate", "retained"]
            }
            save()
            print("confirmation", json.dumps(confirmation), flush=True)


if __name__ == "__main__":
    main()
