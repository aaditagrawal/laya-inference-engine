"""Screen sparse-escape lossless BF16 decoding on actual 28-layer weights."""

import argparse
import itertools
import json
import random
from pathlib import Path

import torch

from experiments.native import common

from .engine import FrontierEngine
from .float12_escape import check_decode, matmul, pack
from .tune import timing


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(72891)
    report = {
        "metadata": common.metadata(),
        "method": "28 distinct matrices; lossless 12-bit code and masked original-weight escape reads; BF16 partials",
        "rows": [],
        "packing": {},
    }
    path = Path("results/frontier/matmul-float12-escape.json")
    checks = {"metadata": report["metadata"], "matrices": []}
    with FrontierEngine(policy="bf16-splitk-exact", max_graphs=1) as engine:
        for field in ["attn.Wqkv", "attn.Wo", "mlp.Wi", "mlp.Wo"]:
            group, attr = field.split(".")
            modules = [
                getattr(getattr(layer, group), attr)
                for layer in engine.base.model.net.encoder.layers
            ]
            weights = [pack(module.weight) for module in modules]
            if args.check_only:
                checks["matrices"] += [
                    {"layer": i, "field": field, "bit_mismatches": check_decode(w)}
                    for i, w in enumerate(weights)
                ]
                continue
            report["packing"][field] = {
                "escapes": sum(w.escapes for w in weights),
                "elements": sum(w.original.numel() for w in weights),
                "compressed_bytes": sum(w.low.nbytes + w.high.nbytes for w in weights),
                "retained_original_bytes": sum(w.original.nbytes for w in weights),
            }
            x = [
                torch.randn(
                    (1, 64, w.original.shape[1]), device="cuda", dtype=torch.bfloat16
                )
                for w in weights
            ]

            def baseline(x=x, modules=modules):
                return [module(a) for a, module in zip(x, modules)]

            reference = baseline()
            split = 4 if field == "mlp.Wo" else 1
            configs = [
                (bm, bn, 64, split, 4, stages, tma)
                for bm, bn, stages, tma in itertools.product(
                    (32, 64), (32, 64), (2, 3, 4, 5), (False, True)
                )
            ]
            random.Random(5834).shuffle(configs)
            for config in configs:
                row = {"field": field, "config": config}
                try:

                    def run(x=x, weights=weights, config=config):
                        return [matmul(a, w, config) for a, w in zip(x, weights)]

                    actual = run()
                    row["mismatches"] = sum(
                        int((a != b).sum()) for a, b in zip(actual, reference)
                    )
                    row["ms"], row["samples_ms"] = timing(run, repeats=20, rounds=3)
                    row["baseline_ms"], row["baseline_samples_ms"] = timing(
                        baseline, repeats=20, rounds=3
                    )
                    row["speedup"] = row["baseline_ms"] / row["ms"]
                except Exception as error:  # noqa: BLE001 - retain failed configurations.
                    row["error"] = str(error)[:2000]
                report["rows"].append(row)
                path.write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps(row), flush=True)
    if args.check_only:
        checks["all_exact"] = all(r["bit_mismatches"] == 0 for r in checks["matrices"])
        Path("results/frontier/float12-decode-check.json").write_text(
            json.dumps(checks, indent=2) + "\n"
        )
        print(
            "Matrices",
            len(checks["matrices"]),
            "all exact",
            checks["all_exact"],
            flush=True,
        )
        if not checks["all_exact"]:
            raise RuntimeError("GPU decode changed weight bits")


if __name__ == "__main__":
    main()
