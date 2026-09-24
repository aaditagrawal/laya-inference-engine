"""Screen uniform-header lossless tiles over all distinct encoder weights."""

import json
import random
from pathlib import Path

import torch

from experiments.latency.engine import V2Engine
from experiments.native import common

from .tile_lossless import matmul, pack
from .tune import timing


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    torch.manual_seed(934)
    report = {
        "metadata": common.metadata(),
        "rows": [],
        "packing": {},
        "method": "28 distinct checkpoint matrices per captured graph; three timing blocks. Lossless packing verified bitwise.",
    }
    path = Path("results/frontier/matmul-tile-lossless.json")
    with V2Engine(optimization="native", max_graphs=1) as engine:
        for field in ("attn.Wqkv", "attn.Wo", "mlp.Wi", "mlp.Wo"):
            group, attr = field.split(".")
            weights = [
                getattr(getattr(layer, group), attr).weight
                for layer in engine.base.model.net.encoder.layers
            ]
            n, k = weights[0].shape
            inputs = [
                torch.randn(1, 64, k, device="cuda", dtype=torch.bfloat16)
                for _ in weights
            ]

            def baseline(inputs=inputs, weights=weights):
                return [
                    torch.nn.functional.linear(x, w) for x, w in zip(inputs, weights)
                ]

            expected = baseline()
            ms, samples = timing(baseline)
            report["rows"].append(
                {"field": field, "variant": "cublas", "ms": ms, "samples_ms": samples}
            )
            configs = [
                (bm, bn, bk, split, warps, stages)
                for bm, bn, bk, warps in [
                    (32, 32, 64, 4),
                    (32, 64, 64, 4),
                    (64, 32, 64, 4),
                    (64, 64, 64, 4),
                    (64, 64, 128, 4),
                ]
                if n % bn == 0 and k % bk == 0
                for split in (1, 2, 4)
                for stages in (1, 3)
            ]
            random.Random(934).shuffle(configs)
            banks = {}
            for config in configs:
                bn, bk = config[1:3]
                key = (bn, bk)
                if key not in banks:
                    banks[key] = [pack(w, bn, bk) for w in weights]
                    report["packing"][f"{field}-{bn}-{bk}"] = [
                        w.report for w in banks[key]
                    ]
                packed = banks[key]
                row = {"field": field, "variant": "tile-lossless", "config": config}
                try:

                    def candidate(inputs=inputs, packed=packed, config=config):
                        return [matmul(x, w, config) for x, w in zip(inputs, packed)]

                    actual = candidate()
                    row["mismatches"] = sum(
                        int((a != b).sum()) for a, b in zip(actual, expected)
                    )
                    row["max_error"] = max(
                        float((a.float() - b.float()).abs().max())
                        for a, b in zip(actual, expected)
                    )
                    row["ms"], row["samples_ms"] = timing(candidate)
                    row["speedup"] = ms / row["ms"]
                except Exception as error:  # noqa: BLE001 - record failed configurations
                    row["error"] = str(error)
                report["rows"].append(row)
                path.write_text(json.dumps(report, indent=2) + "\n")
                print(row, flush=True)
            del banks


if __name__ == "__main__":
    main()
