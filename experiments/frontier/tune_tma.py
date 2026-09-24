"""Screen descriptor loads, pipeline depth and warp specialization."""

import json
import random
from pathlib import Path

import torch

from experiments.latency.engine import V2Engine
from experiments.native import common

from .tma import COMPILED, matmul
from .tune import timing


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    torch.manual_seed(924)
    path = Path("results/frontier/matmul-tma.json")
    report = {"metadata": common.metadata(), "rows": []}
    with V2Engine(optimization="native", max_graphs=1) as engine:
        for field in ["attn.Wqkv", "attn.Wo", "mlp.Wi", "mlp.Wo"]:
            group, attr = field.split(".")
            weights = [
                getattr(getattr(layer, group), attr).weight
                for layer in engine.base.model.net.encoder.layers
            ]
            inputs = [
                torch.randn(1, 64, w.shape[1], device="cuda", dtype=torch.bfloat16)
                for w in weights
            ]
            expected = [
                torch.nn.functional.linear(x, w) for x, w in zip(inputs, weights)
            ]
            base, samples = timing(
                lambda inputs=inputs, weights=weights: [
                    torch.nn.functional.linear(x, w) for x, w in zip(inputs, weights)
                ]
            )
            report["rows"].append(
                {"field": field, "variant": "cublas", "ms": base, "samples_ms": samples}
            )
            configs = [
                (bm, bn, bk, split, warps, stages, ws)
                for bm, bn, bk, warps in [
                    (32, 32, 64, 4),
                    (32, 64, 64, 4),
                    (64, 32, 64, 4),
                    (64, 64, 64, 4),
                    (32, 64, 128, 4),
                    (64, 64, 128, 4),
                    (64, 128, 64, 8),
                ]
                for split in [1, 2, 4]
                for stages in [2, 4]
                for ws in [False, True]
                if (bm + bn) * bk * 2 * stages <= 98304
            ]
            random.Random(924).shuffle(configs)
            for config in configs:
                row = {"field": field, "config": config, "variant": "tma"}
                try:

                    def candidate(config=config, inputs=inputs, weights=weights):
                        return [matmul(x, w, config) for x, w in zip(inputs, weights)]

                    actual = candidate()
                    row["mismatches"] = sum(
                        int((a != b).sum()) for a, b in zip(actual, expected)
                    )
                    row["max_abs_error"] = max(
                        float((a.float() - b.float()).abs().max())
                        for a, b in zip(actual, expected)
                    )
                    row["ms"], row["samples_ms"] = timing(candidate)
                    row["speedup"] = base / row["ms"]
                    kernel = COMPILED[(64, *weights[0].shape, *config)]
                    row["tma_instruction"] = next(
                        (
                            line.strip()
                            for line in kernel.asm["ptx"].splitlines()
                            if "cp.async.bulk.tensor" in line
                        ),
                        None,
                    )
                    row["shared_bytes"] = kernel.metadata.shared
                except Exception as error:  # noqa: BLE001 - preserve failed configurations
                    row["error"] = str(error)
                report["rows"].append(row)
                path.write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
