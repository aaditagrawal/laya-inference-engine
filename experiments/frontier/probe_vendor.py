import json
from pathlib import Path

import torch

from experiments.latency.engine import V2Engine
from experiments.native import common

from .mxfp8_vendor import matmul, quantize
from .tune import timing


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    torch.manual_seed(924)
    report = {
        "metadata": common.metadata(),
        "rows": [],
        "scope": "Vendor block-scaled FP8 matrix probe, not accepted model accuracy",
    }
    path = Path("results/frontier/matmul-mxfp8-vendor.json")
    with V2Engine(optimization="native", max_graphs=1) as engine:
        for field in ["attn.Wqkv", "attn.Wo", "mlp.Wi", "mlp.Wo"]:
            group, name = field.split(".")
            weights = [
                getattr(getattr(layer, group), name).weight
                for layer in engine.base.model.net.encoder.layers
            ]
            inputs = [
                torch.randn(1, 64, w.shape[1], device="cuda", dtype=torch.bfloat16)
                for w in weights
            ]
            quantized = [quantize(w) for w in weights]
            row = {"field": field}
            try:

                def baseline(inputs=inputs, weights=weights):
                    return [
                        torch.nn.functional.linear(x, w)
                        for x, w in zip(inputs, weights)
                    ]

                def candidate(inputs=inputs, quantized=quantized):
                    return [
                        matmul(x, w, scale) for x, (w, scale) in zip(inputs, quantized)
                    ]

                candidate()
                row["baseline_ms"], _ = timing(baseline)
                row["candidate_ms"], row["samples_ms"] = timing(candidate)
                row["speedup"] = row["baseline_ms"] / row["candidate_ms"]
            except Exception as error:  # noqa: BLE001 - retain failed experiment evidence
                row["error"] = str(error)
            report["rows"].append(row)
            path.write_text(json.dumps(report, indent=2) + "\n")
            print(row, flush=True)


if __name__ == "__main__":
    main()
