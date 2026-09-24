"""Check exact GEGLU zeros before considering a lossless sparse MLP projection.

This is a workload audit, not a speed measurement. Run under the exclusive
experiment lock. No values are thresholded or pruned.
"""

import hashlib
import json
from pathlib import Path

import torch

from experiments.native import common

from .engine import FrontierEngine
from .holdout import requests


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    output = Path("results/frontier/activation-sparsity.json")
    constructor = json.loads(Path("results/frontier/summary.json").read_text())[
        "recommended_constructor"
    ]
    # Run the same installed arithmetic eagerly so hooks capture real inputs.
    constructor["policy"] = constructor["policy"].removesuffix("-short-compiled")
    fixtures = [common.workload(1, "short"), *common.validation_requests(), *requests()]
    sources = [Path(__file__), Path(__file__).with_name("mlp_geglu.py")]
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    report = {
        "metadata": common.metadata(),
        "scope": "Exact GEGLU output zeros for eligible short requests; no inference timing",
        "constructor": constructor,
        "requests": [],
        "source_sha256": hashes,
    }
    aggregates = [
        {
            "layer": i,
            "values": 0,
            "exact_zeros": 0,
            "four_value_groups": 0,
            "groups_with_at_least_two_zeros": 0,
            "token_rows": 0,
            "all_zero_rows": 0,
            "complete_zero_feature_columns": 0,
            "feature_columns": 0,
            "blocks": {str(bm): {"total": 0, "all_zero": 0} for bm in (16, 32, 64)},
        }
        for i in range(28)
    ]
    captured = {}

    def hook(index):
        def capture(module, args):
            captured[index] = args[0].clone()

        return capture

    with FrontierEngine(**constructor, max_graphs=1) as engine:
        handles = [
            layer.mlp.Wo.register_forward_pre_hook(hook(i))
            for i, layer in enumerate(engine.base.model.net.encoder.layers)
        ]
        try:
            for index, request in enumerate(fixtures):
                prepared = engine.prepare(**request)
                key = engine.base._graph_key(prepared)
                if key[:2] != (1, 64):
                    continue
                captured.clear()
                host = engine.base._allocate(key[:3], host=True)
                engine.base._fill(host, prepared)
                inputs = {k: v.to(engine.base.device) for k, v in host.items()}
                inputs["global_attention_unmasked"] = key[-1]
                engine.base._forward(inputs)
                assert len(captured) == 28, len(captured)
                row = {"fixture": index, "tokens": prepared.input_tokens, "layers": []}
                for i, activation in sorted(captured.items()):
                    assert activation.shape == (1, 64, 2624)
                    bits = activation.view(torch.int16).cpu().numpy().reshape(64, 2624)
                    zero = (bits & 0x7FFF) == 0
                    valid = zero[: prepared.input_tokens]
                    item = aggregates[i]
                    item["values"] += int(valid.size)
                    item["exact_zeros"] += int(valid.sum())
                    groups = valid.reshape(-1, 4)
                    item["four_value_groups"] += len(groups)
                    item["groups_with_at_least_two_zeros"] += int(
                        (groups.sum(axis=1) >= 2).sum()
                    )
                    item["token_rows"] += len(valid)
                    item["all_zero_rows"] += int(valid.all(axis=1).sum())
                    item["feature_columns"] += valid.shape[1]
                    item["complete_zero_feature_columns"] += int(
                        valid.all(axis=0).sum()
                    )
                    blocks = {}
                    for bm in (16, 32, 64):
                        # Include every padded matrix row, matching current GEMM work.
                        tiles = zero.reshape(64 // bm, bm, 41, 64).transpose(0, 2, 1, 3)
                        skipped = int(tiles.all(axis=(2, 3)).sum())
                        count = tiles.shape[0] * tiles.shape[1]
                        item["blocks"][str(bm)]["all_zero"] += skipped
                        item["blocks"][str(bm)]["total"] += count
                        blocks[str(bm)] = skipped
                    row["layers"].append(
                        {
                            "layer": i,
                            "valid_zero_fraction": float(valid.mean()),
                            "all_zero_blocks": blocks,
                        }
                    )
                report["requests"].append(row)
                if len(report["requests"]) % 32 == 0:
                    print("Audited", len(report["requests"]), "requests", flush=True)
        finally:
            for handle in handles:
                handle.remove()
    report["aggregates"] = aggregates
    report["eligible_requests"] = len(report["requests"])
    report["valid_values"] = sum(row["values"] for row in aggregates)
    report["exact_zeros"] = sum(row["exact_zeros"] for row in aggregates)
    report["zero_fraction"] = report["exact_zeros"] / report["valid_values"]
    report["all_zero_blocks"] = {
        str(bm): sum(row["blocks"][str(bm)]["all_zero"] for row in aggregates)
        for bm in (16, 32, 64)
    }
    report["source_stable"] = hashes == {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
    }
    assert report["source_stable"]
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in {"requests", "aggregates"}},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
