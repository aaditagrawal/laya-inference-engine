"""Bounded lossless leading-dimension screen over all distinct encoder weights."""

import hashlib
import json
import random
import statistics
from pathlib import Path

import torch

from experiments.latency.engine import V2Engine
from experiments.native import common

from .matmul import matmul
from .mlp_geglu import project as mlp_project
from .tma import matmul as tma_matmul
from .weight_pitch_kernel import binary_hashes, padded_copy, project

PADDINGS = (0, 8, 16, 32, 64, 128, 256)


def source_hashes():
    root = Path(__file__).parent
    files = sorted(root.glob("weight_pitch*.py")) + [
        root / name for name in ("matmul.py", "tma.py", "mlp_geglu.py")
    ]
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def configurations():
    root = Path("results/frontier")
    rows = []
    for name in (
        "matmul-bf16.json",
        "matmul-tma.json",
        "matmul-pipeline.json",
        "matmul-splitk.json",
    ):
        rows.extend(json.loads((root / name).read_text())["rows"])
    config = {}
    for field in ("attn.Wqkv", "attn.Wo", "mlp.Wo"):
        eligible = [
            r
            for r in rows
            if r["field"] == field
            and r.get("mismatches") == 0
            and r.get("speedup", 0) > 1.02
        ]
        selected = min(eligible, key=lambda r: r["ms"])
        config[field] = selected["config"]
    mlp = json.loads((root / "mlp-geglu-unpacked.json").read_text())["rows"]
    config["mlp.Wi"] = max(
        [r for r in mlp if r.get("mismatches") == 0 and r.get("speedup", 0) > 1.03],
        key=lambda r: r["speedup"],
    )["config"]
    return config


def baseline(x, w, field, config, index):
    if field == "mlp.Wi":
        return mlp_project(x, w, config)
    if field == "attn.Wqkv":
        return tma_matmul(x, w, config)
    return matmul(
        x,
        w,
        config,
        partial_bf16=field == "mlp.Wo",
        return_partials=field == "mlp.Wo" and index < 27,
    )


def candidate(x, w, field, config, index):
    return project(
        x, w, field, config, return_partials=field == "mlp.Wo" and index < 27
    )


def capture(call):
    for _ in range(2):
        call()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = call()
    return graph, outputs


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    torch.manual_seed(42317)
    path = Path("results/frontier/weight_pitch-screen.json")
    before = source_hashes()
    configs = configurations()
    report = {
        "metadata": common.metadata(),
        "source_before": before,
        "configs": configs,
        "scope": "28 distinct matrices per field; independent seeded BF16 inputs, identical retained arithmetic, tile geometry and reduction order. QKV bank includes first-layer weight although token tables remove that request projection.",
        "method": "Exclusive experiment lock; separate CUDA graphs per variant, randomized variant order for 9 rounds, 50 replays per sample. Unused row-padding storage initialized to zero.",
        "fields": {},
    }
    rng = random.Random(84281)
    with V2Engine(optimization="native", max_graphs=1) as engine:
        for field in ("mlp.Wi", "attn.Wqkv", "attn.Wo", "mlp.Wo"):
            group, attr = field.split(".")
            weights = [
                getattr(getattr(layer, group), attr).weight
                for layer in engine.base.model.net.encoder.layers
            ]
            config = configs[field]
            inputs = [
                torch.randn((1, 64, w.shape[1]), device=w.device, dtype=w.dtype)
                for w in weights
            ]
            references = [
                baseline(x, w, field, config, i)
                for i, (x, w) in enumerate(zip(inputs, weights))
            ]
            banks = {
                padding: [padded_copy(w, padding) for w in weights]
                for padding in PADDINGS
            }
            graphs = {}
            rows = {}
            graphs["retained"] = capture(
                lambda inputs=inputs, weights=weights, field=field, config=config: [
                    baseline(x, w, field, config, i)
                    for i, (x, w) in enumerate(zip(inputs, weights))
                ]
            )
            rows["retained"] = {"samples_ms": []}
            for padding, bank in banks.items():
                label = f"pitch+{padding}"
                actual = [
                    candidate(x, w, field, config, i)
                    for i, (x, w) in enumerate(zip(inputs, bank))
                ]
                rows[label] = {
                    "padding_bf16_elements": padding,
                    "mismatches": sum(
                        int((a != b).sum()) for a, b in zip(actual, references)
                    ),
                    "max_abs_error": max(
                        float((a.float() - b.float()).abs().max())
                        for a, b in zip(actual, references)
                    ),
                    "weight_bytes": sum(w.untyped_storage().nbytes() for w in bank),
                    "samples_ms": [],
                }
                graphs[label] = capture(
                    lambda bank=bank, inputs=inputs, field=field, config=config: [
                        candidate(x, w, field, config, i)
                        for i, (x, w) in enumerate(zip(inputs, bank))
                    ]
                )
            binary_before = binary_hashes()
            order = []
            for _ in range(9):
                names = list(graphs)
                rng.shuffle(names)
                order.append(names)
                for name in names:
                    graph = graphs[name][0]
                    for _ in range(3):
                        graph.replay()
                    start, end = (
                        torch.cuda.Event(enable_timing=True),
                        torch.cuda.Event(enable_timing=True),
                    )
                    start.record()
                    for _ in range(50):
                        graph.replay()
                    end.record()
                    end.synchronize()
                    rows[name]["samples_ms"].append(start.elapsed_time(end) / 50)
            for row in rows.values():
                row["median_ms"] = statistics.median(row["samples_ms"])
                row["speedup_vs_retained"] = (
                    statistics.median(rows["retained"]["samples_ms"]) / row["median_ms"]
                )
                row["speedup_vs_pitch0"] = (
                    statistics.median(rows["pitch+0"]["samples_ms"]) / row["median_ms"]
                )
                row["faster_rounds_vs_retained"] = sum(
                    a < b
                    for a, b in zip(row["samples_ms"], rows["retained"]["samples_ms"])
                )
                row["faster_rounds_vs_pitch0"] = sum(
                    a < b
                    for a, b in zip(row["samples_ms"], rows["pitch+0"]["samples_ms"])
                )
            binary_after = binary_hashes()
            assert binary_before == binary_after
            report["fields"][field] = {
                "shape": list(weights[0].shape),
                "matrices": len(weights),
                "rows": rows,
                "order": order,
                "binary_before": binary_before,
                "binary_after": binary_after,
            }
            report["source_after"] = source_hashes()
            assert report["source_after"] == before
            path.write_text(json.dumps(report, indent=2) + "\n")
            print(
                field,
                {
                    name: {k: v for k, v in row.items() if k != "samples_ms"}
                    for name, row in rows.items()
                },
                flush=True,
            )
            del graphs, banks, weights, inputs, references, actual


if __name__ == "__main__":
    main()
