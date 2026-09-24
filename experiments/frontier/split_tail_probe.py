"""Exact split-tail arithmetic and nine-round bank timing under exclusive lock."""

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
from .split_tail import COMPILED, binary_hashes, partial
from .tune import timing


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    torch.manual_seed(339090)
    path = Path("results/frontier/split-tail.json")
    sources = [Path(__file__), Path(__file__).with_name("split_tail.py")]
    before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    report = {
        "metadata": common.metadata(),
        "scope": "28 real MLP-output matrices; isolated four-BF16-partial output, no following normalization",
        "source_sha256": before,
        "rows": [],
    }
    with FrontierEngine(policy="bf16-splitk-exact", max_graphs=1) as engine:
        weights = [
            layer.mlp.Wo.weight for layer in engine.base.model.net.encoder.layers
        ]
        inputs = [
            torch.randn(1, 64, 2624, dtype=torch.bfloat16, device="cuda")
            for _ in weights
        ]
        config = engine.selection["mlp.Wo"]["config"]
        report["retained_config"] = config

        def retained():
            return [
                matmul(x, w, config, partial_bf16=True, return_partials=True)
                for x, w in zip(inputs, weights)
            ]

        reference = retained()
        candidates = {"retained": retained}
        for mode, order in itertools.product(range(3), repeat=2):
            label = f"mode-{mode}-order-{order}"

            def run(mode=mode, order=order):
                return [partial(x, w, mode, order) for x, w in zip(inputs, weights)]

            actual = run()
            mismatches = sum(
                int((a.view(torch.int16) != b.view(torch.int16)).sum())
                for a, b in zip(actual, reference)
            )
            row = {
                "label": label,
                "mode": mode,
                "order": order,
                "mismatches": mismatches,
            }
            report["rows"].append(row)
            if mismatches == 0:
                candidates[label] = run
            print(row, flush=True)
        # Check zero/signed-zero and sparse boundary activations independently.
        edge_results = []
        for kind in ("zero", "negative-zero", "sparse", "small"):
            x = torch.zeros_like(inputs[0])
            if kind == "negative-zero":
                x.neg_()
            elif kind == "sparse":
                x[..., [0, 703, 704, 1407, 1408, 2111, 2112, 2623]] = 1
            elif kind == "small":
                x.fill_(2**-100)
            expected = matmul(
                x, weights[0], config, partial_bf16=True, return_partials=True
            )
            for mode, order in itertools.product(range(3), repeat=2):
                actual = partial(x, weights[0], mode, order)
                errors = int(
                    (actual.view(torch.int16) != expected.view(torch.int16)).sum()
                )
                edge_results.append(
                    {
                        "kind": kind,
                        "mode": mode,
                        "order": order,
                        "bit_mismatches": errors,
                    }
                )
                if errors:
                    candidates.pop(f"mode-{mode}-order-{order}", None)
        report["edge_cases"] = edge_results
        report["timings"] = []
        rng = random.Random(419399)
        for round_id in range(9):
            labels = list(candidates)
            rng.shuffle(labels)
            for label in labels:
                ms, samples = timing(candidates[label], repeats=32, rounds=3)
                report["timings"].append(
                    {
                        "round": round_id,
                        "variant": label,
                        "ms": ms,
                        "samples_ms": samples,
                    }
                )
            path.write_text(json.dumps(report, indent=2) + "\n")
            print("Finished round", round_id, flush=True)
        report["summary_ms"] = {
            label: statistics.median(
                s
                for row in report["timings"]
                if row["variant"] == label
                for s in row["samples_ms"]
            )
            for label in candidates
        }
        report["resources"] = {
            str(key): {
                "registers": kernel.n_regs,
                "shared_bytes": kernel.metadata.shared,
            }
            for key, kernel in COMPILED.items()
        }
        report["binary_sha256"] = binary_hashes()
    report["source_stable"] = before == {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
    }
    assert report["source_stable"]
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report["summary_ms"], indent=2))


if __name__ == "__main__":
    main()
