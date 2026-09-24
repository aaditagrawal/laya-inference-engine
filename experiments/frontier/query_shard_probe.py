"""Exact real-QKV gate and isolated timing for fixed-32Q query sharding.

Run under the exclusive experiment lock. Masks are converted to additive BF16
bias before capture. Measurements contain 18 local or 10 padded-global calls.
"""

import hashlib
import json
import random
import statistics
from pathlib import Path

import torch

from experiments.native import common

from .attention_special_adapter import load as load_special
from .attention_special_probe import capture as capture_local
from .engine import FrontierEngine
from .global_attention_padding import capture as capture_global
from .holdout import requests
from .query_shard import attention, load
from .tune import timing


def special(q, k, v, tile, bias):
    return torch.ops.laya_frontier_attention_special.forward(
        q, k, v, tile, True, True, bias
    )


def bit_mismatches(actual, expected):
    return int((actual.view(torch.int16) != expected.view(torch.int16)).sum())


def additive(group, unmasked=False):
    return [
        (
            q,
            k,
            v,
            None
            if unmasked
            else torch.zeros_like(mask, dtype=q.dtype).masked_fill_(
                ~mask, float("-inf")
            ),
        )
        for q, k, v, mask in group
    ]


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    torch.manual_seed(539083)
    load()
    load_special()
    report = {
        "metadata": common.metadata(),
        "scope": "Same 32Q x 64K template with 32/16/8 real query rows per block. No changed K reduction or softmax math.",
        "build": json.loads(
            Path(".research/frontier-query-shard/build.json").read_text()
        ),
        "variants": {
            "native-special-32": {"tile_queries": 32, "grid_ctas": 32},
            "native-special-64": {"tile_queries": 64, "grid_ctas": 16},
            "shard-32": {"tile_queries": 32, "real_queries": 32, "grid_ctas": 32},
            "shard-16": {"tile_queries": 32, "real_queries": 16, "grid_ctas": 64},
            "shard-8": {"tile_queries": 32, "real_queries": 8, "grid_ctas": 128},
        },
    }
    functions = {
        "native-special-32": lambda q, k, v, bias: special(q, k, v, 32, bias),
        "native-special-64": lambda q, k, v, bias: special(q, k, v, 64, bias),
        "shard-32": lambda q, k, v, bias: attention(q, k, v, 32, bias),
        "shard-16": lambda q, k, v, bias: attention(q, k, v, 16, bias),
        "shard-8": lambda q, k, v, bias: attention(q, k, v, 8, bias),
    }
    with FrontierEngine(policy="bf16-exact", max_graphs=2) as engine:
        fixtures = [common.workload(1, "short"), *requests()[::16]]
        groups = []
        for index, fixture in enumerate(fixtures):
            key = engine.base._graph_key(engine.prepare(**fixture))
            if key[:2] != (1, 64):
                raise RuntimeError("Fixture escaped 64-token geometry")
            group = capture_local(engine, fixture, allow_padding=True)
            groups.append(
                {
                    "source": index,
                    "case": "local-full" if key[-1] else "local-padded",
                    "tensors": additive(group, unmasked=key[-1]),
                }
            )
            global_group = capture_global(engine, fixture)
            if global_group is not None:
                groups.append(
                    {
                        "source": index,
                        "case": "global-padded",
                        "tensors": additive(global_group),
                    }
                )
        report["requests"] = fixtures
        gates = []
        for group in groups:
            expected = [
                functions["native-special-32"](*values) for values in group["tensors"]
            ]
            for name, function in functions.items():
                actual = [function(*values) for values in group["tensors"]]
                mismatch = [bit_mismatches(a, b) for a, b in zip(actual, expected)]
                gates.append(
                    {
                        "source": group["source"],
                        "case": group["case"],
                        "variant": name,
                        "calls": len(mismatch),
                        "mismatches": sum(mismatch),
                        "per_call": mismatch,
                    }
                )
        report["real_qkv_gate"] = gates
        print("Real QKV parity complete", flush=True)

        # Distinct head/row bias values and strides expose duplicate/omitted offsets.
        q, k, v, _ = groups[0]["tensors"][0]
        shape = (1, 16, 64, 64)
        dense = torch.randn(shape, device=q.device, dtype=q.dtype)
        mask = torch.rand(shape, device=q.device) > 0.2
        dense.masked_fill_(~mask, float("-inf"))
        dense[:, :, 3, :] = float("-inf")
        head_broadcast = dense[:, :, :1, :].contiguous()
        strided = torch.empty((1, 16, 128, 64), device=q.device, dtype=q.dtype)[
            :, :, ::2, :
        ]
        strided.copy_(dense)
        biases = {
            "dense-head-row": dense,
            "head-broadcast": head_broadcast,
            "strided-row": strided,
            "shared-head-row": dense[:, :1].contiguous(),
            "shared-head-broadcast": head_broadcast[:, :1].contiguous(),
        }
        adversarial = []
        for layout in ("original", "contiguous", "sequence-major"):
            tensors = (q, k, v)
            if layout == "contiguous":
                tensors = tuple(value.contiguous() for value in tensors)
            elif layout == "sequence-major":
                tensors = tuple(
                    value.transpose(1, 2).contiguous().transpose(1, 2)
                    for value in tensors
                )
            for bias_name, bias in biases.items():
                expected = functions["native-special-32"](*tensors, bias)
                for name, function in functions.items():
                    actual = function(*tensors, bias)
                    adversarial.append(
                        {
                            "layout": layout,
                            "bias": bias_name,
                            "variant": name,
                            "mismatches": bit_mismatches(actual, expected),
                        }
                    )
        report["offset_stride_gate"] = adversarial
        eligible = [
            name
            for name in functions
            if all(
                row["mismatches"] == 0
                for row in gates + adversarial
                if row["variant"] == name
            )
        ]
        report["exact_variants"] = eligible
        print(f"Exact variants: {eligible}", flush=True)

        rows, order = [], []
        rng = random.Random(87307)
        for case in ("local-full", "local-padded", "global-padded"):
            bank = next(group["tensors"] for group in groups if group["case"] == case)
            for round_id in range(9):
                names = list(eligible)
                rng.shuffle(names)
                order.append({"case": case, "round": round_id, "variants": names})
                for name in names:

                    def call(name=name, bank=bank):
                        return [functions[name](*values) for values in bank]

                    ms, samples = timing(call, repeats=200, rounds=1)
                    rows.append(
                        {
                            "case": case,
                            "round": round_id,
                            "variant": name,
                            "calls": len(bank),
                            "ms": ms,
                            "samples_ms": samples,
                        }
                    )
            print(f"Timed {case}", flush=True)
        report["timing"] = {"rows": rows, "order": order}
        summary = {}
        for case in ("local-full", "local-padded", "global-padded"):
            medians = {
                name: statistics.median(
                    [
                        row["ms"]
                        for row in rows
                        if row["case"] == case and row["variant"] == name
                    ]
                )
                for name in eligible
            }
            by_round = {
                name: {
                    row["round"]: row["ms"]
                    for row in rows
                    if row["case"] == case and row["variant"] == name
                }
                for name in eligible
            }
            summary[case] = {
                "median_ms": medians,
                "speedup": {
                    name: medians["native-special-32"] / value
                    for name, value in medians.items()
                },
                "faster_rounds": {
                    name: sum(
                        by_round[name][i] < by_round["native-special-32"][i]
                        for i in range(9)
                    )
                    for name in eligible
                },
            }
        report["summary"] = summary
    report["source_sha256"] = {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(Path("experiments/frontier").glob("query_shard*"))
        if path.is_file()
    }
    Path("results/frontier/query-shard.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(
        json.dumps({"exact_variants": eligible, "summary": summary}, indent=2),
        flush=True,
    )


if __name__ == "__main__":
    main()
