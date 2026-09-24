"""Exact selected-Q projection and chained final-head screen on real requests."""

import argparse
import hashlib
import json
import types
from pathlib import Path

import torch

from experiments.native import common

from .engine import FrontierEngine
from .head_gemm import gemm
from .head_q_select_adapter import replacement
from .head_q_select_kernel import binary_hashes, project
from .holdout import requests
from .tune import timing


def source_hashes():
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(Path(__file__).parent.glob("head_q_select*.py"))
    }


@torch.inference_mode()
def capture(engine, request):
    prepared = engine.prepare(**request)
    key = engine.base._graph_key(prepared)
    if key[:2] != (1, 64):
        return None
    host = engine.base._allocate(key[:3], host=True)
    engine.base._fill(host, prepared)
    inputs = {k: v.to(engine.base.device) for k, v in host.items()}
    inputs["global_attention_unmasked"] = key[-1]
    model = engine.base.model
    original = model._head_layer
    namespace = original.__func__.__globals__
    old_qkv = namespace["head_qkv"]
    record = {}
    final_layer = model.net.head.layers[-1]

    def qkv(x, module):
        if module is final_layer.self_attn:
            record["x"] = x
        return old_qkv(x, module)

    def head(self, h, layer, mask, select=None):
        result = original(h, layer, mask, select)
        if layer is final_layer:
            record.update(h=h, layer=layer, mask=mask, select=select, output=result)
        return result

    namespace["head_qkv"] = qkv
    model._head_layer = types.MethodType(head, model)
    try:
        engine.base._forward(inputs)
    finally:
        namespace["head_qkv"] = old_qkv
        model._head_layer = original
    if not record or record["select"].shape[1] > 32:
        return None
    return record


def projected_baseline(record):
    module = record["layer"].self_attn
    packed = gemm(
        record["x"],
        module.in_proj_weight,
        module.in_proj_bias,
        (32, 64, 64, 1, 4, 3, False),
    )
    q, k, v = packed.view(1, 64, 3, 16, 64).permute(0, 3, 2, 1, 4).unbind(2)
    q = q.gather(2, record["select"][:, None, :, None].expand(-1, 16, -1, 64))
    return q, k, v


def projected_candidate(record, bm):
    module = record["layer"].self_attn
    q, packed = project(
        record["x"], module.in_proj_weight, module.in_proj_bias, record["select"], bm
    )
    unused, k, v = packed.view(1, 64, 3, 16, 64).permute(0, 3, 2, 1, 4).unbind(2)
    return (q, k, v), unused


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/frontier/head_q_select-screen.json"),
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    before = source_hashes()
    report = {
        "metadata": common.metadata(),
        "source_before": before,
        "sources": [],
        "rows": [],
        "scope": "Final head Q-selected/KV-full projection and chained head; not full-request latency",
        "reference_config": [32, 64, 64, 1, 4, 3, False],
    }
    with FrontierEngine(policy="bf16-exact", head_kernels=True, max_graphs=2) as engine:
        model = engine.base.model
        original = model._head_layer
        fixtures = [
            ("benchmark", [common.workload(1, "short")]),
            ("original", common.validation_requests()),
            ("holdout", requests()),
        ]
        if args.quick:
            fixtures = [fixtures[0], ("holdout", requests()[:2])]
        else:
            fixtures += [
                (
                    "extra-fully-occupied",
                    json.loads(
                        Path(
                            "results/frontier/attention-special-screen.json"
                        ).read_text()
                    )["requests"][1:],
                )
            ]
        records = []
        for suite, cases in fixtures:
            for index, request in enumerate(cases):
                record = capture(engine, request)
                if record is not None:
                    records.append(record)
                    report["sources"].append(
                        {
                            "suite": suite,
                            "index": index,
                            "selected_rows": record["select"].shape[1],
                            "request": request,
                        }
                    )
        print(f"Captured {len(records)} final-head inputs", flush=True)
        references = [projected_baseline(record) for record in records]
        for bm in (32, 16):
            modified = types.MethodType(replacement(original, bm), model)
            row = {
                "bm": bm,
                "projection_mismatches": [],
                "chain_mismatches": [],
                "unused_zero": True,
            }
            for record, reference in zip(records, references):
                actual, unused = projected_candidate(record, bm)
                row["projection_mismatches"].append(
                    sum(int((a != b).sum()) for a, b in zip(actual, reference))
                )
                row["unused_zero"] &= bool((unused == 0).all())
                chained = modified(
                    record["h"], record["layer"], record["mask"], record["select"]
                )
                row["chain_mismatches"].append(int((chained != record["output"]).sum()))
            row["all_exact"] = (
                not any(row["projection_mismatches"] + row["chain_mismatches"])
                and row["unused_zero"]
            )
            row["binary_before_timing"] = binary_hashes()
            row["timings"] = {}
            padded = next(
                r for r, s in zip(records, report["sources"]) if s["suite"] == "holdout"
            )
            for case, record in [("full", records[0]), ("padded", padded)]:
                calls = {
                    "projection_reference": lambda r=record: projected_baseline(r),
                    "projection_candidate": lambda r=record, bm=bm: projected_candidate(
                        r, bm
                    ),
                    "chain_reference": lambda r=record: original(
                        r["h"], r["layer"], r["mask"], r["select"]
                    ),
                    "chain_candidate": lambda r=record, fn=modified: fn(
                        r["h"], r["layer"], r["mask"], r["select"]
                    ),
                }
                row["timings"][case] = {}
                for name, call in calls.items():
                    ms, samples = timing(call, repeats=100, rounds=7)
                    row["timings"][case][name] = {"ms": ms, "samples_ms": samples}
            row["binary_after_timing"] = binary_hashes()
            report["rows"].append(row)
            report["source_after"] = source_hashes()
            assert report["source_after"] == before
            assert row["binary_before_timing"] == row["binary_after_timing"]
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(row, flush=True)


if __name__ == "__main__":
    main()
