"""Screen head projections on actual activations and independent random inputs."""

import argparse
import hashlib
import json
import random
import types
from pathlib import Path

import torch
from torch.nn import functional as F

from experiments.native import common

from .engine import FrontierEngine
from .head_gemm import gemm
from .tune import timing


def capture(engine):
    model = engine.base.model
    records = {}
    handles = []

    def hook(name):
        def store(module, args):
            if name not in records:
                records[name] = (
                    args[0].clone().contiguous(),
                    module.weight,
                    module.bias,
                )

        return store

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and not name.startswith("net.encoder"):
            handles.append(module.register_forward_pre_hook(hook(name)))
    original = model._head_layer
    fn = original.__func__
    namespace = dict(fn.__globals__)
    functional = types.SimpleNamespace(**vars(namespace["F"]))
    weight_names = {
        layer.self_attn.in_proj_weight.data_ptr(): f"net.head.layers.{i}.in_proj"
        for i, layer in enumerate(model.net.head.layers)
    }

    def linear(x, weight, bias=None):
        name = weight_names.get(weight.data_ptr())
        if name is not None and name not in records:
            records[name] = (x.clone().contiguous(), weight, bias)
        return F.linear(x, weight, bias)

    functional.linear = linear
    namespace["F"] = functional
    model._head_layer = types.MethodType(
        types.FunctionType(
            fn.__code__, namespace, fn.__name__, fn.__defaults__, fn.__closure__
        ),
        model,
    )
    try:
        engine.predict(**common.workload(1, "short"))
    finally:
        model._head_layer = original
        for handle in handles:
            handle.remove()
    return records


def configs():
    return [
        (bm, bn, bk, split, 4, stages, partial)
        for bm, bn, bk in [
            (16, 32, 32),
            (16, 32, 64),
            (16, 64, 64),
            (32, 32, 64),
            (32, 64, 64),
            (64, 32, 64),
            (64, 64, 64),
        ]
        for stages in [2, 3, 4]
        for split, partial in [(1, False), (2, False), (4, False), (2, True), (4, True)]
    ]


@torch.inference_mode()
def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--output", type=Path, default=Path("results/frontier/head-gemm.json")
    )
    p.add_argument("--fields", nargs="*")
    args = p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(752394)
    report = {
        "metadata": common.metadata(),
        "method": "Isolated CUDA Graph GEMMs with real head weights; actual request input plus four independently random BF16 inputs check parity. No full-request latency claim.",
        "rows": [],
        "source_sha256": {
            name: hashlib.sha256(
                Path(__file__).with_name(name).read_bytes()
            ).hexdigest()
            for name in ["head_gemm.py", "head_probe.py"]
        },
    }
    with FrontierEngine(
        policy="bf16-splitk-exact",
        attention="native",
        fuse_reduce_norm=True,
        token_tables=True,
        max_graphs=1,
    ) as engine:
        records = capture(engine)
        groups = {}
        for name, record in records.items():
            x, w, _b = record
            key = (x.numel() // x.shape[-1], *w.shape)
            groups.setdefault(key, []).append((name, record))
        for shape, entries in groups.items():
            field = "x".join(map(str, shape))
            if args.fields and field not in args.fields:
                continue
            relu = all(name.endswith("linear1") for name, _ in entries)
            samples = [(x, w, b) for _, (x, w, b) in entries]
            checks = [
                sample
                for x, w, b in samples
                for sample in [(x, w, b)]
                + [(torch.randn_like(x), w, b) for _ in range(4)]
            ]

            def reference(x, w, b, relu=relu):
                y = F.linear(x, w, b)
                return F.relu(y) if relu else y

            expected = [reference(*sample) for sample in checks]
            base_ms, base_samples = timing(
                lambda samples=samples, reference=reference: [
                    reference(*sample) for sample in samples
                ],
                repeats=100,
                rounds=7,
            )
            report["rows"].append(
                {
                    "field": field,
                    "names": [name for name, _ in entries],
                    "shape": shape,
                    "variant": "cublas",
                    "relu": relu,
                    "ms": base_ms,
                    "samples_ms": base_samples,
                }
            )
            configurations = configs()
            random.Random(32819).shuffle(configurations)
            for config in configurations:
                row = {
                    "field": field,
                    "variant": "triton-bias",
                    "config": config,
                    "relu": relu,
                }
                try:
                    actual = [gemm(*sample, config, relu=relu) for sample in checks]
                    row["mismatches"] = sum(
                        int((a != b).sum()) for a, b in zip(actual, expected)
                    )
                    row["actual_input_mismatches"] = sum(
                        int((actual[i * 5] != expected[i * 5]).sum())
                        for i in range(len(samples))
                    )
                    row["max_abs_error"] = max(
                        float((a.float() - b.float()).abs().max())
                        for a, b in zip(actual, expected)
                    )
                    row["ms"], row["samples_ms"] = timing(
                        lambda samples=samples, config=config, relu=relu: [
                            gemm(*sample, config, relu=relu) for sample in samples
                        ],
                        repeats=100,
                        rounds=3,
                    )
                    row["speedup"] = base_ms / row["ms"]
                except Exception as error:  # noqa: BLE001 - retain failed configurations
                    row["error"] = str(error)
                report["rows"].append(row)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
            valid = [
                r
                for r in report["rows"]
                if r["field"] == field and r.get("mismatches") == 0
            ]
            best = sorted(valid, key=lambda r: r["ms"])[:3]
            print(
                json.dumps(
                    {
                        "field": field,
                        "names": [name for name, _ in entries],
                        "baseline_ms": base_ms,
                        "exact_configurations": len(valid),
                        "best": best,
                    }
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
