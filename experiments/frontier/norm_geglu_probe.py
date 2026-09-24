"""Measure exact duplicated Welford prerequisites on captured real activations."""

import ctypes
import fcntl
import json
import types
from pathlib import Path

import torch

from experiments.native import common

from .engine import FrontierEngine
from .mlp_geglu import project
from .norm_geglu_build import DIRECTORY, ROOT, SOURCE, digest
from .tune import timing


def mismatch(a, b):
    dtype = torch.int32 if a.dtype == torch.float32 else torch.int16
    return int((a.view(dtype) != b.view(dtype)).sum())


class Operation:
    def __init__(self, lib, row, bm, bn, validate):
        self.lib, self.row, self.bm, self.bn, self.validate = (
            lib,
            row,
            bm,
            bn,
            int(validate),
        )
        self.tiles = (5248 + bn - 1) // bn
        x = row["x"]
        self.stats = torch.empty(
            (self.tiles, 64, 2), device=x.device, dtype=torch.float32
        )
        self.residual = torch.empty_like(x)
        self.normalized = torch.empty_like(x, dtype=torch.bfloat16)
        info = (ctypes.c_int * 4)()
        error = lib.norm_geglu_prologue(
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            bm,
            self.tiles,
            self.validate,
            row["norm"].eps,
            None,
            info,
        )
        if error:
            raise RuntimeError(f"Resource query failed: {error}")
        self.resources = dict(
            zip(
                ["registers", "shared_bytes", "local_bytes", "active_blocks_per_sm"],
                info,
            )
        )

    def __call__(self):
        row, norm = self.row, self.row["norm"]
        error = self.lib.norm_geglu_prologue(
            row["x"].data_ptr(),
            row["r"].data_ptr(),
            norm.weight.data_ptr(),
            norm.bias.data_ptr() if norm.bias is not None else None,
            self.stats.data_ptr(),
            self.residual.data_ptr(),
            self.normalized.data_ptr(),
            self.bm,
            self.tiles,
            self.validate,
            norm.eps,
            torch.cuda.current_stream().cuda_stream,
            None,
        )
        if error:
            raise RuntimeError(f"Prologue launch failed: {error}")
        return self.stats


@torch.inference_mode()
def run():
    torch.set_num_threads(4)
    build = json.loads((DIRECTORY / "build.json").read_text())
    if build["source_sha256"] != digest(SOURCE) or build["library_sha256"] != digest(
        DIRECTORY / "norm_geglu.so"
    ):
        raise RuntimeError("Native source/binary hash changed")
    lib = ctypes.CDLL(str(DIRECTORY / "norm_geglu.so"))
    p, i = ctypes.c_void_p, ctypes.c_int
    lib.norm_geglu_prologue.argtypes = [
        p,
        p,
        p,
        p,
        p,
        p,
        p,
        i,
        i,
        i,
        ctypes.c_float,
        p,
        ctypes.POINTER(i),
    ]
    constructor = json.loads((ROOT / "results/frontier/summary.json").read_text())[
        "recommended_constructor"
    ].copy()
    constructor["policy"] = constructor["policy"].removesuffix("-short-compiled")
    choices = json.loads(
        (ROOT / "results/frontier/mlp-geglu-unpacked.json").read_text()
    )["rows"]
    config = max(
        [r for r in choices if r.get("mismatches") == 0 and r.get("speedup", 0) > 1.03],
        key=lambda r: r["speedup"],
    )["config"]
    report = {
        "metadata": common.metadata(),
        "constructor": constructor,
        "build": build,
        "source_sha256": {
            str(p.relative_to(ROOT)): digest(p)
            for p in [
                SOURCE,
                Path(__file__),
                ROOT / "experiments/frontier/norm_geglu_build.py",
            ]
        },
        "scope": "Optimistic prologue-only mapping control, not fused inference or a rigorous execution lower bound. Timed candidate computes duplicated exact mean/rstd only; omits normalization application, residual output and all GEMM work.",
        "method": "One physical warp simulates each original group of four Welford warps; four rows in parallel per 128-thread CTA. Original online-value order, shfl-down16/8/4/2/1 and warp0+2/warp1+3 tree retained.",
        "retained_fused_config": config,
        "fixtures": [],
        "rows": [],
    }
    with FrontierEngine(**constructor, max_graphs=1) as engine:
        model = engine.base.model
        layers = list(model.net.encoder.layers)
        ids = {id(layer.mlp_norm): idx for idx, layer in enumerate(layers)}
        original = model.forward.__func__
        namespace = dict(original.__globals__)
        add_norm = namespace["add_norm"]
        captured = {}
        outputs = {}

        def capture(x, residual, norm):
            y, n = add_norm(x, residual, norm)
            if id(norm) in ids:
                idx = ids[id(norm)]
                captured[idx] = {
                    "x": x.clone(),
                    "r": residual.clone(),
                    "residual": y.clone(),
                    "normalized": n.clone(),
                    "norm": norm,
                    "weight": layers[idx].mlp.Wi.weight,
                }
            return y, n

        def hook(idx):
            def inner(module, args):
                outputs[idx] = args[0].clone()

            return inner

        namespace["add_norm"] = capture
        forward = types.FunctionType(
            original.__code__,
            namespace,
            original.__name__,
            original.__defaults__,
            original.__closure__,
        )
        forward.__kwdefaults__ = original.__kwdefaults__
        model.forward = types.MethodType(forward, model)
        handles = [
            layer.mlp.Wo.register_forward_pre_hook(hook(i))
            for i, layer in enumerate(layers)
        ]
        bank = None
        try:
            fixtures = [common.workload(1, "short"), *common.validation_requests()]
            for idx, request in enumerate(fixtures):
                prepared = engine.prepare(**request)
                key = engine.base._graph_key(prepared)
                if key[:2] != (1, 64):
                    continue
                captured.clear()
                outputs.clear()
                host = engine.base._allocate(key[:3], host=True)
                engine.base._fill(host, prepared)
                inputs = {
                    name: tensor.to(engine.base.device) for name, tensor in host.items()
                }
                inputs["global_attention_unmasked"] = key[-1]
                engine.base._forward(inputs)
                assert len(captured) == len(outputs) == 28
                records = [dict(captured[i], geglu=outputs[i]) for i in range(28)]
                fixture = {
                    "index": idx,
                    "request": request,
                    "tokens": prepared.input_tokens,
                    "checks": [],
                }
                for bm, bn in [(32, 64), (32, 128), (16, 128)]:
                    row = {"bm": bm, "bn": bn, "layers": []}
                    for record in records:
                        op = Operation(lib, record, bm, bn, True)
                        stats = op()
                        expected_stats = torch.stack(
                            torch.native_layer_norm(
                                record["residual"],
                                [1024],
                                None,
                                None,
                                record["norm"].eps,
                            )[1:],
                            dim=-1,
                        ).reshape(1, 64, 2)
                        item = {
                            "residual_mismatches": mismatch(
                                op.residual, record["residual"]
                            ),
                            "normalized_mismatches": mismatch(
                                op.normalized, record["normalized"]
                            ),
                            "statistics_mismatches": mismatch(
                                stats, expected_stats.expand_as(stats)
                            ),
                            "geglu_mismatches": mismatch(
                                project(op.normalized, record["weight"], config),
                                record["geglu"],
                            ),
                        }
                        row["layers"].append(item)
                        if any(item.values()):
                            raise RuntimeError(f"Exact prologue failed: {item}")
                    fixture["checks"].append(row)
                report["fixtures"].append(fixture)
                if bank is None:
                    bank = records
                if len(report["fixtures"]) == 3:
                    break
        finally:
            model.forward = types.MethodType(original, model)
            for handle in handles:
                handle.remove()
        assert bank is not None

        def retained_norm():
            return [add_norm(r["x"], r["r"], r["norm"]) for r in bank]

        def retained_chain():
            result = []
            for r in bank:
                h, n = add_norm(r["x"], r["r"], r["norm"])
                result.append((h, project(n, r["weight"], config)))
            return result

        report["baseline_norm_ms"], report["baseline_norm_samples_ms"] = timing(
            retained_norm, 30, 5
        )
        report["baseline_chain_ms"], report["baseline_chain_samples_ms"] = timing(
            retained_chain, 30, 5
        )
        for bm, bn in [(32, 64), (32, 128), (16, 128)]:
            operations = [Operation(lib, r, bm, bn, False) for r in bank]

            def candidate(operations=operations):
                return [op() for op in operations]

            actual = candidate()
            # Confirm the stats-only specialization computes the validated values.
            errors = []
            for stats, r in zip(actual, bank):
                expected = torch.stack(
                    torch.native_layer_norm(
                        r["residual"], [1024], None, None, r["norm"].eps
                    )[1:],
                    dim=-1,
                ).reshape(1, 64, 2)
                errors.append(mismatch(stats, expected.expand_as(stats)))
            assert not any(errors)
            ms, samples = timing(candidate, 30, 5)
            row = {
                "bm": bm,
                "bn": bn,
                "n_tiles": operations[0].tiles,
                "grid": [operations[0].tiles, 64 // bm],
                "threads": 128,
                "resources": operations[0].resources,
                "statistics_mismatches": sum(errors),
                "ms": ms,
                "samples_ms": samples,
                "ratio_to_entire_retained_chain": ms / report["baseline_chain_ms"],
                "logical_input_bytes": 28 * 64 * 1024 * 6 * operations[0].tiles,
            }
            report["rows"].append(row)
            print(json.dumps(row), flush=True)
        report["baseline_chain_after_ms"], _ = timing(retained_chain, 30, 5)
    destination = ROOT / "results/frontier/norm-geglu-prologue.json"
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(destination, flush=True)


def main():
    with Path("/tmp/laya-gpu-experiments.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        run()


if __name__ == "__main__":
    main()
