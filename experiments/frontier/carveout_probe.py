"""Screen per-kernel shared-memory/L1 preferences; restore all function settings."""

# Callbacks run synchronously before the enclosing stage loop advances.
# ruff: noqa: B023

import hashlib
import json
import random
import statistics
from pathlib import Path

import torch
from cuda.bindings import driver as cuda

from experiments.native import common

from . import mlp_geglu, tma
from .engine import FrontierEngine
from .tune import timing

ATTR = cuda.CUfunction_attribute.CU_FUNC_ATTRIBUTE_PREFERRED_SHARED_MEMORY_CARVEOUT


def checked(result):
    status, *values = result
    if int(status):
        raise RuntimeError(str(status))
    return values[0] if len(values) == 1 else values


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    torch.manual_seed(60723)
    root = Path(__file__).resolve().parents[2]
    path = root / "results/frontier/carveout.json"
    report = {
        "metadata": common.metadata(),
        "scope": "Per-function shared-memory carveout hints, isolated 28-matrix graph replay. Hints are verified through cuFuncGetAttribute, not measured physical cache partition sizes. Each graph is captured after setting the hint. All preferences restored in finally blocks.",
        "documentation": "https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html#configuring-l1-shared-memory-balance",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "stages": {},
    }

    def save():
        path.write_text(json.dumps(report, indent=2) + "\n")

    rng = random.Random(60723)
    with FrontierEngine(policy="bf16-splitk-exact", max_graphs=1) as engine:
        layers = engine.base.model.net.encoder.layers
        for field in ["attn.Wqkv", "mlp.Wi"]:
            group, attr = field.split(".")
            weights = [getattr(getattr(layer, group), attr).weight for layer in layers]
            inputs = [
                torch.randn(1, 64, 1024, device="cuda", dtype=torch.bfloat16)
                for _ in weights
            ]
            if field == "attn.Wqkv":
                config = tuple(engine.selection[field]["config"])
                operation, compiled = tma.matmul, tma.COMPILED
            else:
                config = (32, 64, 64, 4, 3, 2, False)
                operation, compiled = mlp_geglu.project, mlp_geglu.COMPILED

            def run():
                return [operation(x, w, config) for x, w in zip(inputs, weights)]

            expected = run()
            kernel = list(compiled.values())[-1]
            function = cuda.CUfunction(kernel.function)
            original = checked(cuda.cuFuncGetAttribute(ATTR, function))
            stage = {
                "configuration": config,
                "original_hint": original,
                "kernel_name": kernel.name,
                "cubin_sha256": hashlib.sha256(kernel.asm["cubin"]).hexdigest(),
                "shared_bytes": kernel.metadata.shared,
                "registers": kernel.n_regs,
                "rows": [],
            }
            report["stages"][field] = stage
            hints = sorted({original, 0, 25, 50, 75, 100})
            try:
                for round_id in range(7):
                    order = hints.copy()
                    rng.shuffle(order)
                    for hint in order:
                        checked(cuda.cuFuncSetAttribute(function, ATTR, hint))
                        assert checked(cuda.cuFuncGetAttribute(ATTR, function)) == hint
                        row = {"round": round_id, "requested_hint": hint}
                        actual = run()
                        row["mismatches"] = sum(
                            int((a.view(torch.int16) != b.view(torch.int16)).sum())
                            for a, b in zip(actual, expected)
                        )
                        if row["mismatches"]:
                            raise RuntimeError(
                                "Changing a cache preference changed output"
                            )
                        row["ms"], row["samples_ms"] = timing(run, repeats=40, rounds=1)
                        row["hint_after_replay"] = checked(
                            cuda.cuFuncGetAttribute(ATTR, function)
                        )
                        assert row["hint_after_replay"] == hint
                        stage["rows"].append(row)
                    save()
            finally:
                checked(cuda.cuFuncSetAttribute(function, ATTR, original))
                stage["restored_hint"] = checked(
                    cuda.cuFuncGetAttribute(ATTR, function)
                )
                assert stage["restored_hint"] == original
                save()
            stage["median_ms"] = {
                hint: statistics.median(
                    v
                    for row in stage["rows"]
                    if row["requested_hint"] == hint
                    for v in row["samples_ms"]
                )
                for hint in hints
            }
            save()
            print(field, json.dumps(stage["median_ms"]), flush=True)


if __name__ == "__main__":
    main()
