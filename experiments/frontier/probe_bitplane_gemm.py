"""Compare hardware-compressed bitplanes plus warp decode against BF16 GEMM."""

import hashlib
import itertools
import json
import random
from pathlib import Path

import torch

from experiments.native import common

from .bitplane_gemm import COMPILED, gemm, pack
from .compression import Allocation
from .engine import FrontierEngine
from .tune import timing


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    torch.manual_seed(281004)
    path = Path("results/frontier/bitplane-gemm.json")
    report = {
        "metadata": common.metadata(),
        "scope": "28 distinct MLP input matrices, independent random activations, isolated GEMM only",
        "source_sha256": {
            name: hashlib.sha256(
                Path(__file__).with_name(name).read_bytes()
            ).hexdigest()
            for name in ["bitplane_gemm.py", "probe_bitplane_gemm.py"]
        },
        "rows": [],
    }
    with FrontierEngine(policy="bf16-splitk-exact", max_graphs=1) as engine:
        modules = [layer.mlp.Wi for layer in engine.base.model.net.encoder.layers]
        inputs = [
            torch.randn(1, 64, 1024, device="cuda", dtype=torch.bfloat16)
            for _ in modules
        ]

        def baseline():
            return [m(x) for m, x in zip(modules, inputs)]

        expected = baseline()
        for bn in [32, 64]:
            encoded = torch.stack([pack(m.weight, bn) for m in modules])
            owners = {}
            try:
                for name, compressed in [
                    ("plain-bitplanes", False),
                    ("compressed-bitplanes", True),
                ]:
                    owners[name] = Allocation(encoded.shape, encoded.dtype, compressed)
                    owners[name].copy(encoded)
                bank = {
                    name: list(owner.tensor.unbind()) for name, owner in owners.items()
                }
                bank["bf16-control"] = [m.weight for m in modules]
                configs = list(itertools.product([32, 64], [bn], [1]))
                random.Random(1428).shuffle(configs)
                for config in configs:
                    labels = list(bank)
                    random.Random(1731 + sum(config)).shuffle(labels)
                    for label in labels:
                        row = {"variant": label, "config": config}
                        try:
                            packed = label != "bf16-control"

                            def run(weights=bank[label], config=config, packed=packed):
                                return [
                                    gemm(x, w, 5248, 1024, config, packed)
                                    for x, w in zip(inputs, weights)
                                ]

                            actual = run()
                            row["mismatches"] = sum(
                                int((a != b).sum()) for a, b in zip(actual, expected)
                            )
                            row["ms"], row["samples_ms"] = timing(
                                run, repeats=20, rounds=3
                            )
                            row["baseline_ms"], row["baseline_samples_ms"] = timing(
                                baseline, repeats=20, rounds=3
                            )
                            row["speedup"] = row["baseline_ms"] / row["ms"]
                            kernel = COMPILED[(*config, packed)]
                            row["registers"] = kernel.n_regs
                            row["shared_bytes"] = kernel.metadata.shared
                            row["warp_shuffle"] = "shfl.sync.bfly" in kernel.asm["ptx"]
                        except Exception as error:  # noqa: BLE001 - retain compilation/resource failures.
                            row["error"] = str(error)[:1500]
                        report["rows"].append(row)
                        path.write_text(json.dumps(report, indent=2) + "\n")
                        print(json.dumps(row), flush=True)
            finally:
                for owner in owners.values():
                    owner.close()


if __name__ == "__main__":
    main()
