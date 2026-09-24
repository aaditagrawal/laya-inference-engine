"""Bounded comparison of exact sparse-erf and complete-BF16-LUT epilogues."""

import hashlib
import json
import random
from pathlib import Path

import torch

from experiments.native.common import metadata
from experiments.native.engine import ExperimentalEngine
from experiments.native.kernels.candidates import provenance, triton_geglu_corrected

from .kernels import COMPILED, pack_wi, qkv_rope, wi_geglu
from .microbench import errors, timing


def main():
    torch.set_num_threads(4)
    torch.manual_seed(1701)
    out = Path("results/latency-optimizations/fusion/epilogues.json")
    report = {"metadata": metadata(), "rows": [], "compiled": []}
    with ExperimentalEngine() as engine, torch.inference_mode():
        model = engine.base.model
        linear = model.net.encoder.layers[0].mlp.Wi
        packed, bias = pack_wi(linear.weight, linear.bias)
        for batch, length in [(1, 64), (1, 512), (16, 64), (16, 512)]:
            x = torch.randn(batch, length, 1024, device="cuda", dtype=torch.bfloat16)
            reference = triton_geglu_corrected(linear(x))
            configs = (
                [(32, 32, 32, 4, 3)]
                if batch * length == 64
                else [(64, 32, 32, 4, 3), (64, 64, 32, 4, 3)]
            )
            jobs = [("baseline", None, lambda x=x: triton_geglu_corrected(linear(x)))]
            for config in configs:
                for lut in [False, True]:
                    fn = lambda config=config, lut=lut, x=x: wi_geglu(
                        x, packed, bias, config, packed=True, lut=lut
                    )
                    jobs.append(("lut" if lut else "erf", config, fn))
            random.Random(4903).shuffle(jobs)
            for kind, config, fn in jobs:
                row = {
                    "batch": batch,
                    "length": length,
                    "variant": kind,
                    "config": config,
                    **errors(fn(), reference),
                }
                row["median_us"], row["samples_us"] = timing(fn)
                report["rows"].append(row)
                print(json.dumps(row), flush=True)
                out.write_text(json.dumps(report, indent=2))
        # Keep representative actual compiler output, not inferred ISA claims.
        qlinear = model.net.encoder.layers[0].attn.Wqkv
        qkv_rope(
            x,
            qlinear.weight,
            qlinear.bias,
            model.local_cos,
            model.local_sin,
            (64, 64, 32, 4, 3),
            packed=False,
            padding=64,
        )
        for key, compiled in COMPILED.items():
            ptx = compiled.asm["ptx"]
            row = {
                "key": str(key),
                "registers": compiled.n_regs,
                "spills": compiled.n_spills,
                "shared_bytes": compiled.metadata.shared,
                "ptx_sha256": hashlib.sha256(ptx.encode()).hexdigest(),
                "target_sm120a": ".target sm_120a" in ptx,
                "mma_sync_count": ptx.count("mma.sync"),
                "ldmatrix_count": ptx.count("ldmatrix"),
                "cp_async_count": ptx.count("cp.async"),
            }
            if key[1] == 8192:
                filename = f"{key[0]}-8192-{'lut' if key[0] == 'wi' and key[3] else 'erf'}-{key[-1][0]}x{key[-1][1]}.ptx"
                (out.parent / filename).write_text(ptx)
                row["ptx_file"] = filename
            report["compiled"].append(row)
        report["kernel_provenance"] = provenance()
        out.write_text(json.dumps(report, indent=2))
    print(out, flush=True)


if __name__ == "__main__":
    main()
