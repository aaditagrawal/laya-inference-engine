"""Verify generated MXFP8 instructions independently of timing measurements."""

import hashlib
import json
from pathlib import Path

import torch

from experiments.native import common

from .mxfp8 import COMPILED, matmul, quantize


@torch.inference_mode()
def main():
    path = Path("results/frontier/matmul-mxfp8.json")
    report = json.loads(path.read_text())
    dimensions = {
        "attn.Wqkv": (3072, 1024),
        "attn.Wo": (1024, 1024),
        "mlp.Wi": (5248, 1024),
        "mlp.Wo": (1024, 2624),
    }
    verified = {}
    for field, (n, k) in dimensions.items():
        x = torch.ones(64, k, device="cuda", dtype=torch.bfloat16)
        q, scales = quantize(torch.ones(n, k, device="cuda", dtype=torch.bfloat16))
        for row in report["rows"]:
            if row["field"] != field or row.get("variant") != "mxfp8" or "error" in row:
                continue
            config = tuple(row["config"])
            matmul(x, q, scales, config)
            ptx = COMPILED[(64, n, k, *config)].asm["ptx"]
            instruction = next(
                (
                    line.strip().split(" {")[0]
                    for line in ptx.splitlines()
                    if "kind::mxf8f6f4.block_scale" in line
                ),
                None,
            )
            if instruction is None:
                raise RuntimeError(f"Native block-scaled MMA missing: {field} {config}")
            row.update(
                native_block_scaled_mma=True,
                mma_instruction=instruction,
                ptx_sha256=hashlib.sha256(ptx.encode()).hexdigest(),
            )
            verified[(field, config)] = {
                key: row[key]
                for key in ["native_block_scaled_mma", "mma_instruction", "ptx_sha256"]
            }
    report["ptx_verification"] = {
        "metadata": common.metadata(),
        "verified_configurations": len(verified),
        "correction": "The original detector expected a different PTX qualifier order and recorded false negatives. Generated PTX was rechecked using kind::mxf8f6f4.block_scale. Timing and numerical measurements are unchanged.",
    }
    path.write_text(json.dumps(report, indent=2) + "\n")
    for full_path in Path("results/frontier").glob("full-mx-*.json"):
        full = json.loads(full_path.read_text())
        for field, selected in full["selection"].items():
            selected.update(verified[(field, tuple(selected["config"]))])
        full["ptx_metadata_correction"] = report["ptx_verification"]
        full_path.write_text(json.dumps(full, indent=2) + "\n")
    print(report["ptx_verification"], flush=True)


if __name__ == "__main__":
    main()
