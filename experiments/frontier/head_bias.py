"""Inspect bias rounding of captured head GEMMs before choosing epilogues."""

import json
from pathlib import Path

import torch
from torch.nn import functional as F

from .engine import FrontierEngine
from .head_gemm import gemm
from .head_probe import capture


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    rows = []
    with FrontierEngine(
        policy="bf16-splitk-exact",
        attention="native",
        fuse_reduce_norm=True,
        token_tables=True,
        max_graphs=1,
    ) as engine:
        records = capture(engine)
        for name, (x, w, b) in records.items():
            expected = F.linear(x, w, b)
            no_bias = F.linear(x, w)
            rounded = no_bias + b
            fused = gemm(x, w, b, (32, 64, 64, 1, 4, 3, False))
            rounded_custom = (
                gemm(x, w, torch.zeros_like(b), (32, 64, 64, 1, 4, 3, False)) + b
            )
            row = {
                "name": name,
                "shape": list(x.shape),
                "bias_dtype": str(b.dtype),
                "input_dtype": str(x.dtype),
                "weight_dtype": str(w.dtype),
                "bias_max": float(b.abs().max()),
                "input_max": float(x.abs().max()),
                "native_rounded_bias_mismatches": int((expected != rounded).sum()),
                "custom_fused_bias_mismatches": int((expected != fused).sum()),
                "custom_rounded_bias_mismatches": int(
                    (expected != rounded_custom).sum()
                ),
            }
            rows.append(row)
            print(row, flush=True)
    Path("results/frontier/head-bias.json").write_text(
        json.dumps(rows, indent=2) + "\n"
    )


if __name__ == "__main__":
    main()
