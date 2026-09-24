"""Check every vocabulary row under four different token-position permutations."""

import json
from pathlib import Path

import torch

from experiments.native import common

from .engine import FrontierEngine


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    torch.manual_seed(7295)
    report = {"metadata": common.metadata(), "permutations": []}
    with FrontierEngine(policy="bf16-splitk-exact", token_tables=True) as engine:
        encoder = engine.base.model.net.encoder
        count = encoder.frontier_token_norm.shape[0]
        for seed in range(4):
            ids = torch.randperm(count, device="cuda")
            errors = torch.zeros(2, device="cuda", dtype=torch.int64)
            for offset in range(0, count, 64):
                selected = ids[offset : offset + 64]
                if selected.numel() != 64:
                    selected = torch.cat(
                        (selected, selected.new_zeros(64 - selected.numel()))
                    )
                selected = selected.view(1, 64)
                normalized = encoder.embeddings(selected)
                projected = encoder.layers[0].attn.Wqkv(normalized.bfloat16())
                errors[0] += (
                    normalized.view(torch.int32)
                    != encoder.frontier_token_norm[selected].view(torch.int32)
                ).sum()
                errors[1] += (
                    projected.view(torch.int16)
                    != encoder.frontier_token_qkv[selected].view(torch.int16)
                ).sum()
            values = errors.cpu().tolist()
            report["permutations"].append(
                {
                    "index": seed,
                    "vocabulary_rows": count,
                    "normalized_bit_mismatches": values[0],
                    "qkv_bit_mismatches": values[1],
                }
            )
        report["all_exact"] = all(
            not r["normalized_bit_mismatches"] and not r["qkv_bit_mismatches"]
            for r in report["permutations"]
        )
    Path("results/frontier/token-table-parity.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2), flush=True)
    if not report["all_exact"]:
        raise RuntimeError("Vocabulary table position-independence check failed")


if __name__ == "__main__":
    main()
