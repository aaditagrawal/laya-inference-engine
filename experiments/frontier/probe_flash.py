"""Screen the upstream SM120 CuTe attention implementation on base Torch."""

import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

import torch
from torch.nn import functional as F

from experiments.native import common

from .engine import FrontierEngine
from .probe_attention import capture_inputs
from .tune import timing


@torch.inference_mode()
def main():
    if str(torch.__version__) != "2.14.0+cu132":
        raise RuntimeError(
            "Run from the base environment to preserve the measured Torch/CUDA stack"
        )
    setup = json.loads(Path(".research/frontier-flash-dependencies.json").read_text())
    for path in setup["dependency_paths"]:
        if path not in sys.path and Path(path).exists():
            sys.path.append(path)
    from flash_attn.cute.interface import _flash_attn_fwd

    torch.set_num_threads(4)
    report = {
        "metadata": common.metadata(),
        "rows": [],
        "upstream_revision": subprocess.check_output(
            ["git", "-C", ".research/frontier-flash-attention", "rev-parse", "HEAD"],
            text=True,
        ).strip(),
        "packages": {
            p: importlib.metadata.version(p)
            for p in ["nvidia-cutlass-dsl", "flash-attn-4", "quack-kernels"]
        },
    }
    path = Path("results/frontier/attention-cute.json")
    with FrontierEngine(policy="bf16-exact", max_graphs=1) as engine:
        tensors = capture_inputs(engine)

        def baseline():
            return [
                F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
                for q, k, v, mask in tensors
            ]

        expected = baseline()
        report["baseline_ms"], report["baseline_samples_ms"] = timing(baseline)
        for tile in [(64, 64), (64, 128), (128, 64), (128, 128)]:
            row = {"tile": tile}
            try:

                def candidate(tile=tile):
                    outputs = []
                    for q, k, v, mask in tensors:
                        used = (
                            None
                            if mask is None
                            else mask[:, 0, 0].sum(-1, dtype=torch.int32)
                        )
                        output = _flash_attn_fwd(
                            q.transpose(1, 2),
                            k.transpose(1, 2),
                            v.transpose(1, 2),
                            seqused_k=used,
                            tile_mn=tile,
                            _arch=120,
                        )[0]
                        outputs.append(output.transpose(1, 2))
                    return outputs

                actual = candidate()
                row["mismatches"] = sum(
                    int((a != b).sum()) for a, b in zip(actual, expected)
                )
                row["layer_mismatches"] = [
                    int((a != b).sum()) for a, b in zip(actual, expected)
                ]
                row["max_error"] = max(
                    float((a.float() - b.float()).abs().max())
                    for a, b in zip(actual, expected)
                )
                row["ms"], row["samples_ms"] = timing(candidate)
                row["speedup"] = report["baseline_ms"] / row["ms"]
            except Exception as error:  # noqa: BLE001 - retain failed experiment evidence
                row["error"] = str(error)
            report["rows"].append(row)
            path.write_text(json.dumps(report, indent=2) + "\n")
            print(row, flush=True)


if __name__ == "__main__":
    main()
