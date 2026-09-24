"""Check an AOT artifact on varied values within its fixed input specialization."""

import argparse
import json
from pathlib import Path

import torch

from experiments.native.compiler import (
    install_padded_rope_window,
    pin_libdevice,
    preserve_ops,  # noqa: F401
)
from experiments.native.host.adapter import packed_allocate
from experiments.native.kernels import install
from laya_blackwell.engine import KEYS, BlackwellEngine

from .build import FixedShapeModel
from .runtime import load_checked_package


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--variants", type=int, default=24)
    args = parser.parse_args()
    torch.set_num_threads(4)
    pin_libdevice()
    manifest = json.loads(args.package.with_suffix(".manifest.json").read_text())
    fixture = torch.load(args.package.with_suffix(".inputs.pt"), weights_only=True)
    engine = BlackwellEngine(max_graphs=1)
    install(engine, "cuda_vector_norm_triton_geglu_corrected")
    install_padded_rope_window(engine)
    native = FixedShapeModel(engine.model, manifest["unmasked"])
    loaded, metadata = load_checked_package(
        args.package, expected_torch=manifest["torch"]
    )
    _storage, views = packed_allocate(
        manifest["shape"], engine.agent.tok.pad_token_id, engine.device
    )
    inputs = tuple(views[key] for key in KEYS)
    generator = torch.Generator().manual_seed(230953344)
    rows = []
    try:
        with torch.inference_mode():
            for index in range(args.variants):
                values = [x.clone() for x in fixture["inputs"]]
                if index:
                    values[0][:, 7:15] = torch.randint(
                        100, 30000, values[0][:, 7:15].shape, generator=generator
                    )
                    values[4].fill_(index % 3)
                for target, source in zip(inputs, values):
                    target.copy_(source)
                expected = native(*inputs)
                actual = loaded(*inputs)
                torch.cuda.synchronize()
                rows.append(
                    {
                        "case": index,
                        "exact_logits": torch.equal(actual[0], expected[0]),
                        "exact_actions": torch.equal(actual[1], expected[1]),
                        "max_logit_error": (actual[0] - expected[0]).abs().max().item(),
                        "max_action_error": (actual[1] - expected[1])
                        .abs()
                        .max()
                        .item(),
                        "choice_agreement": int(
                            (actual[0].argmax(-1) == expected[0].argmax(-1))
                            .sum()
                            .item()
                        ),
                    }
                )
        report = {
            "scope": "Synthetic tensor-value parity for one fixed exported shape/mask; original request plus deterministic token and question-type perturbations. Not labeled model quality and not the full 66-request suite.",
            "loader": "checked-cpp",
            "target_metadata": metadata,
            "shape": manifest["shape"],
            "unmasked": manifest["unmasked"],
            "cases": rows,
            "count": len(rows),
            "all_exact_logits": all(r["exact_logits"] for r in rows),
            "all_exact_actions": all(r["exact_actions"] for r in rows),
            "max_logit_error": max(r["max_logit_error"] for r in rows),
            "max_action_error": max(r["max_action_error"] for r in rows),
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: v for k, v in report.items() if k != "cases"}), flush=True)
    finally:
        engine.close()


if __name__ == "__main__":
    main()
