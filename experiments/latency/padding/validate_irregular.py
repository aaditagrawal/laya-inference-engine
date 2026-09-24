"""Exercise changed batch geometry across every four-question reference request."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from experiments.native import common
from laya_blackwell.validate import probabilities

from .engine import DenseEngine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--policy", choices=["batch-exact", "batch-compatible"], default="batch-exact"
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    report = {
        "metadata": common.metadata(),
        "policy": args.policy,
        "status": "running",
        "scope": "Repeat the four original question definitions cyclically to form 5 and 17 questions for each qualifying reference request. Compare newly evaluated current-engine outputs plus cached unbucketed SDK selected decisions. No fitting or calibration.",
    }
    rows = report["cases"] = []
    reference = np.load(common.CACHE / "outputs.npz", allow_pickle=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with (
        DenseEngine(
            model=common.model_path(), shape_policy="stock", max_graphs=4
        ) as baseline,
        DenseEngine(
            model=common.model_path(), shape_policy=args.policy, max_graphs=4
        ) as candidate,
    ):
        for count in (5, 17):
            for idx, original in enumerate(common.validation_requests()):
                if len(original["questions"]) != 4:
                    continue
                definitions = list(original["questions"].values())
                request = {
                    "state": original["state"],
                    "questions": {f"q{i}": definitions[i % 4] for i in range(count)},
                }
                prepared = baseline.prepare(**request)
                expected = baseline.run_prepared(prepared)
                actual = candidate.run_prepared(prepared)
                row = common.compare_outputs(
                    actual, expected, prepared, candidate.agent
                )
                original_prepared = baseline.prepare(**original)
                sdk_probs = probabilities(
                    reference[f"original_logits_{idx}"],
                    original_prepared,
                    baseline.agent,
                )
                actual_probs = probabilities(actual[0], prepared, candidate.agent)
                row.update(
                    original_case=idx,
                    questions=count,
                    request_sha256=hashlib.sha256(
                        json.dumps(request, sort_keys=True).encode()
                    ).hexdigest(),
                    unbucketed_sdk_agreement=sum(
                        int(p.argmax() == sdk_probs[i % 4].argmax())
                        for i, p in enumerate(actual_probs)
                    ),
                    baseline_shape=expected[2]["shape"],
                    candidate_shape=actual[2]["shape"],
                )
                rows.append(row)
                print(
                    idx,
                    count,
                    row["agreement"],
                    row["max_probability_error"],
                    row["passed"],
                    flush=True,
                )
                args.output.write_text(json.dumps(report, indent=2) + "\n")
    report["summary"] = {
        "requests": len(rows),
        "decisions": sum(r["decisions"] for r in rows),
        "agreement": sum(r["agreement"] for r in rows),
        "unbucketed_sdk_agreement": sum(r["unbucketed_sdk_agreement"] for r in rows),
        "action_argmax_agreement": sum(r["action_argmax_agreement"] for r in rows),
        "max_probability_error": max(r["max_probability_error"] for r in rows),
        "max_action_probability_error": max(
            r["max_action_probability_error"] for r in rows
        ),
        "max_action_logit_error": max(r["max_action_logit_error"] for r in rows),
        "exact_all": all(r["exact_logits_and_actions"] for r in rows),
        "passed": all(r["passed"] for r in rows),
    }
    report["status"] = "complete"
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
