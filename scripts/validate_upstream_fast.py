"""Check the upstream fast mode against default SDK responses on benchmark inputs."""

from importlib.metadata import version
import json
from pathlib import Path

import numpy as np

from laya_blackwell.benchmark import load_backend
from laya_blackwell.engine import model_path, REVISION
from laya_blackwell.workloads import workload


def probability_vector(answer):
    if answer["type"] == "noul":
        return np.array([1 - answer["noul"], answer["noul"]])
    return np.array(list(answer["probabilities"].values()))


def main():
    path = model_path()
    stock, fast = load_backend("upstream", path), load_backend("upstream-fast", path)
    report = {
        "revision": REVISION,
        "packages": {p: version(p) for p in ("torch", "transformers", "laya", "tilelang")},
        "scope": "Default SDK versus optional fast SDK, full predict responses on the seven benchmark workloads plus a single-option request. Response probabilities are rounded by the SDK. Not task accuracy or a complete numerical validation.",
        "rows": [],
    }
    for batch, length in ((1, "short"), (4, "short"), (16, "short"), (1, "medium"),
                          (4, "medium"), (1, "long"), (16, "long"), (1, "single-option")):
        request = workload(batch, length)
        if length == "single-option":
            request["questions"] = {"q0": {"type": "choice", "instructions": "Choose.", "criteria": ["ticket"]}}
        row = {"questions": batch, "state_length": length}
        try:
            expected = stock.predict(**request)["answers"]
            actual = fast.predict(**request)["answers"]
            if stock.device.type != "cuda" or fast.device.type != "cuda":
                raise RuntimeError("Unexpected CPU fallback")
            differences, agreement = [], []
            for qid in expected:
                p, q = probability_vector(expected[qid]), probability_vector(actual[qid])
                differences.extend(np.abs(p - q).tolist())
                agreement.append(bool(p.argmax() == q.argmax()))
            row.update(max_probability_error=max(differences),
                       matching_top_options=sum(agreement), decisions=len(agreement))
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
        report["rows"].append(row)
        print(row, flush=True)
    good = [r for r in report["rows"] if "error" not in r]
    report["decisions"] = sum(r["decisions"] for r in good)
    report["matching_top_options"] = sum(r["matching_top_options"] for r in good)
    report["max_probability_error"] = max((r["max_probability_error"] for r in good), default=None)
    report["failed_requests"] = sum("error" in r for r in report["rows"])
    output = Path("results/validation-upstream-fast.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
