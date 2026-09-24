"""Validate completed comparison profiles and write JSON and CSV summaries."""

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path


SCOPE = (
    "Limited synthetic implementation comparison against the recorded CPU FP32 "
    "reference, using SDK-rounded responses. Matching top options and small "
    "numeric differences do not establish task accuracy, calibration or general "
    "numerical equivalence. Timings are serial warm requests; precision, threading "
    "and CPU microbatching follow each recorded profile."
)
TIMINGS = ("p50_ms", "p95_ms", "mean_ms", "decisions_per_second", "first_call_ms")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_values(value, path):
    if isinstance(value, dict):
        for key, item in value.items():
            finite_values(item, f"{path}/{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            finite_values(item, f"{path}/{index}")
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        require(math.isfinite(value), f"Nonfinite numeric value at {path}")


def compare_values(expected, actual, path=()):
    """Check matching response schemas and separate probability from other errors."""
    probability_error = other_error = 0.0
    categorical_mismatches = 0
    location = "/".join(path)
    if isinstance(expected, dict):
        require(isinstance(actual, dict) and expected.keys() == actual.keys(),
                f"Response keys differ at {location}")
        pairs = ((key, expected[key], actual[key]) for key in expected)
    elif isinstance(expected, list):
        require(isinstance(actual, list) and len(expected) == len(actual),
                f"Response list schema differs at {location}")
        pairs = ((str(i), a, b) for i, (a, b) in enumerate(zip(expected, actual)))
    else:
        if isinstance(expected, (int, float)) and not isinstance(expected, bool):
            require(isinstance(actual, (int, float)) and not isinstance(actual, bool),
                    f"Expected a numeric response at {location}")
            error = abs(expected - actual)
            if path[-2:-1] == ("probabilities",) or path[-1:] == ("noul",):
                probability_error = error
            else:
                other_error = error
        else:
            require(type(actual) is type(expected), f"Response type differs at {location}")
            categorical_mismatches = int(actual != expected)
        return probability_error, other_error, categorical_mismatches
    for key, left, right in pairs:
        probability, other, categorical = compare_values(left, right, (*path, key))
        probability_error = max(probability_error, probability)
        other_error = max(other_error, other)
        categorical_mismatches += categorical
    return probability_error, other_error, categorical_mismatches


def top_option(answer):
    if answer["type"] == "noul":
        return answer["noul"] >= 0.5
    require(answer["type"] in {"choice", "score"}, "Unknown response question type")
    probabilities = answer["probabilities"]
    require(bool(probabilities), "Empty response probability vector")
    return max(probabilities, key=probabilities.get)


def case_map(report, filename):
    require(report.get("status") == "complete", f"{filename}: profile is not complete")
    finite_values(report, str(filename))
    cases = report.get("cases", [])
    require(isinstance(cases, list) and bool(cases), f"{filename}: no cases")
    result = {}
    for case in cases:
        name = case["name"]
        require(name not in result, f"{filename}: duplicate case {name}")
        require(type(case["questions"]) is int and case["questions"] > 0,
                f"{filename}/{name}: invalid question count")
        require(len(case["answers"]) == case["questions"],
                f"{filename}/{name}: answer count differs from question count")
        result[name] = case
    return result


def summarize(directory):
    reference_path = directory / "reference.json"
    reference = json.loads(reference_path.read_text())
    require(reference.get("profile") == "reference", "reference.json is not a reference profile")
    reference_cases = case_map(reference, reference_path)
    require(bool(reference.get("model_revision")), "Reference has no checkpoint revision")
    profiles, csv_rows, seen = [], [], set()
    for filename in sorted(directory.glob("*.json")):
        if filename == reference_path or filename.name == "summary.json":
            continue
        report = json.loads(filename.read_text())
        if not isinstance(report, dict) or "profile" not in report:
            continue
        profile = report["profile"]
        require(profile not in seen, f"Duplicate profile {profile}")
        seen.add(profile)
        cases = case_map(report, filename)
        require(cases.keys() == reference_cases.keys(), f"{filename}: reference cases differ")
        require(report.get("model_revision") == reference["model_revision"],
                f"{filename}: checkpoint revision differs")
        options = report["options"]
        workers = options.get("workers", 1)
        metadata = {
            "profile": profile, "source_file": filename.name, "options": options,
            "threads": report["torch_threads"], "workers": workers,
            "threads_per_worker": options.get("threads") if workers > 1 else None,
            "model_load_ms": report["model_load_ms"],
        }
        rows = []
        for name, expected in reference_cases.items():
            actual = cases[name]
            for key in ("request_sha256", "input_tokens", "questions"):
                require(actual[key] == expected[key], f"{filename}/{name}: {key} differs")
            probability, other, categorical = compare_values(expected["answers"], actual["answers"])
            matching = sum(top_option(answer) == top_option(actual["answers"][qid])
                           for qid, answer in expected["answers"].items())
            for key in TIMINGS:
                require(type(actual[key]) in (int, float) and actual[key] >= 0,
                        f"{filename}/{name}: invalid {key}")
            row = {
                "case": name, "questions": actual["questions"],
                "request_sha256": actual["request_sha256"], "input_tokens": actual["input_tokens"],
                **{key: actual[key] for key in TIMINGS},
                "max_rounded_probability_error": probability, "max_other_numeric_error": other,
                "categorical_mismatch_count": categorical, "matching_top_options": matching,
                "decisions": len(expected["answers"]),
                "top_option_agreement": matching / len(expected["answers"]),
            }
            rows.append(row)
            csv_rows.append({**metadata, "options": json.dumps(options, sort_keys=True), **row})
        decisions = sum(row["decisions"] for row in rows)
        matching = sum(row["matching_top_options"] for row in rows)
        profiles.append({
            **metadata, "hardware": report["hardware"], "packages": report["packages"],
            "cuda_runtime": report["cuda_runtime"], "model_revision": report["model_revision"],
            "non_blackwell_benchmark_override": report["non_blackwell_benchmark_override"],
            "cases": rows, "decisions": decisions, "matching_top_options": matching,
            "top_option_agreement": matching / decisions,
            "max_rounded_probability_error": max(row["max_rounded_probability_error"] for row in rows),
            "max_other_numeric_error": max(row["max_other_numeric_error"] for row in rows),
            "categorical_mismatch_count": sum(row["categorical_mismatch_count"] for row in rows),
        })
    require(bool(profiles), "No completed comparison profiles found")
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(), "scope": SCOPE,
        "checks": "Completed profiles, checkpoint revision, fixture names, payload hashes, token counts, response schemas and finite numeric values validated.",
        "error_fields": "Probability error covers option probabilities and noul. Other numeric error covers score, confidence and action probability. Categorical mismatches count unequal nonnumeric response leaves.",
        "top_option_rule": "Choice and score use argmax of rounded option probabilities, resolving ties by response order. Noul uses probability >= 0.5. Published choice labels are also compared as categorical values.",
        "reference": {"file": reference_path.name, "model_revision": reference["model_revision"],
                      "hardware": reference["hardware"], "packages": reference["packages"],
                      "cases": len(reference_cases),
                      "decisions": sum(case["questions"] for case in reference_cases.values())},
        "profiles": profiles,
    }
    return summary, csv_rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=Path, default=Path("results/rtx-a6000"))
    args = parser.parse_args()
    try:
        summary, rows = summarize(args.directory)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.exit(1, f"Comparison summary failed: {exc}\n")
    (args.directory / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    with (args.directory / "summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Summarized {len(summary['profiles'])} profiles into {args.directory / 'summary.json'} and summary.csv")


if __name__ == "__main__":
    main()
