"""Bounded exact-response checks and CPU timing under the exclusive experiment lock."""

import argparse
import hashlib
import json
import math
import random
import statistics
import time
import warnings
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from huggingface_hub.constants import HF_HUB_CACHE
from tokenizers import Tokenizer
from transformers import AutoTokenizer

from experiments.native.common import validation_requests
from laya_blackwell.engine import REVISION
from laya_blackwell.protocol import PreparedRequest, format_response

from .host_prepare import prepare
from .native_format import NativeFormatter


def synthetic(kind, count, rows=1):
    criteria = (
        {f"option_{i}": str(i) for i in range(count)}
        if kind == "choice"
        else [str(i) for i in range(count)]
    )
    return PreparedRequest(
        [f"q{i}" for i in range(rows)],
        [{"t": kind, "crit": criteria} for _ in range(rows)],
        [{"markers": list(range(count))} for _ in range(rows)],
        rows * 64,
    )


def outcome(function, args):
    with warnings.catch_warnings(record=True) as recorded:
        warnings.simplefilter("always")
        try:
            value = function(*args)
            return {
                "json": json.dumps(value, ensure_ascii=True, allow_nan=True),
                "warnings": [
                    (type(w.message).__name__, str(w.message)) for w in recorded
                ],
            }
        except (
            ValueError,
            TypeError,
            KeyError,
            IndexError,
            AttributeError,
            FloatingPointError,
        ) as error:
            return {
                "type": type(error).__name__,
                "error": str(error),
                "warnings": [
                    (type(w.message).__name__, str(w.message)) for w in recorded
                ],
            }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("results/frontier/native-format.json")
    )
    args = parser.parse_args()
    formatter = NativeFormatter()
    rng = np.random.default_rng(554209)
    report = {
        "created_utc": datetime.now(UTC).isoformat(),
        "numpy": np.__version__,
        "build": formatter.build,
        "scope": "CPU formatter only; exclusive experiment lock",
    }
    report["math"] = []
    for kind, values in [
        ("exp", rng.uniform(-80, 0, 200000).astype(np.float32)),
        ("log", np.exp(rng.uniform(-80, 0, 200000).astype(np.float32))),
    ]:
        expected = getattr(np, kind)(values)
        for use_libm in (False, True):
            actual = formatter.native.inspect_math(values, kind == "log", use_libm)
            report["math"].append(
                {
                    "function": kind,
                    "libm": use_libm,
                    "samples": len(values),
                    "bit_mismatches": int(
                        np.count_nonzero(
                            expected.view(np.uint32) != actual.view(np.uint32)
                        )
                    ),
                }
            )

    checks, failures, native_calls, fallback_calls = 0, [], 0, 0
    families = {}

    def check(values, family):
        nonlocal checks, native_calls, fallback_calls
        original_bytes = [
            value.tobytes() if isinstance(value, np.ndarray) else None
            for value in values
        ]
        expected = outcome(format_response, values)
        actual = outcome(formatter, values)
        checks += 1
        family_counts = families.setdefault(
            family, {"checks": 0, "native": 0, "fallback": 0}
        )
        family_counts["checks"] += 1
        if expected != actual:
            failures.append(
                {
                    "case": checks,
                    "family": family,
                    "expected": expected,
                    "actual": actual,
                }
            )
            if len(failures) <= 4:
                print(json.dumps(failures[-1]), flush=True)
        for value, original in zip(values, original_bytes):
            if original is not None and value.tobytes() != original:
                raise RuntimeError("Formatter modified raw input arrays")
        if "json" in expected and type(values[0]) is PreparedRequest:
            try:
                native = formatter.native(*values)
                native_calls += native is not None
                fallback_calls += native is None
                family_counts["native"] += native is not None
                family_counts["fallback"] += native is None
            except (ValueError, TypeError, KeyError, IndexError, AttributeError):
                fallback_calls += 1

    root = Path(HF_HUB_CACHE) / "models--convaiinnovations--laya/snapshots" / REVISION
    cfg = json.loads((root / "rl_agent_config.json").read_text())
    tok = AutoTokenizer.from_pretrained(root / "tokenizer", local_files_only=True)
    backend = Tokenizer.from_str(tok.backend_tokenizer.to_str())
    backend.no_padding()
    backend.no_truncation()
    temperature = cfg["temperature"]
    buckets = cfg["temperature_by_options"]
    reference = np.load("results/native-optimizations/reference/outputs.npz")
    for index, fixture in enumerate(validation_requests()):
        request = prepare(tok, backend, cfg, **fixture, mode="batch")
        check(
            (
                request,
                reference[f"logits_{index}"],
                reference[f"act_{index}"],
                temperature,
                buckets,
            ),
            "cached-reference",
        )

    shapes = [
        (kind, k)
        for kind in ("choice", "score")
        for k in (1, 2, 3, 4, 5, 7, 8, 9, 16, 17, 31, 32, 33, 64, 128, 255, 256)
    ] + [("noul", 2)]
    for kind, k in shapes:
        for count in (1, 2, 7, 16):
            request = synthetic(kind, k, count)
            for _ in range(8):
                logits = rng.normal(0, 4, (count, k + (k < 256))).astype(np.float32)
                actions = rng.normal(0, 4, (count, 2)).astype(np.float32)
                check(
                    (request, logits, actions, temperature, buckets), "random-size-type"
                )

    # Published probabilities and action values at half-decimal rounding thresholds.
    binary = synthetic("choice", 2)
    for integer in rng.integers(1, 9998, 1200):
        request = synthetic(("choice", "score", "noul")[integer % 3], 2)
        probability = (float(integer) + 0.5) / 10000
        center = np.float32(math.log(probability / (1 - probability)))
        values = [center]
        for direction in (-np.inf, np.inf):
            value = center
            for _ in range(3):
                value = np.nextafter(value, np.float32(direction))
                values.append(value)
        for value in values:
            logits = np.array([[0, value]], dtype=np.float32)
            actions = np.array([[value, 0]], dtype=np.float32)
            check(
                (request, logits, actions, [1.0] * 3, {}),
                "probability-rounding-boundary",
            )

    score_request = synthetic("score", 3)
    for integer in rng.integers(8001, 15998, 300):
        score = (float(integer) + 0.5) / 10000
        p2 = score - 0.8
        logits = np.log(np.array([[0.2, 0.8 - p2, p2]], np.float64)).astype(np.float32)
        center = logits[0, 2]
        for value in (
            center,
            np.nextafter(center, np.float32(-np.inf)),
            np.nextafter(center, np.float32(np.inf)),
        ):
            logits[0, 2] = value
            check(
                (score_request, logits, np.zeros((1, 2), np.float32), [1.0] * 3, {}),
                "score-rounding-boundary",
            )

    # Entropy confidence near half-decimal boundaries, with adjacent FP32 logits.
    for integer in rng.integers(1, 9998, 500):
        confidence = (float(integer) + 0.5) / 10000
        low, high = 0.5, 1.0 - 1e-12
        for _ in range(50):
            p = (low + high) * 0.5
            current = 1 + (p * math.log(p) + (1 - p) * math.log(1 - p)) / math.log(2)
            if current < confidence:
                low = p
            else:
                high = p
        logit = np.float32(math.log(p / (1 - p)))
        for value in (
            logit,
            np.nextafter(logit, np.float32(-np.inf)),
            np.nextafter(logit, np.float32(np.inf)),
        ):
            check(
                (
                    binary,
                    np.array([[0.0, value]], np.float32),
                    np.zeros((1, 2), np.float32),
                    [1.0] * 3,
                    {},
                ),
                "confidence-rounding-boundary",
            )

    for kind, count in (("choice", 4), ("score", 4), ("noul", 2)):
        request = synthetic(kind, count)
        for value in (0.0, 1.0, -1.0, 1e-18, -1e-18):
            for direction in (-np.inf, np.inf):
                logits = np.full((1, count), value, np.float32)
                logits[0, -1] = np.nextafter(np.float32(value), np.float32(direction))
                check(
                    (request, logits, np.zeros((1, 2), np.float32), [1.0] * 3, {}),
                    "argmax-boundary",
                )

    for temperatures, mapping in [
        ([None, np.inf, "bad"], {}),
        ([0.001, 99, np.nan], {}),
        ([1.0] * 3, {"choice:2": "bad"}),
        ([], {}),
        (None, {}),
        ([1.0] * 3, None),
    ]:
        check(
            (
                binary,
                np.array([[1, 2]], np.float32),
                np.zeros((1, 2), np.float32),
                temperatures,
                mapping,
            ),
            "temperature-and-error",
        )
    for logits in (
        None,
        [],
        [1, 2],
        [[1, 2]],
        np.ones((1, 1), np.float32),
        np.ones((1, 2), np.float64),
        np.ones((2, 4), np.float32)[:, ::2],
        np.array([[np.inf, -np.inf]], np.float32),
        np.array([[np.nan, 0]], np.float32),
        np.array([[0.0, -1000.0]], np.float32),
        np.array([[0.0, 1e-40]], np.float32),
    ):
        check(
            (binary, logits, np.zeros((1, 2), np.float32), [1.0] * 3, {}),
            "fallback-and-error",
        )
    check((PreparedRequest([], [], [], 0), None, None, None, None), "empty")
    for mode in ("ignore", "warn", "raise"):
        with np.errstate(under=mode):
            for rows in (1, 7, 16):
                request = synthetic("choice", 4, rows)
                logits = np.zeros((rows, 4), np.float32)
                for gap in (79.0, 88.0, 90.0, 100.0, 105.0, 4000.0):
                    actions = np.tile(
                        np.array([[0.0, -gap, 0.0, -gap]], np.float32), (rows, 1)
                    )
                    check((request, logits, actions, [1.0] * 3, {}), "action-underflow")
            for spread in (38.0, 39.0, 40.0, 80.0, 100.0):
                check(
                    (
                        binary,
                        np.array([[0.0, -spread]], np.float32),
                        np.array([[0.0, -4000.0]], np.float32),
                        [1.0] * 3,
                        {},
                    ),
                    "mixed-fallback-underflow",
                )
    callback_checks = []
    old_callback = np.geterrcall()
    try:
        for spread in (38.0, 39.0, 40.0, 80.0, 100.0):
            values = (
                binary,
                np.array([[0.0, -spread]], np.float32),
                np.array([[0.0, -4000.0]], np.float32),
                [1.0] * 3,
                {},
            )
            calls = {}
            for name, function in (
                ("reference", format_response),
                ("native", formatter),
            ):
                calls[name] = []
                np.seterrcall(
                    lambda error, flag, dest=calls[name]: dest.append((error, flag))
                )
                with np.errstate(under="call"):
                    function(*values)
            callback_checks.append(
                {
                    "spread": spread,
                    "calls": calls,
                    "exact": calls["reference"] == calls["native"],
                }
            )
    finally:
        np.seterrcall(old_callback)
    if not all(row["exact"] for row in callback_checks):
        failures.append({"family": "underflow-callback", "rows": callback_checks})
    print(f"Validated {checks} formatter cases, {len(failures)} failures", flush=True)

    request = synthetic("choice", 4)
    logits = np.array([[0.1, -0.3, 0.7, 2]], np.float32)
    actions = np.array([[1, 0.7]], np.float32)
    timing_args = (request, logits, actions, temperature, buckets)
    order_rng = random.Random(9884)
    timing = []
    functions = {"reference": format_response, "native": formatter}
    for round_id in range(9):
        order = list(functions)
        order_rng.shuffle(order)
        for name in order:
            samples = []
            for _ in range(2000):
                start = time.perf_counter_ns()
                functions[name](*timing_args)
                samples.append((time.perf_counter_ns() - start) / 1e6)
            timing.append(
                {
                    "variant": name,
                    "round": round_id,
                    "p50_ms": statistics.median(samples),
                    "samples_ms": samples,
                }
            )
    report.update(
        {
            "checks": checks,
            "native_calls": native_calls,
            "fallback_calls": fallback_calls,
            "families": families,
            "underflow_callbacks": callback_checks,
            "all_exact": not failures,
            "failures": failures,
            "timing": timing,
            "summary_ms": {
                name: statistics.median(
                    [
                        sample
                        for row in timing
                        if row["variant"] == name
                        for sample in row["samples_ms"]
                    ]
                )
                for name in functions
            },
            "source_sha256": {
                str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(Path("experiments/frontier").glob("native_format*"))
                if path.is_file()
            },
        }
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "math",
                    "checks",
                    "native_calls",
                    "fallback_calls",
                    "all_exact",
                    "summary_ms",
                )
            },
            indent=2,
        ),
        flush=True,
    )
    if failures:
        raise RuntimeError("Native response formatting differs")


if __name__ == "__main__":
    main()
