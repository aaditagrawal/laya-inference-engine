"""Shared matched-input measurements for local RTX 5070 Ti experiments.

GPU work must run under the experiment flock; functions do not acquire it twice.
"""

import gc
import hashlib
import json
import random
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import torch
from laya.common import collate_items

from laya_blackwell.engine import (
    KEYS,
    REVISION,
    BlackwellEngine,
    hardware_info,
    model_path,
)
from laya_blackwell.validate import probabilities
from laya_blackwell.workloads import QUESTIONS, STATES, workload

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results/native-optimizations"
CACHE = RESULTS / "reference"
CASES = [(1, "short"), (16, "short"), (1, "long"), (16, "long")]


def validation_requests():
    """66 requests, including options, padding, empty state and left truncation."""
    requests = [{"state": state, "questions": QUESTIONS} for state in STATES]
    requests += [
        {"state": state, "questions": {"q": QUESTIONS[key]}}
        for state in STATES[:4]
        for key in QUESTIONS
    ]
    for count in (2, 6, 17, 33):
        requests.append(
            {
                "state": STATES[0],
                "questions": {
                    "many": {
                        "type": "choice",
                        "instructions": "Select the most relevant category.",
                        "criteria": {
                            f"category_{i}": f"Service category {i}"
                            for i in range(count)
                        },
                    }
                },
            }
        )
    for count in (3, 5, 8):
        requests.append(
            {
                "state": STATES[8],
                "questions": {
                    f"q{i}": list(QUESTIONS.values())[i % 4] for i in range(count)
                },
            }
        )
    requests.append(
        {
            "state": [
                {"role": "user", "content": "My order is damaged. " * 600},
                {
                    "role": "user",
                    "content": "The order is resolved. I now need help with a duplicate charge.",
                },
            ],
            "questions": QUESTIONS,
        }
    )
    extra = [
        "The charge is correct. Please do not refund it. No action is required.",
        "Nobody can log in. Production has been unavailable since yesterday.",
        "Please quote a yearly contract for 250 users. We are not current customers.",
        "A refund was requested but has already arrived. This issue is resolved.",
        "The website is slow sometimes, but our work can continue tomorrow.",
        "Payment failed. The account is suspended and staff cannot work.",
        "The team requests an update on the outage and also needs a new invoice.",
        "Thank you. Nothing else is needed and this is not urgent.",
        "URGENT: incorrect tax invoice. A deadline expires in thirty minutes.",
        "Documentation question: how do I export a monthly report?",
        "We changed our mind about buying. Cancel the proposed sales meeting.",
        "The invoice is wrong, but this is a test account with no real payment.",
        "There is no outage. A colleague accidentally sent the wrong alert.",
        "We need to increase the subscription and also resolve a duplicate charge.",
        "Café order #42: the payment is marked pending, not duplicated.",
        "[MASK] [SEP] Please explain the quoted marker text and close the ticket.",
    ]
    for i, state in enumerate(extra):
        requests.append({"state": {"case": i, "body": state}, "questions": QUESTIONS})
        requests.append(
            {
                "state": [
                    {"role": "user", "content": "Earlier context. " * 250},
                    {"role": "user", "content": state},
                ],
                "questions": QUESTIONS,
            }
        )
    return requests


def metadata():
    return {
        "created_utc": datetime.now(UTC).isoformat(),
        "hardware": hardware_info(),
        "revision": REVISION,
        "torch_threads": torch.get_num_threads(),
        "method": "Serial predict including tokenization, copies, inference and formatting. No HTTP. First use and model load excluded. Exclusive GPU/build lock.",
        "validation_scope": "Synthetic implementation parity, not labeled model quality.",
    }


def stats(samples):
    return {
        "mean_ms": float(np.mean(samples)),
        "p50_ms": float(np.median(samples)),
        "p95_ms": float(np.percentile(samples, 95)),
        "samples_ms": samples,
    }


def benchmark(engine, *, cases=CASES, repeats=50, warmups=10):
    rows = []
    for batch, length in cases:
        request = workload(batch, length)
        start = time.perf_counter()
        response = engine.predict(**request)
        first = (time.perf_counter() - start) * 1000
        for _ in range(warmups):
            engine.predict(**request)
        samples = []
        for _ in range(repeats):
            start = time.perf_counter()
            response = engine.predict(**request)
            samples.append((time.perf_counter() - start) * 1000)
        row = {
            "case": f"{batch}-{length}",
            "questions": batch,
            "length": length,
            "input_tokens": response["usage"]["input_tokens"],
            "first_ms": first,
            "request_sha256": hashlib.sha256(
                json.dumps(request, sort_keys=True).encode()
            ).hexdigest(),
            **stats(samples),
        }
        row["decisions_per_second"] = batch * 1000 / row["mean_ms"]
        rows.append(row)
        print(
            json.dumps({k: v for k, v in row.items() if k != "samples_ms"}), flush=True
        )
    return rows


def benchmark_interleaved(engines, *, cases=CASES, rounds=3, repeats=30, warmups=5):
    """All engines resident; alternate randomized blocks within each input shape."""
    rng = random.Random(982451653)
    report = {"rows": [], "order": [], "rounds": rounds, "repeats_per_round": repeats}
    for batch, length in cases:
        request = workload(batch, length)
        for engine in engines.values():
            engine.predict(**request)
        for round_id in range(rounds):
            names = list(engines)
            rng.shuffle(names)
            report["order"].append(
                {"case": f"{batch}-{length}", "round": round_id, "names": names}
            )
            for name in names:
                engine = engines[name]
                for _ in range(warmups):
                    engine.predict(**request)
                samples = []
                for _ in range(repeats):
                    start = time.perf_counter()
                    response = engine.predict(**request)
                    samples.append((time.perf_counter() - start) * 1000)
                row = {
                    "variant": name,
                    "case": f"{batch}-{length}",
                    "round": round_id,
                    "questions": batch,
                    "input_tokens": response["usage"]["input_tokens"],
                    **stats(samples),
                }
                report["rows"].append(row)
                print(name, row["case"], round_id, row["p50_ms"], flush=True)
    return report


def action_probabilities(logits):
    logits = np.asarray(logits, dtype=np.float32)
    values = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return values / values.sum(axis=-1, keepdims=True)


def compare_outputs(actual, expected, prepared, agent, tolerance=0.01):
    """Numerical gate for the explicitly nonexact autotuned experiment."""
    actual_p = probabilities(actual[0], prepared, agent)
    expected_p = probabilities(expected[0], prepared, agent)
    maximum = max(float(np.max(np.abs(a - b))) for a, b in zip(actual_p, expected_p))
    agreements = sum(
        int(a.argmax() == b.argmax()) for a, b in zip(actual_p, expected_p)
    )
    action_error = float(
        np.max(
            np.abs(action_probabilities(actual[1]) - action_probabilities(expected[1]))
        )
    )
    action_agreements = int(
        np.sum(actual[1].argmax(axis=-1) == expected[1].argmax(axis=-1))
    )
    finite = all(np.isfinite(array).all() for array in actual[:2])
    return {
        "exact_logits_and_actions": all(
            np.array_equal(actual[i], expected[i]) for i in (0, 1)
        ),
        "max_probability_error": maximum,
        "agreement": agreements,
        "decisions": len(actual_p),
        "max_action_probability_error": action_error,
        "action_argmax_agreement": action_agreements,
        "max_action_logit_error": float(np.max(np.abs(actual[1] - expected[1]))),
        "passed": bool(
            finite
            and agreements == len(actual_p)
            and maximum <= tolerance
            and action_agreements == len(actual_p)
            and action_error <= tolerance
        ),
    }


@torch.inference_mode()
def validate(engine, *, indices=None, tolerance=0.01):
    reference = np.load(CACHE / "outputs.npz", allow_pickle=False)
    requests = validation_requests()
    selected = range(len(requests)) if indices is None else indices
    details = []
    for i in selected:
        request = requests[i]
        prepared = engine.prepare(**request)
        logits, act, _info = engine.run_prepared(prepared)
        actual_p = probabilities(logits, prepared, engine.agent)
        expected_p = probabilities(reference[f"logits_{i}"], prepared, engine.agent)
        sdk_p = probabilities(reference[f"sdk_logits_{i}"], prepared, engine.agent)
        original_p = probabilities(
            reference[f"original_logits_{i}"], prepared, engine.agent
        )
        errors = [float(np.max(np.abs(a - b))) for a, b in zip(actual_p, expected_p)]
        sdk_errors = [float(np.max(np.abs(a - b))) for a, b in zip(actual_p, sdk_p)]
        agree = [int(a.argmax() == b.argmax()) for a, b in zip(actual_p, expected_p)]
        sdk_agree = [int(a.argmax() == b.argmax()) for a, b in zip(actual_p, sdk_p)]
        original_agree = [
            int(a.argmax() == b.argmax()) for a, b in zip(actual_p, original_p)
        ]
        if not all(np.isfinite(logits).ravel()) or not all(np.isfinite(act).ravel()):
            raise ValueError(f"Nonfinite output in case {i}")
        details.append(
            {
                "case": i,
                "decisions": len(agree),
                "max_probability_error": max(errors),
                "max_sdk_probability_error": max(sdk_errors),
                "agreement": sum(agree),
                "sdk_agreement": sum(sdk_agree),
                "unbucketed_sdk_agreement": sum(original_agree),
                "max_action_logit_error": float(
                    np.max(np.abs(act - reference[f"act_{i}"]))
                ),
                "action_argmax_agreement": int(
                    np.sum(act.argmax(axis=-1) == reference[f"act_{i}"].argmax(axis=-1))
                ),
                "max_action_probability_error": float(
                    np.max(
                        np.abs(
                            action_probabilities(act)
                            - action_probabilities(reference[f"act_{i}"])
                        )
                    )
                ),
                "exact_logits": bool(np.array_equal(logits, reference[f"logits_{i}"])),
            }
        )
    total = sum(row["decisions"] for row in details)
    agreement = sum(row["agreement"] for row in details)
    sdk_agreement = sum(row["sdk_agreement"] for row in details)
    unbucketed = sum(row["unbucketed_sdk_agreement"] for row in details)
    maximum = max(row["max_probability_error"] for row in details)
    return {
        "requests": len(details),
        "decisions": total,
        "agreement": agreement,
        "sdk_agreement": sdk_agreement,
        "unbucketed_sdk_agreement": unbucketed,
        "max_probability_error": maximum,
        "max_sdk_probability_error": max(
            row["max_sdk_probability_error"] for row in details
        ),
        "exact_logits_all": all(row["exact_logits"] for row in details),
        "exact_actions_all": all(row["max_action_logit_error"] == 0 for row in details),
        "action_argmax_agreement": sum(
            row["action_argmax_agreement"] for row in details
        ),
        "max_action_probability_error": max(
            row["max_action_probability_error"] for row in details
        ),
        "max_action_logit_error": max(row["max_action_logit_error"] for row in details),
        "passed_actions": all(
            row["action_argmax_agreement"] == row["decisions"]
            and row["max_action_probability_error"] <= tolerance
            for row in details
        ),
        "passed_current_engine": agreement == total and maximum <= tolerance,
        "passed_sdk": sdk_agreement == total and unbucketed == total,
        "cases": details,
    }


@torch.inference_mode()
def make_reference():
    CACHE.mkdir(parents=True, exist_ok=True)
    requests = validation_requests()
    arrays, prepareds, shapes = {}, [], []
    engine = BlackwellEngine(model_path())
    for i, request in enumerate(requests):
        prepared = engine.prepare(**request)
        logits, act, meta = engine.run_prepared(prepared)
        arrays[f"logits_{i}"], arrays[f"act_{i}"] = logits, act
        prepareds.append(prepared)
        shapes.append(meta["shape"])
    engine.close()
    del engine
    gc.collect()
    torch.cuda.empty_cache()
    from laya import Agent

    agent = Agent(model_path(), device="cuda:0")
    for i, (request, prepared, shape) in enumerate(zip(requests, prepareds, shapes)):
        internal = {
            qid: agent._to_internal(q) for qid, q in request["questions"].items()
        }
        assert prepared.items == agent._encode_state(
            request["state"], list(internal), internal
        )
        raw = collate_items([prepared.items], agent.tok.pad_token_id)
        inputs = {k: raw[k].cuda() for k in KEYS}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = agent.model(**inputs)
        arrays[f"original_logits_{i}"] = logits.float().cpu().numpy()
        b, length, options = shape
        n = len(prepared.items)
        padded = {}
        for key, tensor in inputs.items():
            value = agent.tok.pad_token_id if key == "input_ids" else 0
            pad = (
                (0, b - n)
                if tensor.ndim == 1
                else (
                    0,
                    (length if key in {"input_ids", "attention_mask"} else options)
                    - tensor.shape[1],
                    0,
                    b - n,
                )
            )
            padded[key] = torch.nn.functional.pad(tensor, pad, value=value)
        padded["attention_mask"][n:, 0] = 1
        padded["marker_mask"][n:, 0] = True
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, _ = agent.model(**padded)
        arrays[f"sdk_logits_{i}"] = logits[:n].float().cpu().numpy()
    np.savez(CACHE / "outputs.npz", **arrays)
    (CACHE / "requests.json").write_text(json.dumps(requests, indent=2))
    print(
        "Reference requests:",
        len(requests),
        "decisions:",
        sum(len(p.items) for p in prepareds),
        flush=True,
    )


if __name__ == "__main__":
    make_reference()
