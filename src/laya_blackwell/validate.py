"""Numerical regression against upstream FP32-residual/BF16-autocast inference."""
import argparse
import gc
from importlib.metadata import version
import json
from pathlib import Path

import numpy as np
import torch
from laya import Agent
from laya.common import collate_items, temp_bucket

from .engine import BlackwellEngine, KEYS, model_path, REVISION
from .workloads import QUESTIONS, STATES


def probabilities(logits, prepared, agent):
    result = []
    for row, item in zip(logits, prepared.items):
        k = len(item["markers"])
        qt = item["qtype"]
        scale = agent.temperature_by_options.get(temp_bucket(qt, k), agent.temperature[qt])
        x = row[:k] / scale
        p = np.exp(x - x.max())
        result.append(p / p.sum())
    return result


@torch.inference_mode()
def run(backend="fused", tolerance=0.01):
    path = model_path()
    engine = BlackwellEngine(path, backend=backend)
    requests = [(state, QUESTIONS) for state in STATES]
    # Also exercise same-shape replay with changed IDs, question type and marker count.
    requests += [(state, {"q": QUESTIONS[key]})
                 for state in STATES[:4] for key in QUESTIONS]
    for count in (2, 6, 17, 33):
        requests.append((STATES[0], {"many": {
            "type": "choice", "instructions": "Select the most relevant category.",
            "criteria": {f"category_{i}": f"Service category {i}" for i in range(count)},
        }}))
    for count in (3, 5, 8):
        requests.append((STATES[8], {
            f"q{i}": list(QUESTIONS.values())[i % 4] for i in range(count)
        }))
    requests.append(([
        {"role": "user", "content": "My order is damaged. " * 600},
        {"role": "user", "content": "The order is resolved. I now need help with a duplicate charge."},
    ], QUESTIONS))
    prepared = [engine.prepare(state, questions) for state, questions in requests]
    fast = []
    for item in prepared:
        logits, act, metadata = engine.run_prepared(item)
        fast.append((logits.copy(), act.copy(), metadata["shape"]))
    engine.close()
    del engine
    gc.collect()
    torch.cuda.empty_cache()
    reference = Agent(path, device="cuda:0")
    # Compare SDK preprocessing independently. Reusing our own token IDs in
    # the numerical checks below cannot detect serialization/truncation bugs.
    for (state, questions), item in zip(requests, prepared):
        internal = {qid: reference._to_internal(q) for qid, q in questions.items()}
        expected = reference._encode_state(state, list(questions), internal)
        if item.items != expected:
            raise AssertionError("Engine tokenization differs from upstream SDK")
    errors, logit_errors, agreement, margins, action_errors = [], [], [], [], []
    unbucketed_errors, unbucketed_agreement, padding_errors = [], [], []
    rows = []
    for case, (item, (test_logits, test_act, shape)) in enumerate(zip(prepared, fast)):
        b = collate_items([item.items], reference.tok.pad_token_id)
        inputs = {key: b[key].cuda() for key in KEYS}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            unbucketed_logits, _ = reference.model(**inputs)
        unbucketed_p = probabilities(unbucketed_logits.float().cpu().numpy(), item, reference)
        # BF16 GEMM algorithms change with shape. Compare the kernels at the
        # same shape, and report the upstream SDK's padding drift separately.
        batch, length, options = shape
        n = len(item.items)
        padded = {}
        for key, tensor in inputs.items():
            value = reference.tok.pad_token_id if key == "input_ids" else 0
            if tensor.ndim == 1:
                padded[key] = torch.nn.functional.pad(tensor, (0, batch-n), value=value)
            else:
                width = length if key in {"input_ids", "attention_mask"} else options
                padded[key] = torch.nn.functional.pad(tensor, (0, width-tensor.shape[1], 0, batch-n), value=value)
        padded["attention_mask"][n:, 0] = 1
        padded["marker_mask"][n:, 0] = True
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits, act = reference.model(**padded)
        logits, act = logits[:n], act[:n]
        logits, act = logits.float().cpu().numpy(), act.float().cpu().numpy()
        ref_p, fast_p = probabilities(logits, item, reference), probabilities(test_logits, item, reference)
        for p, q, u in zip(ref_p, fast_p, unbucketed_p):
            unbucketed_errors.extend(np.abs(q-u).tolist())
            padding_errors.extend(np.abs(p-u).tolist())
            unbucketed_agreement.append(int(q.argmax() == u.argmax()))
        case_errors = []
        for i, (p, q) in enumerate(zip(ref_p, fast_p)):
            delta = np.abs(p-q)
            errors.extend(delta.tolist())
            logit_errors.extend(np.abs(logits[i, :len(p)]-test_logits[i, :len(p)]).tolist())
            agreement.append(int(p.argmax() == q.argmax()))
            margins.append(float(np.sort(p)[-1] - np.sort(p)[-2]) if len(p) > 1 else 1.)
            case_errors.append(float(delta.max()))
        action_errors.extend(np.abs(act-test_act).ravel().tolist())
        rows.append({"case": case, "questions": len(item.items), "tokens": item.input_tokens,
                     "max_probability_error": max(case_errors)})
    report = {"backend": backend, "revision": REVISION, "requests": len(prepared),
              "packages": {p: version(p) for p in ("torch", "triton", "transformers", "laya", "huggingface-hub", "tokenizers")},
              "decisions": len(agreement), "max_probability_error": max(errors),
              "sdk_preprocessing_match": True,
              "mean_probability_error": float(np.mean(errors)), "max_logit_error": max(logit_errors),
              "max_action_logit_error": max(action_errors), "argmax_agreement": float(np.mean(agreement)),
              "min_reference_top2_margin": min(margins), "probability_tolerance": tolerance,
              "passed": max(errors) <= tolerance and all(agreement) and all(unbucketed_agreement), "cases": rows,
              "reference_shape_policy": "Same batch, sequence and option padding as the engine",
              "max_probability_error_vs_unbucketed_sdk": max(unbucketed_errors),
              "argmax_agreement_vs_unbucketed_sdk": float(np.mean(unbucketed_agreement)),
              "upstream_own_max_probability_shift_from_padding": max(padding_errors),
              "scope": "Synthetic numerical regression, not labeled task accuracy or calibration validation. SDK preprocessing checked independently including long conversation lists. Unbucketed numbers compare upstream model forward at original shapes; they are not full SDK response comparisons."}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", default="fused")
    parser.add_argument("--tolerance", type=float, default=0.01)
    parser.add_argument("--output", default="results/validation.json")
    args = parser.parse_args()
    report = run(args.backend, args.tolerance)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
