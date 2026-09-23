"""CPU request preparation and responses compatible with the pinned Laya SDK.

Adapted from the Apache-2.0 Laya SDK with bounded validation, reusable request
preparation and NumPy response formatting. See NOTICE.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from laya.agent import Agent
from laya.common import (
    QTYPES,
    build_sequence,
    clamp_temperature,
    confidence_from_probs,
    render_options,
    temp_bucket,
)


@dataclass
class PreparedRequest:
    """Ordered questions and unpadded model inputs, before any device work."""

    ids: list[str]
    questions: list[dict[str, Any]]
    items: list[dict[str, Any]]
    input_tokens: int


def _positive_limit(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


def prepare_request(
    tok,
    cfg: dict[str, Any],
    state: str | dict | list,
    questions: dict[str, dict[str, Any]],
    *,
    max_questions: int = 64,
    max_options: int = 256,
    truncate_left: bool = False,
) -> PreparedRequest:
    """Validate and tokenize questions using the upstream sequence builder.

    Limits apply per request and per question respectively. Validate every
    definition before tokenizing, so a malformed late question cannot trigger
    unnecessary tokenization of the rest of the request.
    """
    _positive_limit("max_questions", max_questions)
    _positive_limit("max_options", max_options)
    if not isinstance(questions, dict):
        raise ValueError("questions must be a dict mapping question IDs to definitions")
    if len(questions) > max_questions:
        raise ValueError(
            f"request has {len(questions)} questions; max_questions={max_questions}"
        )
    ids = list(questions)
    if not ids:
        return PreparedRequest([], [], [], 0)

    max_len = cfg.get("max_len", 512)
    head_max_len = cfg.get("head_max_len", 192)
    _positive_limit("max_len", max_len)
    _positive_limit("head_max_len", head_max_len)

    normalized = []
    for qid in ids:
        definition = questions[qid]
        try:
            Agent._check_question(qid, definition)
        except TypeError as exc:
            raise ValueError(f"question {qid!r}: invalid definition: {exc}") from exc
        n_options = (
            2 if definition["type"] == "noul" else len(definition["criteria"])
        )
        if n_options > max_options:
            raise ValueError(
                f"question {qid!r} has {n_options} options; max_options={max_options}"
            )
        try:
            normalized.append(Agent._to_internal(definition))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"question {qid!r}: invalid definition: {exc}") from exc

    items = []
    for qid, q in zip(ids, normalized):
        seq, markers = build_sequence(
            tok, state, q, max_len, head_max_len, truncate_left=truncate_left
        )
        if len(markers) != len(render_options(q)):
            raise ValueError(
                f"question {qid!r} options exceed head_max_len={head_max_len} "
                f"or max_len={max_len}"
            )
        items.append({"ids": seq, "markers": markers, "qtype": QTYPES[q["t"]]})
    return PreparedRequest(ids, normalized, items, sum(len(it["ids"]) for it in items))


def format_response(
    prepared: PreparedRequest,
    logits: np.ndarray,
    act_logits: np.ndarray,
    temperature: list[float],
    temperature_by_options: dict[str, float],
) -> dict[str, Any]:
    """Convert raw CPU logits into Laya's calibrated, rounded response.

    Logits may contain extra padded rows or columns; only the prepared questions
    and their actual options are used. Action logits are uncalibrated, as in
    the upstream Agent.
    """
    result = {
        "model": "laya-rl-agent",
        "answers": {},
        "usage": {"input_tokens": prepared.input_tokens, "output_tokens": 0},
    }
    if not prepared.ids:
        return result

    n = len(prepared.ids)
    kmax = max(len(item["markers"]) for item in prepared.items)
    logits = np.asarray(logits, dtype=np.float32)
    act_logits = np.asarray(act_logits, dtype=np.float32)
    if logits.ndim != 2 or logits.shape[0] < n or logits.shape[1] < kmax:
        raise ValueError(f"logits must have at least shape ({n}, {kmax}), got {logits.shape}")
    if act_logits.ndim != 2 or act_logits.shape[0] < n or act_logits.shape[1] < 1:
        raise ValueError(f"act_logits must have at least shape ({n}, 1), got {act_logits.shape}")
    act_z = act_logits[:n] - act_logits[:n].max(axis=-1, keepdims=True)
    act = np.exp(act_z)
    act /= act.sum(axis=-1, keepdims=True)

    for row, (qid, q, item) in enumerate(
        zip(prepared.ids, prepared.questions, prepared.items)
    ):
        k = len(item["markers"])
        qt = QTYPES[q["t"]]
        t_scale = clamp_temperature(
            temperature_by_options.get(temp_bucket(qt, k), temperature[qt])
        )
        z = logits[row, :k] / t_scale
        p = np.exp(z - z.max())
        p /= p.sum()
        confidence = round(confidence_from_probs(p, k), 4)
        action = {"act_probability": round(float(act[row, 0]), 4)}

        if q["t"] == "choice":
            keys = list(q["crit"])
            answer = {
                "type": "choice",
                "choice": keys[int(p.argmax())],
                "probabilities": {
                    key: round(float(value), 4) for key, value in zip(keys, p)
                },
                "confidence": confidence,
                "action": action,
            }
        elif q["t"] == "score":
            answer = {
                "type": "score",
                "score": round(float((np.arange(k) * p).sum()), 4),
                "legend": {str(i): value for i, value in enumerate(q["crit"])},
                "probabilities": {
                    str(i): round(float(value), 4) for i, value in enumerate(p)
                },
                "confidence": confidence,
                "action": action,
            }
        else:
            answer = {
                "type": "noul",
                "noul": round(float(p[1]), 4),
                "confidence": round(max(float(p[1]), 1.0 - float(p[1])), 4),
                "action": action,
            }
        result["answers"][qid] = answer
    return result
