"""Request and response parity without checkpoint downloads or a GPU."""

import numpy as np
import pytest
import torch
from laya.agent import Agent
from laya.common import QTYPES, build_sequence, clamp_temperature

from laya_blackwell.protocol import format_response, prepare_request


class TinyTokenizer:
    mask_token = "[MASK]"
    mask_token_id = 3
    cls_token_id = 1
    sep_token_id = 2
    pad_token_id = 0

    def __init__(self):
        self.calls = 0

    def __call__(self, text, *, add_special_tokens=False):
        assert not add_special_tokens
        self.calls += 1
        return {"input_ids": [ord(char) + 10 for char in text]}


@pytest.fixture
def questions():
    return {
        "route": {
            "type": "choice",
            "instructions": "Pick [MASK] a route",
            "criteria": {"yes": {"meaning": "accept"}, "no": False, "later": 0},
        },
        "quality": {
            "type": "score",
            "instructions": {"task": "rate quality"},
            "criteria": ["poor", {"meaning": "fine"}, "excellent"],
        },
        "ready": {
            "type": "noul",
            "instructions": "Can we proceed?",
            "criteria": {False: "wait", True: ["ready", "now"]},
        },
        "single": {
            "type": "choice",
            "instructions": "Only one option",
            "criteria": ["only"],
        },
    }


def test_preparation_matches_upstream_sequences(questions):
    tok = TinyTokenizer()
    cfg = {"max_len": 128, "head_max_len": 80}
    state = [{"speaker": "user", "text": "hello [MASK] नमस्ते " * 30}]
    prepared = prepare_request(tok, cfg, state, questions)
    assert prepared.ids == list(questions)
    assert len(prepared.items) == len(questions)
    for index, qid in enumerate(questions):
        internal = Agent._to_internal(questions[qid])
        ids, markers = build_sequence(tok, state, internal, 128, 80)
        assert prepared.questions[index] == internal
        assert prepared.items[index] == {
            "ids": ids,
            "markers": markers,
            "qtype": QTYPES[internal["t"]],
        }
        assert all(ids[pos] == tok.mask_token_id for pos in markers)
    assert prepared.input_tokens == sum(len(item["ids"]) for item in prepared.items)


@pytest.mark.parametrize("state_kind", ["list", "dict", "str"])
def test_long_state_truncation_matches_upstream_agent(state_kind):
    """Conversation lists retain the newest turn; strings and dicts retain the start."""
    from laya_blackwell.engine import BlackwellEngine

    old, newest = "earlier context " * 100, "latest request Ω"
    state = {
        "list": [{"text": old}, {"text": newest}],
        "dict": {"earlier": old, "latest": newest},
        "str": old + newest,
    }[state_kind]
    questions = {"route": {"type": "choice", "instructions": "Pick", "criteria": ["a", "b"]}}
    agent = Agent.__new__(Agent)
    agent.tok = TinyTokenizer()
    agent.cfg = {"max_len": 512, "head_max_len": 192}
    normalized = {key: Agent._to_internal(value) for key, value in questions.items()}
    expected = agent._encode_state(state, list(questions), normalized)

    engine = BlackwellEngine.__new__(BlackwellEngine)
    engine.agent = agent
    engine.max_questions = 64
    actual = engine.prepare(state, questions)

    assert actual.items == expected
    assert actual.input_tokens == 512
    assert (ord("Ω") + 10 in actual.items[0]["ids"]) == (state_kind == "list")


def test_empty_request_does_not_tokenize_or_read_logits():
    tok = TinyTokenizer()
    prepared = prepare_request(tok, {}, object(), {})
    assert prepared.ids == prepared.questions == prepared.items == []
    assert tok.calls == 0
    assert format_response(prepared, None, None, [], {}) == {
        "model": "laya-rl-agent",
        "answers": {},
        "usage": {"input_tokens": 0, "output_tokens": 0},
    }


@pytest.mark.parametrize(
    "bad, message",
    [
        (None, "definition must be a dict"),
        ({"type": "unknown", "instructions": "?"}, "unknown type"),
        ({"type": "choice", "criteria": ["a"]}, "no 'instructions'"),
        ({"type": "choice", "instructions": "?", "criteria": []}, "at least one"),
        ({"type": "score", "instructions": "?", "criteria": {}}, "as a list"),
        ({"type": "noul", "instructions": "?", "criteria": []}, "as a dict"),
        ({"type": [], "instructions": "?"}, "invalid definition"),
        ({"type": "choice", "instructions": "?", "criteria": [["a"]]}, "invalid definition"),
    ],
)
def test_invalid_late_question_is_rejected_before_tokenization(bad, message):
    tok = TinyTokenizer()
    with pytest.raises(ValueError, match=message):
        prepare_request(
            tok,
            {},
            "state",
            {"valid": {"type": "noul", "instructions": "?"}, "broken": bad},
        )
    assert tok.calls == 0


def test_request_and_option_limits_reject_before_tokenization():
    tok = TinyTokenizer()
    question = {"type": "choice", "instructions": "?", "criteria": ["a", "b"]}
    with pytest.raises(ValueError, match="max_questions=64"):
        prepare_request(tok, {}, "", {str(i): question for i in range(65)})
    with pytest.raises(ValueError, match="max_options=256"):
        prepare_request(tok, {}, "", {"big": {**question, "criteria": list(range(257))}})
    with pytest.raises(ValueError, match="max_questions=1"):
        prepare_request(tok, {}, "", {"a": question, "b": question}, max_questions=1)
    with pytest.raises(ValueError, match="max_options=1"):
        prepare_request(tok, {}, "", {"a": question}, max_options=1)
    assert tok.calls == 0
    prepared = prepare_request(tok, {}, "", {"a": question, "b": question}, max_questions=2)
    assert len(prepared.items) == 2


@pytest.mark.parametrize("questions", [[], None, "bad"])
def test_questions_must_be_a_mapping(questions):
    with pytest.raises(ValueError, match="questions must be a dict"):
        prepare_request(TinyTokenizer(), {}, "", questions)


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_limits_must_be_positive_integers(limit):
    with pytest.raises(ValueError, match="max_questions must be a positive integer"):
        prepare_request(TinyTokenizer(), {}, "", {}, max_questions=limit)


def test_truncated_option_markers_are_rejected():
    with pytest.raises(ValueError, match="question 'route' options exceed"):
        prepare_request(
            TinyTokenizer(),
            {"max_len": 16, "head_max_len": 16},
            "",
            {"route": {"type": "choice", "instructions": "?", "criteria": list("abcdef")}},
        )


class FixedModel:
    def __init__(self, logits, act_logits):
        self.logits = torch.tensor(logits, dtype=torch.float32)
        self.act_logits = torch.tensor(act_logits, dtype=torch.float32)

    def __call__(self, *args):
        return self.logits, self.act_logits


@pytest.mark.parametrize(
    "temperatures, buckets",
    [
        ([1.0, 1.0, 1.0], {}),
        ([0.01, 80.0, float("nan")], {"choice:3-5": 0.1006, "noul:2": 2.125}),
        ([None, float("inf"), "bad"], {"choice:2": float("-inf"), "score:3-5": 99}),
    ],
)
def test_response_matches_upstream_including_temperature_and_rounding(
    questions, temperatures, buckets
):
    tok = TinyTokenizer()
    state = {"text": "a decision"}
    logits = np.array([[1.01234, -0.32345, 0.721], [0.12, 2.131, -1.2],
                       [-0.125, 0.8456, -1e4], [0.0, -1e4, -1e4]], dtype=np.float32)
    act_logits = np.array([[0.1, 0.8], [-0.99, 0.001], [0.731, 0.144], [0.0, 0.0]])
    prepared = prepare_request(tok, {}, state, questions)
    agent = Agent.__new__(Agent)
    agent.cfg = {}
    agent.tok = tok
    agent.device = torch.device("cpu")
    agent.dtype = torch.float32
    agent.temperature = [clamp_temperature(t) for t in temperatures]
    agent.temperature_by_options = {key: clamp_temperature(t) for key, t in buckets.items()}
    agent.model = FixedModel(logits, act_logits)
    expected = agent.system_one(state, questions)
    actual = format_response(prepared, logits, act_logits, temperatures, buckets)
    assert actual == expected
    assert actual["answers"]["single"]["probabilities"] == {"only": 1.0}
    assert actual["answers"]["single"]["confidence"] == 1.0
    assert actual["usage"]["output_tokens"] == 0


def test_padded_rows_and_columns_do_not_affect_output():
    prepared = prepare_request(TinyTokenizer(), {}, "", {"q": {"type": "noul", "instructions": "?"}})
    actual = format_response(prepared, [[0.0, 0.0, 999], [999, 999, 999]],
                             [[1e4, 1e4], [999, 999]], [1.0] * 3, {})
    assert actual["answers"]["q"] == {
        "type": "noul", "noul": 0.5, "confidence": 0.5,
        "action": {"act_probability": 0.5},
    }


@pytest.mark.parametrize("logits, act", [([0, 1], [[0, 1]]), ([[0]], [[0, 1]]), ([[0, 1]], [])])
def test_invalid_model_output_shapes_raise(logits, act):
    prepared = prepare_request(TinyTokenizer(), {}, "", {"q": {"type": "noul", "instructions": "?"}})
    with pytest.raises(ValueError, match="logits must have at least shape"):
        format_response(prepared, logits, act, [1.0] * 3, {})
