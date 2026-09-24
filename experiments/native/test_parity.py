"""CPU checks for the nonexact experimental mode's acceptance gate."""

from types import SimpleNamespace

import numpy as np

from .common import compare_outputs


def check(logits, action, tolerance=0.01):
    prepared = SimpleNamespace(items=[{"markers": [0, 1], "qtype": 0}])
    agent = SimpleNamespace(temperature_by_options={}, temperature={0: 1.0})
    expected = (
        np.array([[1.0, 0.0]], dtype=np.float32),
        np.array([[10000.0, -10000.0]], dtype=np.float32),
    )
    actual = tuple(np.array([values], dtype=np.float32) for values in (logits, action))
    return compare_outputs(actual, expected, prepared, agent, tolerance)


def test_small_drift_is_explicitly_nonexact():
    result = check([1.01, 0.0], [9990.0, -9990.0])
    assert result["passed"]
    assert not result["exact_logits_and_actions"]
    assert 0 < result["max_probability_error"] < 0.01
    assert result["max_action_logit_error"] == 10
    assert result["max_action_probability_error"] == 0


def test_changed_decision_is_rejected_even_with_loose_tolerance():
    assert not check([-1.0, 1.0], [10000.0, -10000.0], tolerance=1)["passed"]


def test_changed_action_is_rejected_even_with_loose_tolerance():
    assert not check([1.0, 0.0], [-10000.0, 10000.0], tolerance=1)["passed"]


def test_probability_drift_is_rejected_with_unchanged_argmax():
    assert not check([10.0, 0.0], [10000.0, -10000.0])["passed"]
