"""Opt-in tests of graph reuse, eviction, concurrency and owned outputs."""
from concurrent.futures import ThreadPoolExecutor
import os

import numpy as np
import pytest

from laya_blackwell.engine import BlackwellEngine
from laya_blackwell.workloads import STATES, QUESTIONS

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(
    os.environ.get("LAYA_RUN_MODEL_TESTS") != "1", reason="Set LAYA_RUN_MODEL_TESTS=1 to load the checkpoint"
)]


def test_graph_replay_eviction_and_threads():
    with BlackwellEngine(max_graphs=2) as engine:
        question = {"q": QUESTIONS["department"]}
        first = engine.prepare(STATES[0], question)
        logits, _, meta = engine.run_prepared(first)
        owned = logits.copy()
        assert meta["graph_miss"]
        assert not engine.run_prepared(first)[2]["graph_miss"]
        expected = [engine.predict(state, question)["answers"] for state in STATES[:3]]
        with ThreadPoolExecutor(max_workers=3) as pool:
            got = list(pool.map(lambda state: engine.predict(state, question)["answers"], STATES[:3] * 2))
        assert got == expected * 2
        engine.predict(STATES[8], question)
        engine.predict(STATES[9], question)
        assert len(engine.graphs) == 2
        replay, _, meta = engine.run_prepared(first)
        assert meta["graph_miss"]
        np.testing.assert_array_equal(logits, owned)
        np.testing.assert_array_equal(replay, owned)
    with pytest.raises(RuntimeError, match="closed"):
        engine.run_prepared(first)


def test_same_shape_with_and_without_padding_uses_distinct_graphs():
    with BlackwellEngine(max_graphs=2) as engine:
        question = {"q": QUESTIONS["department"]}
        full = engine.prepare(STATES[9], question)
        padded = engine.prepare(STATES[8] * 2, question)
        assert engine._shape(full) == engine._shape(padded)
        assert engine._graph_key(full)[-1] is True
        assert engine._graph_key(padded)[-1] is False
        expected = []
        for request in (full, padded):
            logits, _, metadata = engine.run_prepared(request)
            assert metadata["graph_miss"]
            expected.append(logits.copy())
        assert len(engine.graphs) == 2
        for request, logits in zip((full, padded), expected):
            replay, _, metadata = engine.run_prepared(request)
            assert not metadata["graph_miss"]
            np.testing.assert_array_equal(replay, logits)
