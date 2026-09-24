"""Package preflight and opt-in replay/lifetime checks for the fixed fast mode."""

import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

from laya_blackwell import FastEngine
from laya_blackwell.fast import engine as fast_module
from laya_blackwell.workloads import QUESTIONS, STATES


@pytest.fixture
def no_model_or_gpu(monkeypatch):
    def forbidden(*_args, **_kwargs):
        pytest.fail("Preflight must reject this configuration before model/GPU work")

    monkeypatch.setattr(fast_module, "BlackwellEngine", forbidden)
    monkeypatch.setattr(fast_module, "hardware_info", forbidden)
    monkeypatch.setattr(fast_module.np, "__version__", "2.5.3")


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"model": "another/checkpoint"}, "pinned"),
        ({"revision": "main"}, "pinned"),
        ({"subfolder": "another-model"}, "pinned"),
        ({"backend": "fp8"}, "BF16"),
        ({"backend": "eager"}, "BF16"),
        ({"device": "cpu"}, "CUDA device"),
        ({"max_graphs": 0}, "positive"),
        ({"max_questions": 0}, "positive"),
    ],
)
def test_fast_rejects_unsupported_configuration(no_model_or_gpu, kwargs, message):
    with pytest.raises(ValueError, match=message):
        FastEngine(**kwargs)


def test_fast_rejects_unpinned_numpy_before_model(no_model_or_gpu, monkeypatch):
    monkeypatch.setattr(fast_module.np, "__version__", "2.5.4")
    with pytest.raises(RuntimeError, match="NumPy 2.5.3"):
        FastEngine()


def test_fast_rejects_other_blackwell_architecture_before_model(monkeypatch):
    monkeypatch.setattr(fast_module.np, "__version__", "2.5.3")
    monkeypatch.setattr(
        fast_module, "hardware_info", lambda *_: {"compute_capability": "10.0"}
    )
    monkeypatch.setattr(
        fast_module,
        "BlackwellEngine",
        lambda **_: pytest.fail("Unsupported hardware must fail before loading"),
    )
    with pytest.raises(RuntimeError, match="SM120"):
        FastEngine()


@pytest.mark.parametrize("arguments", [[], ["build-fast", "--help"]])
def test_fast_import_and_build_help_need_no_native_artifacts_or_gpu(
    tmp_path, arguments
):
    script = """
import sys
import torch

def forbidden(*args, **kwargs):
    raise AssertionError("Import/help must not initialize CUDA or load an extension")

torch.cuda._lazy_init = forbidden
torch.ops.load_library = forbidden
from laya_blackwell import FastEngine
from laya_blackwell.fast import paths
assert callable(FastEngine)
assert not any(name.startswith("experiments.") for name in sys.modules)
assert "laya_fast_host" not in sys.modules
assert "laya_fast_format" not in sys.modules
paths.require_build = forbidden
if sys.argv[1:]:
    from laya_blackwell.cli import main
    main(sys.argv[1:])
"""
    environment = dict(
        os.environ, LAYA_FAST_CACHE=str(tmp_path / "missing"), CUDA_VISIBLE_DEVICES=""
    )
    completed = subprocess.run(
        [sys.executable, "-c", script, *arguments],
        capture_output=True,
        text=True,
        env=environment,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert not (tmp_path / "missing").exists()
    if arguments:
        assert "--offline" in completed.stdout


@pytest.mark.gpu
@pytest.mark.skipif(
    os.environ.get("LAYA_RUN_FAST_TESTS") != "1",
    reason="Set LAYA_RUN_FAST_TESTS=1 after building the fast extensions",
)
def test_fast_graph_replay_masks_ownership_eviction_and_concurrency():
    """Exercise the installed runtime through its public request methods."""
    question = {"q": QUESTIONS["department"]}
    engine = FastEngine(max_graphs=3)
    with engine:
        prepared = [engine.prepare(state, question) for state in STATES[:3]]
        logits, actions, first = engine.run_prepared(prepared[0])
        owned_logits, owned_actions = logits.copy(), actions.copy()
        assert first["graph_miss"]
        assert first["backend"] == "fast"
        assert not engine.run_prepared(prepared[0])[2]["graph_miss"]
        expected = [engine.predict(state, question)["answers"] for state in STATES[:3]]
        other_logits, _, _ = engine.run_prepared(prepared[1])
        assert not np.array_equal(logits, other_logits)
        with ThreadPoolExecutor(max_workers=3) as pool:
            actual = list(
                pool.map(
                    lambda state: engine.predict(state, question)["answers"],
                    STATES[:3] * 3,
                )
            )
        assert actual == expected * 3

        full = engine.prepare(STATES[9], question)
        padded = engine.prepare(STATES[8] * 2, question)
        full_logits, _, full_metadata = engine.run_prepared(full)
        padded_logits, _, padded_metadata = engine.run_prepared(padded)
        assert full_metadata["shape"] == padded_metadata["shape"]
        assert len(full.items[0]["ids"]) == full_metadata["shape"][1]
        assert len(padded.items[0]["ids"]) < padded_metadata["shape"][1]
        assert full_metadata["graph_miss"] and padded_metadata["graph_miss"]
        for request, expected_logits in ((full, full_logits), (padded, padded_logits)):
            replay, _, metadata = engine.run_prepared(request)
            assert not metadata["graph_miss"]
            np.testing.assert_array_equal(replay, expected_logits)

        # Touch the original key, then three other keys to force its eviction.
        engine.run_prepared(prepared[0])
        engine.run_prepared(full)
        engine.run_prepared(padded)
        engine.predict(STATES[0], {"a": question["q"], "b": question["q"]})
        replay, replay_actions, metadata = engine.run_prepared(prepared[0])
        assert metadata["graph_miss"]
        np.testing.assert_array_equal(replay, owned_logits)
        np.testing.assert_array_equal(replay_actions, owned_actions)
        np.testing.assert_array_equal(logits, owned_logits)
        np.testing.assert_array_equal(actions, owned_actions)
    engine.close()
    with pytest.raises(RuntimeError, match="closed"):
        engine.run_prepared(prepared[0])
    with pytest.raises(RuntimeError, match="closed"):
        engine.predict(STATES[0], question)
