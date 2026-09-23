"""HTTP validation, authentication, and lifecycle tests without a GPU."""

import json
import sys
import threading
from types import SimpleNamespace

from fastapi.testclient import TestClient
import pytest

from laya_blackwell.server import create_app


class FakeEngine:
    backend = "fused"
    device = "cuda:0"

    def __init__(self, error=None):
        self.error = error
        self.calls = []
        self.warmups = []
        self.closed = False

    def predict(self, state, questions):
        self.calls.append((state, questions, threading.current_thread().name))
        if self.error is not None:
            raise self.error
        return {
            "model": "laya-rl-agent",
            "answers": {},
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "engine": {"graph_miss": False, "total_ms": 0.2},
        }

    def warmup(self, state, questions):
        self.warmups.append((state, questions))
        return {"graph_miss": True, "graph_build_ms": 100}

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


@pytest.fixture(autouse=True)
def clear_api_key(monkeypatch):
    monkeypatch.delenv("LAYA_API_KEY", raising=False)


@pytest.fixture
def payload():
    return {"state": "Please help with my order.", "questions": {
        "needs_help": {"type": "noul", "instructions": "Does the customer need help?"}
    }}


@pytest.mark.parametrize("state", ["hello", {"text": "hello"}, [{"text": "hello"}]])
def test_valid_request_runs_in_worker_and_preserves_response(payload, state):
    engine = FakeEngine()
    payload["state"] = state
    with TestClient(create_app(engine)) as client:
        response = client.post("/v1/systemone", json=payload)
        assert response.status_code == 200
        assert response.json()["engine"] == {"graph_miss": False, "total_ms": 0.2}
        assert engine.calls[0][:2] == (state, payload["questions"])
        assert "worker" in engine.calls[0][2].lower()
        assert client.get("/health").json() == {"status": "ok", "backend": "fused"}
    assert not engine.closed
    assert engine.warmups == []


@pytest.mark.parametrize("payload", [
    None,
    {},
    {"state": None, "questions": {}},
    {"state": 123, "questions": {}},
    {"state": True, "questions": {}},
    {"state": "hi", "questions": []},
    {"state": "hi", "questions": {"q": "not a definition"}},
    {"state": "hi", "questions": {}, "typo": 1},
])
def test_malformed_request_is_422_before_engine_call(payload):
    engine = FakeEngine()
    with TestClient(create_app(engine)) as client:
        assert client.post("/v1/systemone", json=payload).status_code == 422
    assert engine.calls == []


def test_malformed_json_is_422():
    engine = FakeEngine()
    with TestClient(create_app(engine)) as client:
        response = client.post("/v1/systemone", content="{", headers={"Content-Type": "application/json"})
    assert response.status_code == 422
    assert engine.calls == []


def test_empty_questions_are_supported():
    with TestClient(create_app(FakeEngine())) as client:
        response = client.post("/v1/systemone", json={"state": "", "questions": {}})
    assert response.status_code == 200
    assert response.json()["usage"] == {"input_tokens": 0, "output_tokens": 0}


def test_engine_validation_error_is_422(payload):
    with TestClient(create_app(FakeEngine(ValueError("question 'q': unknown type")))) as client:
        response = client.post("/v1/systemone", json=payload)
    assert response.status_code == 422
    assert response.json()["detail"] == "question 'q': unknown type"


def test_runtime_errors_do_not_expose_internal_messages(payload):
    app = create_app(FakeEngine(RuntimeError("private checkpoint path /private/weights")))
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/v1/systemone", json=payload)
    assert response.status_code == 500
    assert "private" not in response.text
    assert response.text == "Internal Server Error"


@pytest.mark.parametrize("headers", [{}, {"Authorization": "Bearer wrong"}, {"Authorization": "Basic secret"}])
def test_api_key_rejects_unauthorized_requests(payload, headers):
    engine = FakeEngine()
    with TestClient(create_app(engine, api_key="secret")) as client:
        response = client.post("/v1/systemone", json=payload, headers=headers)
        assert client.get("/health").status_code == 200
    assert response.status_code == 401
    assert response.headers["WWW-Authenticate"] == "Bearer"
    assert engine.calls == []


def test_api_key_from_environment_and_explicit_override(monkeypatch, payload):
    monkeypatch.setenv("LAYA_API_KEY", "from-env")
    with TestClient(create_app(FakeEngine())) as client:
        assert client.post("/v1/systemone", json=payload).status_code == 401
        assert client.post("/v1/systemone", json=payload, headers={"Authorization": "Bearer from-env"}).status_code == 200
    with TestClient(create_app(FakeEngine(), api_key="explicit")) as client:
        assert client.post("/v1/systemone", json=payload, headers={"Authorization": "Bearer from-env"}).status_code == 401
        assert client.post("/v1/systemone", json=payload, headers={"Authorization": "Bearer explicit"}).status_code == 200


def test_owned_engine_loads_once_warms_and_closes(monkeypatch, payload):
    engine = FakeEngine()
    loads = []

    def load(**kwargs):
        loads.append(kwargs)
        return engine

    monkeypatch.setattr("laya_blackwell.server._load_engine", load)
    app = create_app(model="local/model", revision="commit", backend="eager", device="cuda:1")
    with TestClient(app) as client:
        assert len(engine.warmups) == 3
        for _ in range(2):
            assert client.post("/v1/systemone", json=payload).status_code == 200
        assert not engine.closed
    assert loads == [{"model": "local/model", "revision": "commit", "backend": "eager", "device": "cuda:1"}]
    assert engine.closed
    assert app.state.engine is None


def test_owned_engine_can_skip_warmup_and_keeps_default_revision(monkeypatch):
    engine = FakeEngine()
    loads = []
    monkeypatch.setattr("laya_blackwell.server._load_engine", lambda **kw: loads.append(kw) or engine)
    with TestClient(create_app(warmup=False)) as client:
        assert client.get("/health").status_code == 200
    assert loads == [{"backend": "fused", "device": "cuda:0"}]
    assert engine.warmups == []
    assert engine.closed


def test_owned_engine_closes_if_warmup_fails(monkeypatch):
    engine = FakeEngine()

    def fail(*args):
        raise RuntimeError("warmup failed")

    engine.warmup = fail
    monkeypatch.setattr("laya_blackwell.server._load_engine", lambda **kwargs: engine)
    with pytest.raises(RuntimeError, match="warmup failed"):
        with TestClient(create_app()):
            pass
    assert engine.closed


def test_closed_engine_is_unavailable(payload):
    engine = FakeEngine()
    engine.close()
    with TestClient(create_app(engine)) as client:
        assert client.get("/health").status_code == 503
        assert client.post("/v1/systemone", json=payload).status_code == 503
    assert engine.calls == []


def test_cli_predict_loads_one_engine_and_outputs_json(monkeypatch, tmp_path, capsys, payload):
    from laya_blackwell.cli import main

    engine = FakeEngine()
    options = []
    monkeypatch.setitem(sys.modules, "laya_blackwell.engine", SimpleNamespace(
        BlackwellEngine=lambda **kw: options.append(kw) or engine
    ))
    path = tmp_path / "request.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert main(["predict", "--request", str(path), "--backend", "eager",
                 "--device", "cuda:1", "--model", "local/model", "--revision", "commit"]) == 0
    assert options == [{"model": "local/model", "backend": "eager", "device": "cuda:1", "revision": "commit"}]
    assert json.loads(capsys.readouterr().out)["model"] == "laya-rl-agent"
    assert len(engine.calls) == 1
    assert engine.closed


def test_cli_info_prints_hardware(monkeypatch, capsys):
    from laya_blackwell.cli import main

    monkeypatch.setitem(sys.modules, "laya_blackwell.engine", SimpleNamespace(
        hardware_info=lambda device: {"device": device, "name": "Fake Blackwell"}
    ))
    assert main(["info", "--device", "cuda:2"]) == 0
    assert json.loads(capsys.readouterr().out) == {"device": "cuda:2", "name": "Fake Blackwell"}


def test_cli_serve_uses_loopback_and_loads_engine_in_lifespan(monkeypatch):
    from laya_blackwell.cli import main

    apps = []
    runs = []
    monkeypatch.setattr("laya_blackwell.server.create_app", lambda **kw: apps.append(kw) or "fake-app")
    monkeypatch.setitem(sys.modules, "uvicorn", SimpleNamespace(run=lambda app, **kw: runs.append((app, kw))))
    assert main(["serve", "--no-warmup"]) == 0
    assert apps == [{"model": "convaiinnovations/laya", "backend": "fused", "device": "cuda:0",
                     "api_key": None, "warmup": False}]
    assert runs == [("fake-app", {"host": "127.0.0.1", "port": 8000})]
