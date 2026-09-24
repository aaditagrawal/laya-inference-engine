"""Public mode routing without CUDA or optional native builds."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from laya_blackwell.cli import main
from laya_blackwell.modes import create_engine
from laya_blackwell.server import create_app


class Engine:
    backend = "fast"

    def __init__(self):
        self.closed = False
        self.requests = []

    def predict(self, state, questions):
        self.requests.append((state, questions))
        return {"answers": {}, "engine": {"backend": self.backend}}

    def close(self):
        self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def test_unknown_mode_fails_before_loading_an_engine():
    with pytest.raises(ValueError, match="mode must be balanced or fast"):
        create_engine(mode="typo")


def test_fast_cli_uses_public_engine_and_closes_it(monkeypatch, tmp_path, capsys):
    engine, options = Engine(), []
    monkeypatch.setitem(
        sys.modules,
        "laya_blackwell.fast",
        SimpleNamespace(FastEngine=lambda **kwargs: options.append(kwargs) or engine),
    )
    request = tmp_path / "request.json"
    request.write_text(json.dumps({"state": "hello", "questions": {}}))
    assert main(["predict", "--mode", "fast", "--request", str(request)]) == 0
    assert options == [
        {"model": "convaiinnovations/laya", "backend": "fused", "device": "cuda:0"}
    ]
    assert json.loads(capsys.readouterr().out)["engine"]["backend"] == "fast"
    assert engine.requests == [("hello", {})]
    assert engine.closed


def test_fast_server_routes_mode_and_owns_engine_lifetime(monkeypatch):
    engine, options = Engine(), []
    monkeypatch.setattr(
        "laya_blackwell.server._load_engine",
        lambda **kwargs: options.append(kwargs) or engine,
    )
    with TestClient(create_app(mode="fast", warmup=False)) as client:
        assert client.get("/health").json() == {"status": "ok", "backend": "fast"}
        response = client.post(
            "/v1/systemone", json={"state": "hello", "questions": {}}
        )
        assert response.status_code == 200
        assert not engine.closed
    assert options == [{"backend": "fused", "device": "cuda:0", "mode": "fast"}]
    assert engine.closed


def test_build_fast_cli_forwards_offline_sources_without_loading_engine(
    monkeypatch, capsys
):
    options = []
    monkeypatch.setitem(
        sys.modules,
        "laya_blackwell.fast.build",
        SimpleNamespace(
            build_all=lambda **kwargs: (
                options.append(kwargs)
                or {"directory": "/cache/laya/build", "artifacts": {"host": "host.so"}}
            )
        ),
    )
    assert (
        main(
            [
                "build-fast",
                "--cuda-home",
                "/toolkit",
                "--cutlass",
                "/headers/cutlass",
                "--flash-attention",
                "/headers/flash",
                "--cache-dir",
                "/cache/laya",
                "--offline",
            ]
        )
        == 0
    )
    assert options == [
        {
            "cuda_home": Path("/toolkit"),
            "cutlass": Path("/headers/cutlass"),
            "flash_attention": Path("/headers/flash"),
            "cache_dir": Path("/cache/laya"),
            "offline": True,
        }
    ]
    assert json.loads(capsys.readouterr().out) == {
        "ready": True,
        "cache": "/cache/laya/build",
        "native_libraries": 1,
    }
