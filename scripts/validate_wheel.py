"""Install a wheel in isolation and check fast inference outside the checkout.

Requires the fast native build, downloaded model and results/fast-package.json.
Run with the same exclusive GPU lock as the package benchmark.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SMOKE = r"""
import importlib.abc
import json
import sys
from pathlib import Path

installed = Path(sys.argv[1]).resolve()
fixture = json.loads(Path(sys.argv[2]).read_text())
output = Path(sys.argv[3])
sys.path.insert(0, str(installed))

class NoExperiments(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "experiments" or fullname.startswith("experiments."):
            raise AssertionError("Installed runtime imported research code")

sys.meta_path.insert(0, NoExperiments())
import torch
torch.set_num_threads(4)
from laya_blackwell import create_engine
from laya_blackwell.server import create_app
from fastapi.testclient import TestClient

with create_engine(mode="fast") as engine:
    first = engine.predict(**fixture["request"])
    assert first.pop("engine")["backend"] == "fast"
    assert first == fixture["expected_response"]
    replay = engine.predict(**fixture["request"])
    assert not replay.pop("engine")["graph_miss"]
    assert replay == fixture["expected_response"]
    with TestClient(create_app(engine=engine)) as client:
        assert client.get("/health").json() == {"status": "ok", "backend": "fast"}
        response = client.post("/v1/systemone", json=fixture["request"])
        assert response.status_code == 200, response.text
        result = response.json()
        assert not result.pop("engine")["graph_miss"]
        assert result == fixture["expected_response"]
    assert not engine.closed, "HTTP app closed its caller-owned engine"
    fingerprint = engine.build["fingerprint"]
    native_hashes = engine.build["library_sha256"]
assert engine.closed
modules = {}
for name, module in tuple(sys.modules.items()):
    if name == "laya_blackwell" or name.startswith("laya_blackwell."):
        path = Path(module.__file__).resolve()
        assert path.is_relative_to(installed), (name, path)
        modules[name] = str(path.relative_to(installed))
assert not any(name == "experiments" or name.startswith("experiments.") for name in sys.modules)
output.write_text(json.dumps({
    "exact_public_response": True,
    "warm_graph_replay": True,
    "http_response_exact": True,
    "http_engine_ownership": True,
    "closed_cleanly": True,
    "no_experiment_imports": True,
    "all_package_imports_from_wheel": True,
    "modules": modules,
    "native_fingerprint": fingerprint,
    "native_library_sha256": native_hashes,
}, indent=2) + "\n")
"""


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wheel", type=Path)
    parser.add_argument(
        "--validation", type=Path, default=ROOT / "results/fast-package.json"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results/package-checks.json"
    )
    args = parser.parse_args()
    wheel = args.wheel.resolve()
    validation = json.loads(args.validation.read_text())
    assert validation["parity"]["all_exact"]
    with zipfile.ZipFile(wheel) as archive:
        contents = archive.namelist()
        required = {
            "laya_blackwell/fast/config.json",
            "laya_blackwell/fast/native/NOTICE.txt",
            "laya_blackwell/fast/native/pytorch-LICENSE.txt",
            "laya_blackwell/fast/native/cutlass-LICENSE.txt",
            "laya_blackwell/fast/native/flash-attention-LICENSE.txt",
            "laya_blackwell/fast/native/flash-attention-AUTHORS.txt",
        }
        required.update(
            f"laya_blackwell/fast/native/{name}"
            for name in (
                "host.cpp",
                "format.cpp",
                "vector.cu",
                "reduce_norm.cu",
                "attention.cu",
                "attention_special.cu",
                "global_attention.cu",
            )
        )
        assert required.issubset(contents), sorted(required.difference(contents))
        assert not any(
            name.startswith(("experiments/", "results/", ".research/"))
            for name in contents
        )
        packaged_hashes = {
            name: hashlib.sha256(archive.read(name)).hexdigest()
            for name in contents
            if name.startswith("laya_blackwell/")
        }
        for name, digest in packaged_hashes.items():
            measured = validation["source_before"].get(f"src/{name}")
            if name.endswith((".py", "/config.json")):
                assert measured == digest, (
                    f"Wheel differs from validated source: {name}"
                )
    with tempfile.TemporaryDirectory(prefix="laya-wheel-check-") as temporary:
        directory = Path(temporary)
        installed = directory / "installed"
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                sys.executable,
                "--no-deps",
                "--target",
                str(installed),
                str(wheel),
            ],
            check=True,
        )
        fixture = directory / "fixture.json"
        fixture.write_text(json.dumps(validation["parity"]["wheel_smoke"]))
        smoke_output = directory / "smoke.json"
        environment = dict(os.environ, HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
        subprocess.run(
            [
                sys.executable,
                "-c",
                SMOKE,
                str(installed),
                str(fixture),
                str(smoke_output),
            ],
            cwd=directory,
            env=environment,
            check=True,
            timeout=600,
        )
        smoke = json.loads(smoke_output.read_text())
        assert smoke["native_fingerprint"] == validation["native_build"]["fingerprint"]
        assert (
            smoke["native_library_sha256"]
            == validation["native_build"]["library_sha256"]
        )
    report = {
        "wheel": wheel.name,
        "wheel_sha256": sha256(wheel),
        "validation_sha256": sha256(args.validation),
        "script_sha256": sha256(Path(__file__)),
        "external_working_directory": True,
        "huggingface_offline": True,
        "matches_validated_source_and_native_build": True,
        "packaged_files_sha256": packaged_hashes,
        "smoke": smoke,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"wheel": wheel.name, "checks_passed": True}))


if __name__ == "__main__":
    main()
