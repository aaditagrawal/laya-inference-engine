"""Check setup failure modes without CUDA or a compiler."""

import json
import subprocess
import sys
import sysconfig
from pathlib import Path

import pytest

from laya_blackwell.fast import build, paths


@pytest.fixture
def built_cache(tmp_path, monkeypatch):
    configuration = {"source": "frozen", "python_soabi": "test"}
    monkeypatch.setattr(paths, "fingerprint", lambda: configuration)
    monkeypatch.setattr(paths, "_verified_manifest", None)
    directory = paths.build_directory(tmp_path, configuration)
    directory.mkdir(parents=True)
    filenames = paths.artifact_filenames()
    for filename in filenames.values():
        (directory / filename).write_bytes(b"stand-in for a compiled extension")
    manifest = {
        "complete": True,
        "fingerprint": configuration,
        "artifacts": filenames,
        "library_sha256": {
            name: paths.sha256(directory / filename)
            for name, filename in filenames.items()
        },
    }
    (directory / "build.json").write_text(json.dumps(manifest))
    return tmp_path, directory, manifest


def test_require_build_resolves_and_checks_every_artifact(built_cache):
    cache, directory, _ = built_cache
    manifest = paths.require_build(cache)
    for name, filename in paths.artifact_filenames().items():
        assert manifest["artifacts"][name] == str(directory / filename)
        assert paths.artifact_path(name) == directory / filename


@pytest.mark.parametrize(
    "mutation", ["corrupt", "missing", "source", "unfinished", "escape"]
)
def test_require_build_rejects_stale_or_damaged_setup(built_cache, mutation):
    cache, directory, manifest = built_cache
    filename = manifest["artifacts"]["attention"]
    if mutation == "corrupt":
        (directory / filename).write_bytes(b"changed binary")
    elif mutation == "missing":
        (directory / filename).unlink()
    elif mutation == "source":
        manifest["fingerprint"] = {"source": "old"}
    elif mutation == "unfinished":
        manifest["complete"] = False
    else:
        manifest["artifacts"]["attention"] = "../../another-library.so"
    (directory / "build.json").write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="build-fast"):
        paths.require_build(cache)


def test_offline_missing_headers_never_runs_git(tmp_path, monkeypatch):
    def no_command(_):
        pytest.fail("Offline missing-cache validation must not run git")

    monkeypatch.setattr(build, "_output", no_command)
    with pytest.raises(RuntimeError, match="Offline setup needs cutlass"):
        build._checkout("cutlass", None, tmp_path, offline=True)


@pytest.mark.parametrize(
    "revision,dirty",
    [("wrong-revision", ""), (paths.CUTLASS_REVISION, " M include/header.h")],
)
def test_local_headers_must_match_clean_pinned_checkout(
    tmp_path, monkeypatch, revision, dirty
):
    commands = []

    def git_output(command):
        commands.append(command)
        return revision if "rev-parse" in command else dirty

    monkeypatch.setattr(build, "_output", git_output)
    with pytest.raises(RuntimeError, match="must be checked out|local changes"):
        build._checkout("cutlass", tmp_path, tmp_path / "cache", offline=True)
    assert all(
        "fetch" not in command and "checkout" not in command for command in commands
    )


def test_verified_build_needs_no_compiler_or_header_checkout(built_cache, monkeypatch):
    cache, _, manifest = built_cache
    monkeypatch.setattr(build, "fingerprint", lambda: manifest["fingerprint"])

    def no_setup(*args, **kwargs):
        pytest.fail("A valid cached build must not compile or fetch headers")

    monkeypatch.setattr(build, "_cuda_directory", no_setup)
    monkeypatch.setattr(build, "_checkout", no_setup)
    monkeypatch.setattr(build, "_compile", no_setup)
    assert build.build_all(cache_dir=cache, offline=True)["complete"]


def test_dependency_manifest_handles_paths_with_spaces(tmp_path):
    header = tmp_path / "header with spaces.h"
    header.write_text("// header\n")
    escaped = str(header).replace(" ", "\\ ")
    (tmp_path / "kernel.d").write_text(f"kernel.o: \\\n {escaped}\n")
    assert build._dependency_hashes(tmp_path) == {str(header): paths.sha256(header)}


def test_cache_environment_precedence(tmp_path, monkeypatch):
    monkeypatch.setenv("LAYA_FAST_CACHE", str(tmp_path / "environment"))
    assert paths.cache_root() == tmp_path / "environment"
    assert paths.cache_root(tmp_path / "explicit") == tmp_path / "explicit"
    assert isinstance(paths.cache_root(), Path)


@pytest.mark.parametrize("packaged_first", [False, True])
def test_native_python_types_coexist_with_measured_reference(packaged_first):
    """The optional built-artifact check needs no CUDA context or model weights."""
    root = Path(__file__).resolve().parents[1]
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    originals = {
        "host": root / ".research/native-build/host" / f"laya_native_host{suffix}",
        "format": root
        / ".research/frontier-native-format"
        / f"laya_native_format{suffix}",
    }
    if not all(path.is_file() for path in originals.values()):
        pytest.skip("The optional measured reference extensions have not been built")
    try:
        manifest = paths.require_build()
    except RuntimeError as error:
        pytest.skip(
            f"The optional packaged native extensions are not available: {error}"
        )
    modules = []
    for prefix, artifacts in (
        ("laya_native", originals),
        ("laya_fast", manifest["artifacts"]),
    ):
        modules.extend(
            (f"{prefix}_{name}", str(artifacts[name])) for name in ("host", "format")
        )
    if packaged_first:
        modules.reverse()
    script = """
import importlib.util
import json
import sys
import torch

loaded = {}
for name, path in json.loads(sys.argv[1]):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    loaded[name] = module
for kind, exported in (("host", "Session"), ("format", "Formatter")):
    original = getattr(loaded["laya_native_" + kind], exported)
    packaged = getattr(loaded["laya_fast_" + kind], exported)
    assert original is not packaged
    assert original.__name__ == packaged.__name__ == exported
assert not torch.cuda.is_initialized()
"""
    result = subprocess.run(
        [sys.executable, "-c", script, json.dumps(modules)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
