"""Find and verify explicitly built native extensions without building anything."""

import hashlib
import json
import os
import platform
import sysconfig
from pathlib import Path

from .. import __version__

TORCH_REVISION = "08187d9e0fba026dc8217405802ab5381dc88d90"
CUTLASS_REVISION = "e05f953a5b3d38adc240df2ff928e0421c2abba3"
FLASH_REVISION = "14c377950125c70b7a9dabf9c561fca53715ac7d"
NUMPY_VERSION = "2.5.3"
SCHEMA_VERSION = 1
NATIVE = Path(__file__).resolve().parent / "native"
ARTIFACT_NAMES = (
    "host",
    "vector",
    "reduce_norm",
    "attention",
    "attention_special",
    "global_attention",
    "format",
)
_verified_manifest = None


def sha256(path):
    """Hash a file without loading a potentially large shared library at once."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def cache_root(cache_dir=None):
    """Return the user cache, with an explicit argument taking precedence."""
    configured = cache_dir or os.environ.get("LAYA_FAST_CACHE")
    if configured:
        return Path(configured).expanduser().resolve()
    base = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    return (base / "laya-blackwell" / "fast").expanduser().resolve()


def fingerprint():
    """Describe the sources and runtime ABI needed by this exact implementation."""
    import numpy
    import torch

    if platform.system() != "Linux" or platform.machine() != "x86_64":
        raise RuntimeError("Fast mode currently supports Linux x86_64 only.")
    if torch.version.git_version != TORCH_REVISION:
        raise RuntimeError(
            f"Fast mode requires Torch revision {TORCH_REVISION}; installed "
            f"revision is {torch.version.git_version}. Install the documented "
            "pinned CUDA 13.2 Torch build."
        )
    if numpy.__version__ != NUMPY_VERSION:
        raise RuntimeError(
            f"Fast mode requires numpy=={NUMPY_VERSION}; installed version is "
            f"{numpy.__version__}. Install the package's fast extra."
        )
    here = NATIVE.parent
    sources = [here / "build.py", here / "paths.py", *sorted(NATIVE.iterdir())]
    return {
        "schema": SCHEMA_VERSION,
        "package_version": __version__,
        "architecture": "sm_120",
        "torch": str(torch.__version__),
        "torch_git": torch.version.git_version,
        "torch_cuda": torch.version.cuda,
        "cxx11_abi": int(torch._C._GLIBCXX_USE_CXX11_ABI),
        "numpy": numpy.__version__,
        "python_soabi": sysconfig.get_config_var("SOABI"),
        "machine": platform.machine(),
        "cutlass_revision": CUTLASS_REVISION,
        "flash_revision": FLASH_REVISION,
        "sources_sha256": {
            str(path.relative_to(here)): sha256(path)
            for path in sources
            if path.is_file()
        },
    }


def build_directory(cache_dir=None, configuration=None):
    configuration = configuration if configuration is not None else fingerprint()
    encoded = json.dumps(configuration, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(encoded.encode()).hexdigest()
    return cache_root(cache_dir) / __version__ / digest


def artifact_filenames():
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    return {
        name: f"laya_fast_{name}{suffix if name in {'host', 'format'} else '.so'}"
        for name in ARTIFACT_NAMES
    }


def require_build(cache_dir=None):
    """Verify source, ABI and binary hashes, then return resolved artifact paths.

    This reads existing files only. The compiler and downloaded header checkouts
    are not needed after setup. A missing or stale build requires an explicit
    ``laya-blackwell build-fast`` command.
    """
    global _verified_manifest
    configuration = fingerprint()
    directory = build_directory(cache_dir, configuration)
    manifest_path = directory / "build.json"
    instruction = "Run `laya-blackwell build-fast` for this environment."
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, ValueError) as error:
        raise RuntimeError(
            f"Fast mode has no valid native build at {directory}. {instruction}"
        ) from error
    if (
        not isinstance(manifest, dict)
        or manifest.get("fingerprint") != configuration
        or manifest.get("complete") is not True
        or manifest.get("artifacts") != artifact_filenames()
        or not isinstance(manifest.get("library_sha256"), dict)
    ):
        raise RuntimeError(f"Fast mode's native build is stale. {instruction}")
    for name, filename in manifest["artifacts"].items():
        path = directory / filename
        try:
            valid = sha256(path) == manifest["library_sha256"].get(name)
        except OSError:
            valid = False
        if not valid:
            raise RuntimeError(
                f"Fast mode's {name} library is missing or changed. {instruction}"
            )
    resolved = {
        **manifest,
        "directory": str(directory),
        "cache_root": str(cache_root(cache_dir)),
        "artifacts": {
            name: str(directory / filename)
            for name, filename in manifest["artifacts"].items()
        },
    }
    _verified_manifest = resolved
    return resolved


def artifact_path(name):
    """Resolve an artifact from the last verified build, checking setup if needed."""
    if name not in ARTIFACT_NAMES:
        raise ValueError(f"Unknown fast native artifact: {name}")
    manifest = _verified_manifest
    if manifest is None:
        manifest = require_build()
    return Path(manifest["artifacts"][name])
