"""Explicitly build the pinned FastEngine extensions for Linux x86_64 / SM120."""

import argparse
import contextlib
import fcntl
import json
import os
import re
import shlex
import shutil
import subprocess
import sysconfig
import tempfile
import time
from pathlib import Path

from .paths import (
    CUTLASS_REVISION,
    FLASH_REVISION,
    NATIVE,
    artifact_filenames,
    build_directory,
    cache_root,
    fingerprint,
    require_build,
    sha256,
)

DEPENDENCIES = {
    "cutlass": (
        "https://github.com/NVIDIA/cutlass.git",
        CUTLASS_REVISION,
        "include/cutlass/cutlass.h",
    ),
    "flash-attention": (
        "https://github.com/Dao-AILab/flash-attention.git",
        FLASH_REVISION,
        "csrc/flash_attn/src/flash_fwd_launch_template.h",
    ),
}


@contextlib.contextmanager
def _build_lock(root):
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def _output(command):
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
    except FileNotFoundError as error:
        raise RuntimeError(
            f"Required build tool was not found: {command[0]}"
        ) from error
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"Build command failed: {shlex.join(command)}\n{error.stderr[-6000:]}"
        ) from error
    return result.stdout.strip()


def _cuda_directory(explicit=None):
    configured = explicit or os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    if configured:
        candidate = Path(configured).expanduser().resolve()
    else:
        nvcc = shutil.which("nvcc")
        candidate = (
            Path(nvcc).resolve().parent.parent
            if nvcc
            else Path("/usr/local/cuda").resolve()
        )
    if (
        not (candidate / "bin/nvcc").is_file()
        or not (candidate / "include/cuda_runtime_api.h").is_file()
    ):
        raise RuntimeError(
            "A CUDA toolkit with nvcc and headers is required. "
            "Set CUDA_HOME or pass --cuda-home."
        )
    return candidate


def _checkout(name, explicit, root, offline):
    url, revision, required_header = DEPENDENCIES[name]
    directory = (
        Path(explicit).expanduser().resolve()
        if explicit
        else root / "sources" / f"{name}-{revision}"
    )
    if not directory.exists():
        if explicit:
            raise RuntimeError(
                f"The supplied {name} checkout does not exist: {directory}"
            )
        if offline:
            raise RuntimeError(
                f"Offline setup needs {name} revision {revision} at {directory}, "
                "or a supplied local checkout. Run build-fast without --offline "
                "to fetch the pinned source."
            )
        directory.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(
            prefix=f".{name}-", dir=directory.parent
        ) as temporary:
            checkout = Path(temporary) / "checkout"
            _output(["git", "init", "--quiet", str(checkout)])
            _output(["git", "-C", str(checkout), "fetch", "--depth=1", url, revision])
            _output(
                [
                    "git",
                    "-C",
                    str(checkout),
                    "checkout",
                    "--quiet",
                    "--detach",
                    "FETCH_HEAD",
                ]
            )
            checkout.rename(directory)
    actual = _output(["git", "-C", str(directory), "rev-parse", "HEAD"])
    if actual != revision:
        raise RuntimeError(
            f"{name} must be checked out at {revision}; {directory} is at {actual}. "
            "The build does not change supplied checkouts."
        )
    if _output(
        ["git", "-C", str(directory), "status", "--porcelain", "--untracked-files=all"]
    ):
        raise RuntimeError(
            f"The pinned {name} checkout contains local changes: {directory}"
        )
    if not (directory / required_header).is_file():
        raise RuntimeError(
            f"The {name} checkout is missing {required_header}: {directory}"
        )
    return directory


def _run(command, log, commands):
    start = time.perf_counter()
    with log.open("a") as stream:
        stream.write(shlex.join(command) + "\n")
        stream.flush()
        try:
            result = subprocess.run(
                command, stdout=stream, stderr=subprocess.STDOUT, check=False
            )
        except FileNotFoundError as error:
            raise RuntimeError(
                f"Required build tool was not found: {command[0]}"
            ) from error
    commands.append(
        {
            "command": command,
            "seconds": time.perf_counter() - start,
            "returncode": result.returncode,
        }
    )
    if result.returncode:
        raise RuntimeError(f"Native build failed; compiler output is in {log}.")


def _dependency_hashes(directory):
    """Record the headers actually read by the compilers, including Torch headers."""
    paths = set()
    for depfile in directory.glob("*.d"):
        content = depfile.read_text().replace("\\\n", "")
        paths.update(Path(path) for path in shlex.split(content.split(":", 1)[1]))
    return {str(path.resolve()): sha256(path) for path in sorted(paths)}


def _compile(directory, cuda, cutlass, flash, configuration):
    import numpy
    import torch
    from torch.utils.cpp_extension import include_paths

    compiler = shlex.split(os.environ.get("CXX", "c++"))
    if not compiler:
        raise RuntimeError("CXX must name a C++ compiler.")
    nvcc = cuda / "bin/nvcc"
    nvcc_version = _output([str(nvcc), "--version"])
    version_match = re.search(r"release (\d+)\.(\d+)", nvcc_version)
    runtime_major = int(configuration["torch_cuda"].split(".")[0])
    if (
        version_match is None
        or int(version_match[1]) != runtime_major
        or (int(version_match[1]), int(version_match[2])) < (13, 1)
    ):
        raise RuntimeError(
            f"Use a CUDA {runtime_major}.x toolkit, version 13.1 or newer, matching "
            f"the pinned Torch runtime major version. Detected nvcc output: {nvcc_version}"
        )
    torch_root = Path(torch.__file__).resolve().parent
    torch_lib = torch_root / "lib"
    cuda_lib = next(
        (
            path
            for path in (
                cuda / "lib64",
                cuda / "targets/x86_64-linux/lib",
                cuda / "lib",
            )
            if (path / "libcudart.so").is_file()
        ),
        None,
    )
    if cuda_lib is None:
        raise RuntimeError(
            f"The CUDA toolkit has no development libcudart.so under {cuda}."
        )
    toolchain = {
        "cuda_home": str(cuda),
        "nvcc": nvcc_version,
        "nvcc_sha256": sha256(nvcc),
        "cxx": compiler,
        "cxx_version": _output([*compiler, "--version"]),
    }
    includes = ["-I" + path for path in include_paths(device_type="cpu")]
    includes += ["-I" + str(cuda / "include")]
    half_defines = [
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
    ]
    commands = []
    filenames = artifact_filenames()
    for name in filenames:
        print(f"Building fast {name}...", flush=True)
        output = directory / filenames[name]
        depfile = directory / f"{name}.d"
        log = directory / f"{name}.log"
        if name in {"host", "format"}:
            command = [
                *compiler,
                "-O3",
                "-DNDEBUG",
                "-std=c++17",
                "-shared",
                "-fPIC",
                "-MMD",
                "-MF",
                str(depfile),
                str(NATIVE / f"{name}.cpp"),
                "-I" + str(torch_root / "include"),
                "-I" + sysconfig.get_paths()["include"],
            ]
            if name == "format":
                command += [
                    "-ffp-contract=off",
                    "-fno-fast-math",
                    "-I" + numpy.get_include(),
                ]
            else:
                command += [
                    "-I" + str(cuda / "include"),
                    "-L" + str(cuda_lib),
                    "-Wl,-rpath," + str(cuda_lib),
                    "-lcudart",
                ]
            _run([*command, "-o", str(output)], log, commands)
            continue
        obj = directory / f"{name}.o"
        command = [
            str(nvcc),
            "-O3",
            "-std=c++20",
            "--expt-relaxed-constexpr",
            "-gencode=arch=compute_120,code=sm_120",
            "--compiler-options",
            "-fPIC",
            f"-D_GLIBCXX_USE_CXX11_ABI={configuration['cxx11_abi']}",
            "-MMD",
            "-MF",
            str(depfile),
            *includes,
        ]
        if name != "reduce_norm":
            command += half_defines
        if name == "vector":
            command += ["-lineinfo"]
        if name in {"attention", "attention_special", "global_attention"}:
            command += ["-I" + str(cutlass / "include")]
        if name == "global_attention":
            command += [
                "-DFLASH_NAMESPACE=laya_fast_global_flash",
                "-DFLASHATTENTION_DISABLE_DROPOUT",
                "-DUNFUSE_FMA",
                "-I" + str(flash / "csrc/flash_attn/src"),
            ]
        _run(
            [*command, "-c", str(NATIVE / f"{name}.cu"), "-o", str(obj)], log, commands
        )
        _run(
            [
                *compiler,
                str(obj),
                "-shared",
                "-L" + str(torch_lib),
                "-Wl,-rpath," + str(torch_lib),
                "-lc10",
                "-lc10_cuda",
                "-ltorch_cpu",
                "-ltorch_cuda",
                "-ltorch",
                "-ltorch_python",
                "-L" + str(cuda_lib),
                "-Wl,-rpath," + str(cuda_lib),
                "-lcudart",
                "-o",
                str(output),
            ],
            log,
            commands,
        )
    return {
        "complete": True,
        "fingerprint": configuration,
        "toolchain": toolchain,
        "dependencies": {
            name: {"revision": DEPENDENCIES[name][1], "path": str(path)}
            for name, path in (("cutlass", cutlass), ("flash-attention", flash))
        },
        "commands": commands,
        "compiler_inputs_sha256": _dependency_hashes(directory),
        "artifacts": filenames,
        "library_sha256": {
            name: sha256(directory / filename) for name, filename in filenames.items()
        },
        "fast_math": False,
        "flash_unfuse_fma": True,
    }


def build_all(
    cuda_home=None, cutlass=None, flash_attention=None, cache_dir=None, offline=False
):
    """Build all seven extensions, fetching pinned headers only when explicitly called.

    Existing valid builds are reused without needing a compiler or checkout.
    A process lock serializes builds in this user cache. Supplied checkouts must
    already match the exact revision and are never changed.
    """
    configuration = fingerprint()
    root = cache_root(cache_dir)
    directory = build_directory(root, configuration)
    with _build_lock(root):
        try:
            return require_build(root)
        except RuntimeError:
            pass
        cuda = _cuda_directory(cuda_home)
        cutlass_path = _checkout("cutlass", cutlass, root, offline)
        flash_path = _checkout("flash-attention", flash_attention, root, offline)
        directory.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{directory.name}-", dir=directory.parent)
        )
        try:
            manifest = _compile(
                temporary, cuda, cutlass_path, flash_path, configuration
            )
            if fingerprint() != configuration:
                raise RuntimeError(
                    "Fast native sources changed during compilation. Run build-fast again."
                )
            (temporary / "build.json").write_text(json.dumps(manifest, indent=2) + "\n")
            if directory.exists():
                # A stale or damaged cache may contain an old loaded library.
                # Keep its inode alive and publish the replacement atomically.
                retired = directory.with_name(
                    f".{directory.name}-retired-{os.getpid()}"
                )
                directory.rename(retired)
                try:
                    temporary.rename(directory)
                except OSError:
                    retired.rename(directory)
                    raise
                shutil.rmtree(retired)
            else:
                temporary.rename(directory)
        except Exception:
            # Keep the compiler logs available when setup fails.
            print(f"Native build files retained at {temporary}", flush=True)
            raise
        return require_build(root)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuda-home", type=Path)
    parser.add_argument("--cutlass", type=Path)
    parser.add_argument("--flash-attention", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--offline", action="store_true")
    args = parser.parse_args(argv)
    try:
        manifest = build_all(**vars(args))
    except RuntimeError as error:
        parser.exit(1, f"{error}\n")
    print(f"Fast native extensions ready at {manifest['directory']}", flush=True)


if __name__ == "__main__":
    main()
