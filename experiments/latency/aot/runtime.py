"""Version-pinned C++ package loading without a deployment compiler probe."""

import platform
from pathlib import Path

import torch


def load_checked_package(package, *, expected_torch):
    """Check the artifact's target, then invoke its already-compiled C++ loader.

    PyTorch 2.14's high-level loader asks whether a C++ compiler can build AVX2
    code, even for an existing CUDA artifact. Deployment needs the target CPU
    instructions and compatible Torch/native libraries, not a new compiler.
    This local experiment checks those recorded target properties explicitly.
    It retains the Python runtime for registered LayerNorm and GELU operators.
    """
    if torch.__version__ != expected_torch:
        raise RuntimeError("AOT runtime must match the artifact's exact Torch version")
    if torch.__version__.split("+")[0].split(".")[:2] != ["2", "14"]:
        raise RuntimeError("This private package-loading API is pinned to Torch 2.14")
    loader_type = torch._C._aoti.AOTIModelPackageLoader
    path = str(Path(package).resolve())
    metadata = loader_type.load_metadata_from_package(path, "model")
    major, minor = torch.cuda.get_device_capability(0)
    target = {
        "AOTI_DEVICE_KEY": "cuda",
        "AOTI_PLATFORM": platform.system().lower(),
        "AOTI_MACHINE": platform.machine(),
        "AOTI_COMPUTE_CAPABILITY": f"{major}{minor}",
    }
    for name, value in target.items():
        if metadata.get(name) != value:
            raise RuntimeError(
                f"AOT target mismatch for {name}: {metadata.get(name)} != {value}"
            )
    if metadata.get("AOTI_CPU_ISA") != "AVX2":
        raise RuntimeError("This experiment validates only AVX2 host artifacts")
    if not torch.cpu._is_avx2_supported():
        raise RuntimeError("The deployed CPU does not expose usable AVX2")
    # AVX2 code generation also enables FMA and F16C. This experiment targets
    # Linux x86_64; require all compile-time CPU features rather than assuming.
    if target["AOTI_PLATFORM"] != "linux" or target["AOTI_MACHINE"] != "x86_64":
        raise RuntimeError("This checked loader targets Linux x86_64")
    flags = next(
        (
            line.split(":", 1)[1].split()
            for line in Path("/proc/cpuinfo").read_text().splitlines()
            if line.startswith("flags")
        ),
        [],
    )
    if not {"avx2", "fma", "f16c"}.issubset(flags):
        raise RuntimeError("Required AVX2/FMA/F16C CPU instructions are unavailable")
    from torch.export.pt2_archive._package import AOTICompiledModel

    return AOTICompiledModel(loader_type(path, "model", True, 1, 0)), metadata
