"""Compilation policy that preserves the tested BF16 rounding contract."""

import hashlib
import os
from pathlib import Path

import torch
import triton


def pin_libdevice():
    """Call before native kernel calibration or first Triton compilation.

    An explicit environment selection is respected. Otherwise an existing
    Triton selection is retained, or the installed Triton library is pinned.
    Inductor may otherwise switch libraries when precision emulation starts,
    invalidating a previously calibrated sparse GELU correction table.
    """
    from triton import knobs

    selected = os.environ.get("TRITON_LIBDEVICE_PATH") or knobs.nvidia.libdevice_path
    if selected is None:
        selected = Path(triton.__file__).parent / "backends/nvidia/lib/libdevice.10.bc"
    path = Path(selected).resolve(strict=True)
    os.environ["TRITON_LIBDEVICE_PATH"] = str(path)
    knobs.nvidia.libdevice_path = str(path)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def install_compiler(engine, *, mode="default"):
    """Install after exact native kernels/window and before any graph capture.

    CUDA Graph allocation and replay remain owned by BlackwellEngine. First
    use compiles each distinct shape and is substantially slower than replay.
    """
    if engine.graphs:
        raise RuntimeError("Install the compiler before capturing any CUDA Graphs")
    if torch.__version__.split("+")[0].split(".")[:2] != ["2", "14"]:
        raise RuntimeError("Fast compilation is validated with PyTorch 2.14 only")
    if mode not in {"default", "max-autotune-no-cudagraphs"}:
        raise ValueError("Use default or max-autotune-no-cudagraphs")
    identity = pin_libdevice()
    from .preserve_ops import install as preserve_ops

    preserve_ops(engine)
    from torch._dynamo import config as dynamo_config
    from torch._inductor import config as inductor_config

    dynamo_config.recompile_limit = max(64, dynamo_config.recompile_limit)
    inductor_config.compile_threads = 2
    options = dict(torch._inductor.list_mode_options(mode))
    options.update({"triton.cudagraphs": False, "emulate_precision_casts": True})
    engine.model = torch.compile(
        engine.model, dynamic=False, fullgraph=True, options=options
    )
    engine.fast_compiler = {
        "mode": mode,
        "libdevice": identity,
        "precision_casts_preserved": True,
        "normalization_and_small_gelu_preserved": True,
    }
    return engine
