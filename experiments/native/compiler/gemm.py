"""Optional matrix autotuning; numerically changed from the exact backend."""

import types

import torch
import torch.nn.functional as F

from .policy import pin_libdevice


def install_gemm_autotune(engine, *, tma=True):
    """Autotune shared Linear shapes while retaining existing nonlinear kernels.

    Install after native kernels and window attention, before graph capture.
    This is an alternative to compiling the complete model. First use tunes
    each matrix shape: the initial 16-long screening took about 90 seconds.
    TMA candidates may compete, but the winning kernel need not use TMA.

    The full 66-request fixture matched 208 decisions with a maximum choice
    probability difference of 0.00432. Logits are not bit-exact; validate new
    workloads before choosing this mode over the exact native backend.
    """
    if engine.graphs:
        raise RuntimeError("Install GEMM autotuning before capturing any CUDA Graphs")
    if hasattr(engine.model, "_orig_mod"):
        raise RuntimeError("Choose GEMM autotuning or full-model compilation")
    if torch.__version__.split("+")[0].split(".")[:2] != ["2", "14"]:
        raise RuntimeError("This experiment is validated with PyTorch 2.14 only")
    identity = pin_libdevice()
    from torch._dynamo import config as dynamo_config
    from torch._inductor import config as inductor_config

    # One callable serves all Linear layers and all request shapes. A small
    # limit would reject valid shape specializations partway through testing.
    dynamo_config.recompile_limit = max(4096, dynamo_config.recompile_limit)
    dynamo_config.accumulated_recompile_limit = max(
        4096, dynamo_config.accumulated_recompile_limit
    )
    inductor_config.compile_threads = 2
    inductor_config.triton.enable_persistent_tma_matmul = bool(tma)
    inductor_config.triton.enable_template_tma_store = bool(tma)

    def linear(input, weight, bias):
        return F.linear(input, weight, bias)

    compiled_linear = torch.compile(
        linear, fullgraph=True, dynamic=False, mode="max-autotune-no-cudagraphs"
    )

    def forward(module, input):
        return compiled_linear(input, module.weight, module.bias)

    for module in engine.model.modules():
        if isinstance(module, torch.nn.Linear):
            module.forward = types.MethodType(forward, module)
    engine.experimental_compiler = {
        "mode": "shared-linear-max-autotune",
        "libdevice": identity,
        "tma_candidates_enabled": tma,
        "tma_selected": "Consult the autotuner records; flags do not prove selection",
        "exact_logits": False,
        "full_model_compilation": False,
        "recompile_limit": dynamo_config.recompile_limit,
    }
    return engine
