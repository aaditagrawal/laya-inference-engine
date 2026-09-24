"""Explicit cuBLASLt plans retain descriptors and workspace through graph replay."""

import importlib.util
import sysconfig
from pathlib import Path

import torch

_native = None


def load():
    global _native
    if _native is None:
        path = (
            Path(__file__).resolve().parents[2]
            / ".research/frontier-build"
            / ("laya_frontier_lt" + sysconfig.get_config_var("EXT_SUFFIX"))
        )
        spec = importlib.util.spec_from_file_location("laya_frontier_lt", path)
        _native = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(_native)
    return _native


class Plan:
    def __init__(self, m, n, k):
        self.shape = m, n, k
        self.workspace = torch.empty(64 * 1024 * 1024, dtype=torch.uint8, device="cuda")
        self.native = load().Plan(m, n, k, self.workspace.numel())

    def __call__(self, x, weight, index=0):
        m, n, k = self.shape
        if (
            x.numel() != m * k
            or weight.shape != (n, k)
            or not x.is_contiguous()
            or not weight.is_contiguous()
        ):
            raise ValueError("Input does not match the cuBLASLt plan")
        output = torch.empty((*x.shape[:-1], n), dtype=torch.bfloat16, device=x.device)
        self.native.run(
            x.data_ptr(),
            weight.data_ptr(),
            output.data_ptr(),
            self.workspace.data_ptr(),
            self.workspace.numel(),
            torch.cuda.current_stream(x.device).cuda_stream,
            index,
        )
        return output
