"""Owned mapped pinned aliases and graph-captured staging kernels.

The model still reads device-local inputs. Only these staging kernels access
mapped host memory. Callers retain this owner for the graph's entire lifetime
and synchronize its stream before reading or changing the CPU storage.
"""

import numpy as np
import torch
import triton as tr
import triton.language as tl
from cuda.bindings import runtime

from experiments.latency.serving.graph_adapter import _checked


class _CudaInterface:
    def __init__(self, host, pointer):
        # Torch retains this owner. It never points back at the alias tensor,
        # avoiding an owner/tensor reference cycle that would leak pinned pages.
        self.host = host
        self.__cuda_array_interface__ = {
            "shape": tuple(host.shape),
            "strides": None,
            "typestr": np.dtype(host.numpy().dtype).str,
            "data": (pointer, False),
            "version": 3,
        }


class MappedPinned:
    def __init__(self, host, device="cuda:0"):
        if (
            host.device.type != "cpu"
            or not host.is_pinned()
            or not host.is_contiguous()
        ):
            raise ValueError("Expected owned contiguous pinned CPU storage")
        self.host = host
        self.device = torch.device(device)
        with torch.cuda.device(self.device):
            self.device_pointer = int(
                _checked(runtime.cudaHostGetDevicePointer(host.data_ptr(), 0))
            )
            self.owner = _CudaInterface(host, self.device_pointer)
            self.alias = torch.as_tensor(self.owner, device=self.device)
        if self.alias.data_ptr() != self.device_pointer:
            raise RuntimeError("Torch copied mapped storage instead of aliasing it")
        if self.alias.dtype != host.dtype or self.alias.shape != host.shape:
            raise RuntimeError("Mapped alias changed storage dtype or geometry")


@tr.jit
def _stage(SOURCE, TARGET, N: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    values = tl.load(SOURCE + index, index < N, 0)
    tl.store(TARGET + index, values, index < N)


@tr.jit
def _collect(
    LOGITS, ACTIONS, HOST, NL: tl.constexpr, NA: tl.constexpr, BLOCK: tl.constexpr
):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    logits = tl.load(LOGITS + index, index < NL, 0)
    actions = tl.load(ACTIONS + index - NL, (index >= NL) & (index < NL + NA), 0)
    value = tl.where(index < NL, logits, actions)
    tl.store(HOST + index, value, index < NL + NA)


def stage(mapped, device_storage):
    if mapped.host.dtype != torch.uint8 or device_storage.dtype != torch.uint8:
        raise ValueError("Packed stage expects byte storage")
    if mapped.host.numel() != device_storage.numel() or device_storage.numel() % 8:
        raise ValueError("Packed stage sizes must match and align to eight bytes")
    source, target = mapped.alias.view(torch.int64), device_storage.view(torch.int64)
    _stage[(tr.cdiv(target.numel(), 128),)](
        source, target, target.numel(), 128, num_warps=4
    )


def collect(logits, actions, mapped):
    if (
        logits.dtype != torch.float32
        or actions.dtype != torch.float32
        or mapped.host.dtype != torch.float32
    ):
        raise ValueError("Output collection expects FP32 storage")
    if not logits.is_contiguous() or not actions.is_contiguous():
        raise ValueError("Output collection expects contiguous tensors")
    size = logits.numel() + actions.numel()
    if mapped.host.numel() != size:
        raise ValueError("Mapped output size differs from model outputs")
    # Copy integer bits, including signed zero and NaN payloads, without math.
    _collect[(tr.cdiv(size, 128),)](
        logits.view(torch.int32),
        actions.view(torch.int32),
        mapped.alias.view(torch.int32),
        logits.numel(),
        actions.numel(),
        128,
        num_warps=4,
    )
