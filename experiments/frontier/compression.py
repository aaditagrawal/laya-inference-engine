"""Explicit CUDA VMM ownership for a hardware-compression experiment.

Call close only after all GPU users, graphs and tensor views are finished.
No allocation is installed into a production model by this module.
"""

import math

import torch
import triton as tr
import triton.language as tl
from cuda.bindings import driver as cuda


def checked(result):
    status, *values = result
    if status != cuda.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"CUDA VMM: {status.name}")
    return values[0] if len(values) == 1 else tuple(values)


@tr.jit
def _copy(SRC, DST, N: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(DST + i, tl.load(SRC + i, i < N, 0), i < N)


class Allocation:
    def __init__(self, shape, dtype=torch.bfloat16, compressed=True):
        torch.cuda.init()
        checked(cuda.cuInit(0))
        self.handle = self.pointer = None
        self.mapped = False
        self.shape, self.dtype = tuple(shape), dtype
        self.bytes = math.prod(shape) * torch.empty((), dtype=dtype).element_size()
        device = torch.cuda.current_device()
        self.supported = checked(
            cuda.cuDeviceGetAttribute(
                cuda.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_GENERIC_COMPRESSION_SUPPORTED,
                device,
            )
        )
        prop = cuda.CUmemAllocationProp()
        prop.type = cuda.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
        prop.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        prop.location.id = device
        prop.allocFlags.compressionType = int(compressed)
        granularity = checked(
            cuda.cuMemGetAllocationGranularity(
                prop,
                cuda.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM,
            )
        )
        self.size = tr.cdiv(self.bytes, granularity) * granularity
        try:
            self.handle = checked(cuda.cuMemCreate(self.size, prop, 0))
            actual = checked(cuda.cuMemGetAllocationPropertiesFromHandle(self.handle))
            self.compression_granted = int(actual.allocFlags.compressionType)
            if compressed and not self.compression_granted:
                raise RuntimeError("Driver did not grant compressible allocation")
            self.pointer = checked(cuda.cuMemAddressReserve(self.size, 0, 0, 0))
            checked(cuda.cuMemMap(self.pointer, self.size, 0, self.handle, 0))
            self.mapped = True
            access = cuda.CUmemAccessDesc()
            access.location = prop.location
            access.flags = cuda.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
            checked(cuda.cuMemSetAccess(self.pointer, self.size, [access], 1))
            self.__cuda_array_interface__ = {
                "shape": (self.bytes,),
                "typestr": "|u1",
                "data": (int(self.pointer), False),
                "version": 3,
                "stream": int(torch.cuda.current_stream().cuda_stream) or 1,
            }
            self.tensor = torch.as_tensor(self, device="cuda").view(dtype).view(shape)
            if self.tensor.data_ptr() != int(self.pointer):
                raise RuntimeError("CUDA array interface unexpectedly copied memory")
        except BaseException:
            self.close()
            raise

    def copy(self, source):
        if source.numel() != self.tensor.numel() or not source.is_contiguous():
            raise ValueError("Expected matching contiguous source")
        # Populate through SM stores, explicitly including dtype conversion.
        _copy[(tr.cdiv(source.numel(), 1024),)](
            source, self.tensor, source.numel(), 1024
        )
        return self.tensor

    def close(self):
        if self.handle is None and self.pointer is None:
            return
        torch.cuda.synchronize()
        self.tensor = None
        if hasattr(self, "__cuda_array_interface__"):
            del self.__cuda_array_interface__
        if self.mapped:
            checked(cuda.cuMemUnmap(self.pointer, self.size))
            self.mapped = False
        if self.handle is not None:
            checked(cuda.cuMemRelease(self.handle))
            self.handle = None
        if self.pointer is not None:
            checked(cuda.cuMemAddressFree(self.pointer, self.size))
            self.pointer = None

    def report(self):
        return {
            "supported": self.supported,
            "compression_granted": self.compression_granted,
            "logical_bytes": self.bytes,
            "allocated_bytes": self.size,
            "dtype": str(self.dtype),
        }
