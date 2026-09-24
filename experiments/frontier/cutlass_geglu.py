"""Checked ctypes interface for the bounded native CUTLASS GEGLU screen."""

import ctypes
import json

import torch

from .cutlass_geglu_build import DIRECTORY, SOURCE, digest

TILES = [
    (32, 64, 3),
    (32, 64, 2),
    (32, 64, 4),
    (64, 64, 2),
    (64, 64, 3),
    (64, 64, 4),
    (32, 32, 3),
    (64, 32, 3),
    (32, 128, 2),
    (32, 128, 3),
    (64, 128, 2),
    (64, 128, 3),
]
LIBRARY = None


def load():
    global LIBRARY
    if LIBRARY is None:
        report = json.loads((DIRECTORY / "build.json").read_text())
        binary = DIRECTORY / "cutlass_geglu.so"
        if report["source_sha256"] != digest(SOURCE) or report[
            "library_sha256"
        ] != digest(binary):
            raise RuntimeError("Native CUTLASS source/binary changed; rebuild first")
        LIBRARY = ctypes.CDLL(str(binary))
        ptr, integer = ctypes.c_void_p, ctypes.c_int
        LIBRARY.cutlass_geglu.argtypes = [
            ptr,
            ptr,
            ptr,
            ptr,
            integer,
            integer,
            ptr,
            ctypes.POINTER(integer),
        ]
        LIBRARY.cutlass_geglu.restype = integer
    return LIBRARY


class Operation:
    def __init__(self, x, w, table, tile, fused):
        if (
            tile not in range(len(TILES))
            or x.shape != (1, 64, 1024)
            or w.shape != (5248, 1024)
        ):
            raise ValueError("Expected bounded MLPWi geometry")
        for value in [x, w, table]:
            if (
                not value.is_cuda
                or value.dtype != torch.bfloat16
                or value.device != x.device
                or not value.is_contiguous()
            ):
                raise ValueError("Expected contiguous BF16 tensors on one GPU")
        if table.shape != (65536,):
            raise ValueError("Expected complete BF16 GELU table")
        self.x, self.w, self.table = x, w, table
        self.tile, self.fused = tile, int(fused)
        self.output = torch.empty(
            (1, 64, 2624 if fused else 5248), device=x.device, dtype=x.dtype
        )
        self.library = load()
        info = (ctypes.c_int * 5)()
        error = self.library.cutlass_geglu(
            None, None, None, None, tile, self.fused, None, info
        )
        if error:
            raise RuntimeError(f"CUTLASS resource query failed: {error}")
        self.resources = dict(
            zip(
                [
                    "registers",
                    "shared_bytes",
                    "threads",
                    "active_blocks_per_sm",
                    "local_bytes",
                ],
                info,
            )
        )

    def __call__(self):
        error = self.library.cutlass_geglu(
            self.x.data_ptr(),
            self.w.data_ptr(),
            self.table.data_ptr(),
            self.output.data_ptr(),
            self.tile,
            self.fused,
            torch.cuda.current_stream().cuda_stream,
            None,
        )
        if error:
            raise RuntimeError(f"CUTLASS launch failed: {error}")
        return self.output
