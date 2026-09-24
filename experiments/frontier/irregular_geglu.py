"""Checked interface for independently compiled irregular tiles."""

import ctypes
import json

import torch

from .irregular_geglu_build import DIRECTORY, SOURCE, digest, key

LIBRARIES = {}


def load(config):
    name = key(config)
    if name not in LIBRARIES:
        report = json.loads((DIRECTORY / "build.json").read_text())
        row = next(r for r in report["rows"] if r["key"] == name)
        binary = DIRECTORY / (name + ".so")
        assert row["returncode"] == 0
        assert report["source_sha256"] == digest(SOURCE)
        assert row["library_sha256"] == digest(binary)
        library = ctypes.CDLL(str(binary))
        ptr, integer = ctypes.c_void_p, ctypes.c_int
        library.irregular_geglu.argtypes = [
            ptr,
            ptr,
            ptr,
            ptr,
            integer,
            ptr,
            ctypes.POINTER(integer),
        ]
        library.irregular_geglu.restype = integer
        LIBRARIES[name] = library
    return LIBRARIES[name]


def mapping(config):
    """Enumerate the direct epilogue's exact per-thread stores on the CPU."""
    bm, bn, wm, wn, _, direct = config
    if not direct:
        return {"kind": "default visitor with compile-time coverage assertions"}
    counts = [[0] * bn for _ in range(bm)]
    for warp in range((bm // wm) * (bn // wn)):
        warp_m, warp_n = warp % (bm // wm), warp // (bm // wm)
        for lane in range(32):
            for n in range(wn // 8):
                for m in range(wm // 16):
                    for r in range(2):
                        row = warp_m * wm + m * 16 + r * 8 + lane // 4
                        col = warp_n * wn + n * 8 + (lane % 4) * 2
                        counts[row][col] += 1
                        counts[row][col + 1] += 1
    values = [v for row in counts for v in row]
    assert min(values) == max(values) == 1
    return {
        "kind": "direct accumulator pairs",
        "elements": len(values),
        "minimum_writes": min(values),
        "maximum_writes": max(values),
    }


class Operation:
    def __init__(self, x, w, table, config, fused=True):
        assert x.shape == (1, 64, 1024) and w.shape == (5248, 1024)
        assert table.shape == (65536,)
        for value in (x, w, table):
            assert value.is_cuda and value.dtype == torch.bfloat16
            assert value.device == x.device and value.is_contiguous()
        self.x, self.w, self.table = x, w, table
        self.config, self.fused = config, int(fused)
        self.library = load(config)
        # NaN initialization makes unwritten finite outputs fail the exact gate.
        self.output = torch.full(
            (1, 64, 2624 if fused else 5248),
            float("nan"),
            device=x.device,
            dtype=x.dtype,
        )
        info = (ctypes.c_int * 5)()
        error = self.library.irregular_geglu(
            None, None, None, None, self.fused, None, info
        )
        assert error == 0, error
        bm, bn = config[:2]
        self.resources = dict(
            zip(
                (
                    "registers",
                    "shared_bytes",
                    "threads",
                    "active_blocks_per_sm",
                    "local_bytes",
                ),
                info,
            )
        )
        self.resources["ctas"] = ((64 + bm - 1) // bm) * ((5248 + bn - 1) // bn)

    def __call__(self):
        error = self.library.irregular_geglu(
            self.x.data_ptr(),
            self.w.data_ptr(),
            self.table.data_ptr(),
            self.output.data_ptr(),
            self.fused,
            torch.cuda.current_stream().cuda_stream,
            None,
        )
        if error:
            raise RuntimeError(f"Irregular CUTLASS launch failed: {error}")
        return self.output
