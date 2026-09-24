"""Local nvCOMPDx SDK interface and reversible checkpoint byte transforms."""

import ctypes
import json
import time

import torch
import triton as tr
import triton.language as tl

from .nvcompdx_build import DIRECTORY, SOURCE, sha

LIBRARY = None
INFO_KEYS = [
    "max_compressed_chunk_bytes",
    "compression_shared_scratch_bytes",
    "compression_global_scratch_per_chunk_bytes",
    "decode_shared_scratch_bytes",
    "decode_global_scratch_per_chunk_bytes",
    "compression_input_alignment",
    "compression_output_alignment",
    "decode_input_alignment",
    "decode_output_alignment",
    "decode_shared_alignment",
    "chunk_bytes",
    "block_threads",
    "decode_registers",
    "decode_static_shared_bytes",
    "decode_local_bytes",
    "active_blocks_per_sm",
]


def check(code):
    if code:
        raise RuntimeError(f"Native nvCOMPDx CUDA error {code}")


def load():
    global LIBRARY
    if LIBRARY is None:
        report = json.loads((DIRECTORY / "probe-build.json").read_text())
        library = DIRECTORY / "nvcompdx_ans.so"
        if report["source_sha256"] != sha(SOURCE) or report["library_sha256"] != sha(
            library
        ):
            raise RuntimeError("Native source or binary changed; rebuild first")
        LIBRARY = ctypes.CDLL(str(library))
        p, i, u = ctypes.c_void_p, ctypes.c_int, ctypes.c_ulonglong
        LIBRARY.nvdx_info.argtypes = [i, i, i, p]
        LIBRARY.nvdx_compress.argtypes = [i, i, i, p, p, u, p, p, i, p]
        LIBRARY.nvdx_compact.argtypes = [p, u, p, p, p, i, p]
        LIBRARY.nvdx_checksum.argtypes = [i, i, i, p, p, p, p, p, p, i, i, p]
    return LIBRARY


@tr.jit
def _transform(
    X,
    Y,
    ELEMENTS: tl.constexpr,
    PER_MATRIX: tl.constexpr,
    MODE: tl.constexpr,
    REVERSE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    offset = (i // PER_MATRIX) * (PER_MATRIX * 2) + i % PER_MATRIX
    if REVERSE:
        a = tl.load(X + offset, i < ELEMENTS, 0).to(tl.uint16)
        b = tl.load(X + offset + PER_MATRIX, i < ELEMENTS, 0).to(tl.uint16)
        if MODE == 1:
            bits = a | (b << 8)
        else:
            bits = (a & 127) | ((a >> 7) << 15) | (b << 7)
        tl.store(Y + i, bits, i < ELEMENTS)
    else:
        bits = tl.load(X + i, i < ELEMENTS, 0).to(tl.uint16)
        if MODE == 1:
            a, b = bits & 255, bits >> 8
        else:
            a, b = (bits & 127) | ((bits >> 15) << 7), (bits >> 7) & 255
        tl.store(Y + offset, a, i < ELEMENTS)
        tl.store(Y + offset + PER_MATRIX, b, i < ELEMENTS)


def transform(bank, mode, reverse=False):
    if mode not in [0, 1, 2] or bank.dtype != torch.uint8 or not bank.is_contiguous():
        raise ValueError("Invalid reversible byte transform")
    if mode == 0:
        return bank
    result = torch.empty_like(bank)
    elements = bank.numel() // 2
    _transform[(tr.cdiv(elements, 256),)](
        bank if reverse else bank.view(torch.uint16),
        result.view(torch.uint16) if reverse else result,
        elements,
        5248 * 1024,
        mode,
        reverse,
        256,
    )
    return result


class Bank:
    def __init__(self, encoded, chunk, block, half=False):
        if chunk not in [4096, 16384] or block not in [128, 256]:
            raise ValueError("Unsupported bounded configuration")
        if (
            encoded.dtype != torch.uint8
            or not encoded.is_cuda
            or not encoded.is_contiguous()
            or encoded.numel() % chunk
        ):
            raise ValueError("Expected contiguous chunk-aligned CUDA byte bank")
        self.library = load()
        self.raw, self.chunk, self.block, self.half = encoded, chunk, block, int(half)
        self.count = encoded.numel() // chunk
        values = (ctypes.c_ulonglong * len(INFO_KEYS))()
        check(self.library.nvdx_info(chunk, block, int(half), values))
        self.info = dict(zip(INFO_KEYS, values))
        align = max(
            self.info["compression_output_alignment"],
            self.info["decode_input_alignment"],
        )
        stride = tr.cdiv(self.info["max_compressed_chunk_bytes"], align) * align
        start_wall = time.perf_counter()
        temporary = torch.empty(
            self.count * stride, device=encoded.device, dtype=torch.uint8
        )
        scratch_bytes = (
            self.count * self.info["compression_global_scratch_per_chunk_bytes"]
        )
        scratch = torch.empty(
            max(scratch_bytes, 1), device=encoded.device, dtype=torch.uint8
        )
        self.sizes = torch.empty(self.count, device=encoded.device, dtype=torch.int64)
        stream = torch.cuda.current_stream().cuda_stream
        first, last = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        first.record()
        check(
            self.library.nvdx_compress(
                chunk,
                block,
                int(half),
                encoded.data_ptr(),
                temporary.data_ptr(),
                stride,
                self.sizes.data_ptr(),
                scratch.data_ptr(),
                self.count,
                stream,
            )
        )
        last.record()
        last.synchronize()
        compression_ms = first.elapsed_time(last)
        if not bool(((self.sizes > 0) & (self.sizes <= stride)).all()):
            raise RuntimeError("Invalid compressed chunk size")
        aligned = ((self.sizes + align - 1) // align) * align
        self.offsets = (torch.cumsum(aligned, 0) - aligned).contiguous()
        compressed_bytes = int(self.sizes.sum())
        padded_bytes = int(aligned.sum())
        self.compressed = torch.zeros(
            padded_bytes + 16, device=encoded.device, dtype=torch.uint8
        )
        check(
            self.library.nvdx_compact(
                temporary.data_ptr(),
                stride,
                self.offsets.data_ptr(),
                self.sizes.data_ptr(),
                self.compressed.data_ptr(),
                self.count,
                stream,
            )
        )
        torch.cuda.synchronize()
        self.setup = {
            "wall_ms": (time.perf_counter() - start_wall) * 1000,
            "compression_device_ms": compression_ms,
            "compression_temporary_output_bytes": temporary.numel(),
            "compression_global_scratch_bytes": scratch_bytes,
            "compressed_bytes": compressed_bytes,
            "aligned_stream_bytes": padded_bytes,
            "allocated_stream_bytes": self.compressed.numel(),
            "metadata_bytes": self.sizes.nbytes + self.offsets.nbytes,
            "ratio_original_to_compressed": encoded.numel() / compressed_bytes,
            "chunk_sizes_min": int(self.sizes.min()),
            "chunk_sizes_max": int(self.sizes.max()),
            "per_matrix_compressed_bytes": self.sizes.view(
                -1, (5248 * 1024 * 2) // chunk
            )
            .sum(1)
            .tolist(),
        }
        self.checksums = torch.empty(
            self.count, device=encoded.device, dtype=torch.int64
        )
        self.decoded_sizes = torch.empty_like(self.checksums)
        self.restored = torch.empty_like(encoded)

    def run(self, mode=1):
        check(
            self.library.nvdx_checksum(
                self.chunk,
                self.block,
                self.half,
                (self.raw if mode == 0 else self.compressed).data_ptr(),
                self.offsets.data_ptr(),
                self.sizes.data_ptr(),
                self.restored.data_ptr(),
                self.checksums.data_ptr(),
                self.decoded_sizes.data_ptr(),
                self.count,
                mode,
                torch.cuda.current_stream().cuda_stream,
            )
        )
        return self.checksums

    def validate(self, canonical, transform_mode):
        self.run(2)
        if not bool((self.decoded_sizes == self.chunk).all()):
            raise RuntimeError("Unexpected decoded byte count")
        if not torch.equal(self.restored, self.raw):
            raise RuntimeError("ANS changed transformed bank bytes")
        restored = transform(self.restored, transform_mode, reverse=True)
        if not torch.equal(restored, canonical):
            raise RuntimeError("ANS reconstruction changed original checkpoint bytes")
        decomp_checksum = self.checksums.clone()
        self.run(0)
        if not torch.equal(decomp_checksum, self.checksums):
            raise RuntimeError(
                "Decompression checksum differs from matched raw control"
            )
        # Timing uses the no-global-output specialization, checked independently.
        self.run(1)
        if not torch.equal(decomp_checksum, self.checksums):
            raise RuntimeError("Shared-only decode checksum differs")
        return {
            "entire_bank_transformed_bytes_exact": True,
            "entire_bank_original_bf16_bytes_exact": True,
            "all_decoded_sizes_exact": True,
            "all_chunk_checksums_exact": True,
        }
