"""Coalesced, once-per-word matrix-weight reads for a streaming diagnostic."""

import torch
import triton as tr
import triton.language as tl

BLOCK = 4096
COMPILED = {}


@tr.jit
def _weight_stream(X, Y, N: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(X + index, index < N, 0)
    tl.store(Y + tl.program_id(0), tl.sum(value, 0))


class Operation:
    def __init__(self, weight):
        if not weight.is_contiguous() or weight.nbytes % 4:
            raise ValueError("Expected contiguous weights with whole 32-bit words")
        self.words = weight.view(torch.uint32).flatten()
        self.output = torch.empty(
            tr.cdiv(self.words.numel(), BLOCK), device=weight.device, dtype=torch.uint32
        )

    def __call__(self):
        kernel = _weight_stream[(self.output.numel(),)](
            self.words,
            self.output,
            self.words.numel(),
            BLOCK,
            num_warps=4,
        )
        if not torch.cuda.is_current_stream_capturing():
            COMPILED[self.words.numel()] = kernel
        return self.output
