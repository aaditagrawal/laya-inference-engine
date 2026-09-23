"""Experimental FP8 encoder linears; calibration preservation is not established.

Weights use static per-output-row E4M3FN quantization. Activations use dynamic
per-input-row scales computed in a fused Triton kernel. ``torch._scaled_mm``
executes the FP8 GEMM with FP32 accumulation and BF16 output. Attention and the
decision head stay in their existing precision; the model retains FP32 residuals.
The scale API is verified with PyTorch 2.14.0+cu130 on SM120.
"""

import torch
import triton as tr
import triton.language as tl
from torch import nn


_FP8_MAX = 448.0
_MIN_SCALE = 1e-12


@tr.jit
def _quantize_rows(X, Q, S, K: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * K + col, col < K, 0).to(tl.float32)
    amax = tl.max(tl.abs(x), 0)
    scale = tl.maximum(amax / 448.0, 1e-12)
    quantized = tl.minimum(tl.maximum(x / scale, -448.0), 448.0)
    tl.store(Q + row * K + col, quantized, col < K)
    tl.store(S + row, scale)


class FP8Linear(nn.Module):
    """Inference-only FP8 GEMM with dynamic row scales and BF16 outputs.

    This path changes model numerics. It is opt-in and must be evaluated on
    representative decision/calibration data before deployment. Input row count
    must be at least 64; input/output feature counts must be multiples of 16.
    """

    def __init__(self, weight, weight_scale, bias):
        super().__init__()
        self.out_features, self.in_features = weight.shape
        self.register_buffer("weight", weight)
        # _scaled_mm rowwise scale shapes are [M, 1] for A and [1, N] for B.
        self.register_buffer("weight_scale", weight_scale)
        self.register_buffer("bias", bias)

    @classmethod
    @torch.no_grad()
    def from_linear(cls, linear: nn.Linear):
        """Quantize an existing CUDA BF16 linear without modifying its weights."""
        if not isinstance(linear, nn.Linear):
            raise TypeError("FP8Linear.from_linear requires torch.nn.Linear")
        if linear.weight.device.type != "cuda":
            raise ValueError("FP8 encoder linears require CUDA-resident weights")
        capability = torch.cuda.get_device_capability(linear.weight.device)
        if capability not in {(10, 0), (10, 3), (11, 0), (12, 0), (12, 1)}:
            raise ValueError(f"FP8 encoder path requires Blackwell; found SM{capability[0]}{capability[1]}")
        if linear.weight.dtype != torch.bfloat16:
            raise ValueError("FP8Linear.from_linear requires BF16 source weights")
        if linear.in_features % 16 or linear.out_features % 16:
            raise ValueError(
                "FP8 GEMM requires input/output feature counts divisible by 16; "
                f"got K={linear.in_features}, N={linear.out_features}"
            )
        weight = linear.weight.detach().float()
        if not torch.isfinite(weight).all().item():
            raise ValueError("FP8 source weights contain NaN or infinity")
        scale = (weight.abs().amax(dim=1, keepdim=True) / _FP8_MAX).clamp_min(_MIN_SCALE)
        packed = (weight / scale).clamp(-_FP8_MAX, _FP8_MAX).to(torch.float8_e4m3fn).contiguous()
        bias = None if linear.bias is None else linear.bias.detach().to(torch.bfloat16).clone()
        return cls(packed, scale.T.contiguous(), bias).eval()

    def forward(self, x):
        if x.device != self.weight.device or x.dtype != torch.bfloat16:
            raise ValueError(
                f"FP8Linear requires BF16 input on {self.weight.device}; "
                f"got {x.dtype} on {x.device}"
            )
        if x.ndim < 2 or x.shape[-1] != self.in_features:
            raise ValueError(
                f"FP8Linear requires input (..., {self.in_features}) with at least two dimensions; "
                f"got {tuple(x.shape)}"
            )
        rows = x.numel() // self.in_features
        if rows < 64:
            raise ValueError(f"FP8Linear requires at least 64 input rows; got {rows}. Use a larger bucket or BF16.")
        matrix = x.reshape(rows, self.in_features).contiguous()
        quantized = torch.empty_like(matrix, dtype=torch.float8_e4m3fn)
        scale = torch.empty((rows, 1), device=x.device, dtype=torch.float32)
        block = tr.next_power_of_2(self.in_features)
        _quantize_rows[(rows,)](
            matrix, quantized, scale, self.in_features, block,
            num_warps=4 if block <= 2048 else 8,
        )
        result = torch._scaled_mm(
            quantized, self.weight.T,
            scale_a=scale, scale_b=self.weight_scale,
            bias=self.bias, out_dtype=torch.bfloat16, use_fast_accum=False,
        )
        return result.reshape(*x.shape[:-1], self.out_features)

    def extra_repr(self):
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}, precision=experimental_e4m3fn_rowwise"
        )


def replace_encoder_linears(model) -> int:
    """Replace only ``FastDecisionModel.net.encoder`` linears; return the count.

    Changes no head, attention softmax, or residual-addition implementation.
    Calling this again is harmless and returns zero after all replacements.
    """
    try:
        encoder = model.net.encoder
    except AttributeError as exc:
        raise TypeError("replace_encoder_linears expects FastDecisionModel with net.encoder") from exc
    replacements = []
    for name, module in list(encoder.named_modules()):
        if isinstance(module, nn.Linear):
            parent_name, _, child_name = name.rpartition(".")
            parent = encoder.get_submodule(parent_name) if parent_name else encoder
            replacements.append((parent, child_name, FP8Linear.from_linear(module)))
    # Finish validation/quantization before replacing any module.
    for parent, name, quantized in replacements:
        setattr(parent, name, quantized)
    return len(replacements)
