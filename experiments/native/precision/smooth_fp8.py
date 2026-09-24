"""Experimental channel balancing with fused FP32 rescaling before FP8 rounding.

S_j = activation_amax_j**alpha / weight_amax_j**(1-alpha).
The FP32 balanced weight is W*S, while the fused input quantizer handles X/S.
Only the final balanced activations/weights are rounded to E4M3FN; no extra BF16
rescaling is inserted. Formula follows the original SmoothQuant implementation:
https://github.com/mit-han-lab/smoothquant/blob/main/smoothquant/smooth.py
"""

import torch
import triton as tr
import triton.language as tl
from torch import nn

from laya_blackwell.quantization import FP8Linear


@tr.jit
def quantize_balanced_rows(
    X, Q, RowScale, Inverse, K: tl.constexpr, BLOCK: tl.constexpr
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * K + col, col < K, 0).to(tl.float32)
    inverse = tl.load(Inverse + col, col < K, 1.0)
    balanced = x * inverse
    amax = tl.max(tl.abs(balanced), 0)
    scale = tl.maximum(amax / 448.0, 1e-12)
    q = tl.minimum(tl.maximum(balanced / scale, -448.0), 448.0)
    tl.store(Q + row * K + col, q, col < K)
    tl.store(RowScale + row, scale)


class SmoothFP8Linear(FP8Linear):
    def __init__(self, weight, weight_scale, bias, inverse):
        super().__init__(weight, weight_scale, bias)
        self.register_buffer("inverse_channel_scale", inverse)

    @classmethod
    @torch.no_grad()
    def from_linear(cls, linear, activation_amax, alpha):
        if not isinstance(linear, nn.Linear) or linear.weight.dtype != torch.bfloat16:
            raise TypeError("Expected a BF16 nn.Linear source")
        if linear.weight.device.type != "cuda":
            raise ValueError("CUDA source required")
        if linear.in_features % 16 or linear.out_features % 16:
            raise ValueError("FP8 dimensions must be divisible by 16")
        weight = linear.weight.detach().float()
        activation_amax = activation_amax.to(
            device=weight.device, dtype=torch.float32
        ).clamp_min(1e-5)
        assert activation_amax.shape == (linear.in_features,)
        weight_amax = weight.abs().amax(dim=0).clamp_min(1e-5)
        channel_scale = (
            activation_amax.pow(alpha) / weight_amax.pow(1.0 - alpha)
        ).clamp_min(1e-5)
        assert torch.isfinite(channel_scale).all().item()
        balanced = weight * channel_scale[None, :]
        row_scale = (balanced.abs().amax(dim=1, keepdim=True) / 448.0).clamp_min(1e-12)
        packed = (
            (balanced / row_scale)
            .clamp(-448.0, 448.0)
            .to(torch.float8_e4m3fn)
            .contiguous()
        )
        bias = None if linear.bias is None else linear.bias.detach().bfloat16().clone()
        module = cls(
            packed,
            row_scale.T.contiguous(),
            bias,
            channel_scale.reciprocal().contiguous(),
        ).eval()
        summary = {
            "scale_min": channel_scale.min().item(),
            "scale_max": channel_scale.max().item(),
            "activation_amax_min": activation_amax.min().item(),
            "activation_amax_max": activation_amax.max().item(),
            "weight_amax_min": weight_amax.min().item(),
            "weight_amax_max": weight_amax.max().item(),
        }
        return module, summary

    def forward(self, x):
        if x.device != self.weight.device or x.dtype != torch.bfloat16:
            raise ValueError("Expected BF16 input on weight device")
        if x.ndim < 2 or x.shape[-1] != self.in_features:
            raise ValueError("Incorrect input shape")
        rows = x.numel() // self.in_features
        if rows < 64:
            raise ValueError("FP8 rowwise scaled GEMM requires at least 64 rows")
        matrix = x.reshape(rows, self.in_features).contiguous()
        quantized = torch.empty_like(matrix, dtype=torch.float8_e4m3fn)
        scale = torch.empty((rows, 1), device=x.device, dtype=torch.float32)
        block = tr.next_power_of_2(self.in_features)
        quantize_balanced_rows[(rows,)](
            matrix,
            quantized,
            scale,
            self.inverse_channel_scale,
            self.in_features,
            block,
            num_warps=4 if block <= 2048 else 8,
        )
        result = torch._scaled_mm(
            quantized,
            self.weight.T,
            scale_a=scale,
            scale_b=self.weight_scale,
            bias=self.bias,
            out_dtype=torch.bfloat16,
            use_fast_accum=False,
        )
        return result.reshape(*x.shape[:-1], self.out_features)
