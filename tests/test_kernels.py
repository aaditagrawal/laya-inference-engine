"""GPU tests compare fused math with the operations it replaces."""
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from laya_blackwell.kernels import add_norm, geglu, rope_qkv

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


@torch.inference_mode()
@pytest.mark.parametrize("bias", [False, True])
def test_residual_layernorm(bias):
    torch.manual_seed(4)
    x = torch.randn(2, 65, 1024, device="cuda")
    r = torch.randn_like(x, dtype=torch.bfloat16)
    norm = nn.LayerNorm(1024, bias=bias, device="cuda").eval()
    y, n = add_norm(x, r, norm)
    torch.testing.assert_close(y, x+r.float(), rtol=0, atol=0)
    torch.testing.assert_close(n.float(), norm(y).bfloat16().float(), rtol=0.008, atol=0.008)


@torch.inference_mode()
def test_geglu_non_power_of_two():
    torch.manual_seed(5)
    x = torch.randn(3, 65, 5248, device="cuda", dtype=torch.bfloat16)
    a, b = x.chunk(2, -1)
    torch.testing.assert_close(geglu(x), F.gelu(a)*b, rtol=0.008, atol=0.001)


@torch.inference_mode()
@pytest.mark.parametrize("fp32", [False, True])
def test_rope_preserves_value_and_rounding(fp32):
    torch.manual_seed(6)
    qkv = torch.randn(2, 65, 3, 16, 64, device="cuda", dtype=torch.bfloat16)
    angles = torch.randn(65, 32, device="cuda")
    angles = torch.cat((angles, angles), -1)
    cos, sin = angles.cos(), angles.sin()
    if not fp32:
        cos, sin = cos.bfloat16(), sin.bfloat16()
    got = rope_qkv(qkv, cos, sin, fp32=fp32)
    qk = qkv[:, :, :2]
    if fp32:
        qk = qk.float()
    rotated = torch.cat((-qk[..., 32:], qk[..., :32]), -1)
    expected = qk*cos[None, :, None, None]+rotated*sin[None, :, None, None]
    torch.testing.assert_close(got[:, :, :2], expected.bfloat16(), rtol=0, atol=0)
    torch.testing.assert_close(got[:, :, 2], qkv[:, :, 2], rtol=0, atol=0)
