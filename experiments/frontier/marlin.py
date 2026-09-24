"""Weight-only INT8 adapter for the pinned, separately built Marlin kernel.

Marlin is Apache-2.0 software from Elias Frantar and vLLM contributors. See
the pinned checkout in .research/frontier-vllm for its source and notices.
"""

import importlib.util
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
_loaded = False
_type_id = None


def load():
    global _loaded, _type_id
    if not _loaded:
        torch.ops.load_library(
            str(ROOT / ".research/frontier-marlin/laya_frontier_marlin.so")
        )
        spec = importlib.util.spec_from_file_location(
            "frontier_scalar_type", ROOT / ".research/frontier-vllm/vllm/scalar_type.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        _type_id = module.scalar_types.uint8b128.id
        _loaded = True
    return torch.ops.laya_frontier_marlin


class PackedWeight:
    @torch.inference_mode()
    def __init__(self, weight, group=32):
        ops = load()
        n, k = weight.shape
        if n % 64 or k % 64 or k % group:
            raise ValueError("This experiment requires 64-aligned weight dimensions")
        self.n, self.k, self.group = n, k, group
        blocks = weight.float().view(n, k // group, group)
        scales = (blocks.abs().amax(-1) / 127).clamp_min(1e-20).bfloat16()
        signed = (blocks / scales.float()[..., None]).round().clamp(-128, 127)
        reference = (signed * scales.float()[..., None]).bfloat16().view(n, k)
        codes = signed.view(n, k).t().contiguous().to(torch.int64) + 128
        packed = torch.zeros((k // 4, n), device=weight.device, dtype=torch.int64)
        for i in range(4):
            packed |= codes[i::4] << (8 * i)
        self.weight = ops.gptq_marlin_repack(packed.to(torch.int32), k, n, 8, False)
        # Marlin's group-scale tile is an 8x8 transpose over consecutive scales.
        scales = scales.t().contiguous()
        self.scales = (
            scales.reshape(-1, 8, 8).transpose(1, 2).reshape(k // group, n).contiguous()
        )
        self.workspace = torch.zeros(
            torch.cuda.get_device_properties(weight.device).multi_processor_count * 4,
            dtype=torch.int32,
            device=weight.device,
        )
        self.reference = reference

    def __call__(self, x):
        shape = x.shape[:-1]
        m = x.numel() // self.k
        result = load().marlin_gemm(
            x.reshape(m, self.k),
            None,
            self.weight,
            None,
            self.scales,
            None,
            None,
            None,
            self.workspace,
            _type_id,
            m,
            self.n,
            self.k,
            False,
            True,
            False,
        )
        return result.view(*shape, self.n)
