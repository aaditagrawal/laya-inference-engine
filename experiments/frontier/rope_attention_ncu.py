"""One real local-attention launch for hardware counter diagnostics."""

import torch

from experiments.native import common

from .engine import FrontierEngine
from .rope_attention_adapter import attention, load
from .rope_attention_probe import capture


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    load()
    with FrontierEngine(policy="bf16-exact") as engine:
        records = capture(engine, common.workload(1, "short"))
        raw, cos, sin, mask, _ = records[1]
        output = attention(raw, cos, sin, 1, mask)
        torch.cuda.synchronize()
        assert output.shape == (1, 16, 64, 64)


if __name__ == "__main__":
    main()
