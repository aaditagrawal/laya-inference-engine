"""Two untimed native launches for external Compute Sanitizer race checking."""

import torch

from .ws_lossless import Operation, pack


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    torch.manual_seed(92026)
    code = torch.randint(0, 256, (5248, 1024), device="cuda", dtype=torch.int32)
    bits = (code & 127) | (119 << 7) | ((code >> 7) << 15)
    weight = bits.to(torch.int16).view(torch.bfloat16)
    storage = pack(weight, True)
    x = torch.randn(1, 64, 1024, device="cuda", dtype=torch.bfloat16)
    for producers, stages in [(1, 3), (2, 2)]:
        operation = Operation(x, storage, 5248, 1024, 1, producers, stages, True, True)
        operation()
        torch.cuda.synchronize()
        if not torch.isfinite(operation.output).all():
            raise RuntimeError("Nonfinite sanitizer output")
        print(f"Completed {producers} producer(s), {stages} stages", flush=True)


if __name__ == "__main__":
    main()
