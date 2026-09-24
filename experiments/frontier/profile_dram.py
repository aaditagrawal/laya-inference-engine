"""Run one warmed graph inside an Nsight Compute profiling range."""

import torch

from experiments.native import common

from .engine import FrontierEngine


def main():
    torch.set_num_threads(4)
    with FrontierEngine(policy="bf16-exact-compiled", max_graphs=1) as engine:
        request = common.workload(1, "short")
        for _ in range(20):
            engine.predict(**request)
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        engine.predict(**request)
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()


if __name__ == "__main__":
    main()
