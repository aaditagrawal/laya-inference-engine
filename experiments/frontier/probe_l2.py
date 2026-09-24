"""Screen CUDA persisting-cache windows with full, uncached requests."""

import json
import random
import time
from pathlib import Path
from unittest.mock import patch

import torch
from cuda.bindings import runtime as cuda

from experiments.latency.serving import graph_adapter
from experiments.native import common

from .engine import FrontierEngine
from .holdout import requests

checked = graph_adapter._checked
LIMIT = cuda.cudaLimit.cudaLimitPersistingL2CacheSize
ATTRIBUTE = cuda.cudaStreamAttrID.cudaLaunchAttributeAccessPolicyWindow


def set_window(stream, pointer, size, hit_ratio):
    value = cuda.cudaStreamAttrValue()
    value.accessPolicyWindow.base_ptr = pointer
    value.accessPolicyWindow.num_bytes = size
    value.accessPolicyWindow.hitRatio = hit_ratio
    value.accessPolicyWindow.hitProp = (
        cuda.cudaAccessProperty.cudaAccessPropertyPersisting
    )
    value.accessPolicyWindow.missProp = (
        cuda.cudaAccessProperty.cudaAccessPropertyStreaming
    )
    checked(cuda.cudaStreamSetAttribute(stream, ATTRIBUTE, value))


def verify_capture_window():
    """Confirm capture actually transfers the stream policy to kernel nodes."""
    x = torch.ones(1024, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    set_window(stream.cuda_stream, x.data_ptr(), x.nbytes, 1.0)
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph, stream=stream):
        y = x + 1
    handle = graph.raw_cuda_graph()
    _, count = checked(cuda.cudaGraphGetNodes(handle))
    nodes, _ = checked(cuda.cudaGraphGetNodes(handle, count))
    windows = []
    for node in nodes:
        if (
            checked(cuda.cudaGraphNodeGetType(node))
            != cuda.cudaGraphNodeType.cudaGraphNodeTypeKernel
        ):
            continue
        value = checked(
            cuda.cudaGraphKernelNodeGetAttribute(
                node, cuda.cudaKernelNodeAttrID.cudaLaunchAttributeAccessPolicyWindow
            )
        )
        window = value.accessPolicyWindow
        if (
            window.base_ptr != x.data_ptr()
            or window.num_bytes != x.nbytes
            or window.hitRatio != 1.0
        ):
            raise RuntimeError("Captured graph did not retain the stream access policy")
        windows.append({"bytes": window.num_bytes, "hit_ratio": window.hitRatio})
    if not windows:
        raise RuntimeError("No captured kernel nodes found")
    graph.replay()
    torch.cuda.synchronize()
    if not torch.equal(y, torch.full_like(x, 2)):
        raise RuntimeError("Control graph produced incorrect output")
    graph.reset()
    return windows


@torch.inference_mode()
def pack_weights(model, prefix):
    selected = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if name.startswith(prefix) and parameter.dtype == torch.bfloat16
    ]
    bank = torch.empty(
        sum(p.numel() for _, p in selected), device="cuda", dtype=torch.bfloat16
    )
    offset = 0
    for name, parameter in selected:
        view = bank[offset : offset + parameter.numel()].view_as(parameter)
        view.copy_(parameter)
        parent, attr = name.rsplit(".", 1)
        setattr(
            model.get_submodule(parent),
            attr,
            torch.nn.Parameter(view, requires_grad=False),
        )
        offset += parameter.numel()
    return bank, [name for name, _ in selected]


def clear_graphs(engine):
    torch.cuda.synchronize()
    for slot in engine.adapter.graphs.values():
        slot.close()
    engine.adapter.graphs.clear()


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    torch.cuda.init()
    prop = checked(cuda.cudaGetDeviceProperties(0))
    original_limit = checked(cuda.cudaDeviceGetLimit(LIMIT))
    report = {
        "metadata": common.metadata(),
        "l2_bytes": prop.l2CacheSize,
        "max_persisting_bytes": prop.persistingL2CacheMaxSize,
        "original_limit_bytes": original_limit,
        "capture_control": verify_capture_window(),
        "targets": [],
    }
    path = Path("results/frontier/l2-policy.json")
    request = common.workload(1, "short")
    fixtures = requests()
    rng = random.Random(9384)
    try:
        for target in [
            "net.head.layers.0.",
            "net.head.layers.1.",
            "net.encoder.layers.27.",
        ]:
            with FrontierEngine(
                policy="bf16-splitk-exact-short-compiled",
                attention="native",
                fuse_reduce_norm=True,
                token_tables=True,
                max_graphs=4,
            ) as engine:
                model = engine.base.model.original
                bank, names = pack_weights(model, target)
                prepared = engine.prepare(**request)
                reference = engine.run_prepared(prepared)
                # Copy outputs because each graph owns reusable host storage.
                reference = tuple(
                    x.copy() if hasattr(x, "copy") else x for x in reference
                )
                target_report = {
                    "prefix": target,
                    "packed_bytes": bank.nbytes,
                    "parameters": names,
                    "rows": [],
                }
                report["targets"].append(target_report)
                configurations = [
                    (0, 0.0),
                    (8, 8 * 2**20 / bank.nbytes),
                    (16, 16 * 2**20 / bank.nbytes),
                    (30, 1.0),
                ]
                for round_id in range(5):
                    rng.shuffle(configurations)
                    for mib, ratio in configurations:
                        clear_graphs(engine)
                        checked(cuda.cudaCtxResetPersistingL2Cache())
                        checked(
                            cuda.cudaDeviceSetLimit(
                                LIMIT, mib * 2**20 if mib else original_limit
                            )
                        )
                        base_capture = graph_adapter.CaptureStream
                        pointer, size = bank.data_ptr(), bank.nbytes
                        hit_ratio = min(1.0, ratio)

                        class CaptureWithWindow(base_capture):
                            def __init__(
                                self,
                                device,
                                priority,
                                enabled=bool(mib),
                                pointer=pointer,
                                size=size,
                                hit_ratio=hit_ratio,
                            ):
                                super().__init__(device, priority)
                                if enabled:
                                    set_window(
                                        self.raw,
                                        pointer,
                                        size,
                                        hit_ratio,
                                    )

                        with patch.object(
                            graph_adapter, "CaptureStream", CaptureWithWindow
                        ):
                            actual = engine.run_prepared(prepared)
                            parity = common.compare_outputs(
                                actual, reference, prepared, engine.agent
                            )
                            for _ in range(20):
                                engine.predict(**request)
                            fixed, changing = [], []
                            for _ in range(80):
                                start = time.perf_counter()
                                engine.predict(**request)
                                fixed.append((time.perf_counter() - start) * 1000)
                            indices = list(range(len(fixtures)))
                            rng.shuffle(indices)
                            for index in indices:
                                start = time.perf_counter()
                                engine.predict(**fixtures[index])
                                changing.append((time.perf_counter() - start) * 1000)
                        row = {
                            "round": round_id,
                            "set_aside_mib": mib,
                            "hit_ratio": min(1.0, ratio),
                            "parity": parity,
                            "fixed": common.stats(fixed),
                            "changing": common.stats(changing),
                        }
                        target_report["rows"].append(row)
                        print(
                            json.dumps(
                                {
                                    "target": target,
                                    "round": round_id,
                                    "mib": mib,
                                    "fixed_ms": row["fixed"]["p50_ms"],
                                    "changing_ms": row["changing"]["p50_ms"],
                                    "exact": parity["exact_logits_and_actions"],
                                }
                            ),
                            flush=True,
                        )
                        path.write_text(json.dumps(report, indent=2) + "\n")
                clear_graphs(engine)
            del bank
    finally:
        torch.cuda.synchronize()
        checked(cuda.cudaCtxResetPersistingL2Cache())
        checked(cuda.cudaDeviceSetLimit(LIMIT, original_limit))


if __name__ == "__main__":
    main()
