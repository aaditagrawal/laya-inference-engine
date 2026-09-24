"""Close/recreate and pinned-owner release checks without loading a model.

Run under the exclusive experiment lock. No timing is performed.
"""

import gc
import hashlib
import json
import weakref
from pathlib import Path

import numpy as np
import torch

from experiments.latency.serving.graph_adapter import CaptureStream, OwnedSlot
from experiments.native.host.adapter import laya_native_host, packed_allocate

from .mapped_io import MappedPinned, collect
from .mapped_io_probe import control


@torch.inference_mode()
def make_slot(shape):
    b, s, k = shape
    owner = CaptureStream(torch.device("cuda:0"), 0)
    slot = OwnedSlot(capture_owner=owner)
    host, host_views = packed_allocate(shape, 50283, "cpu")
    device, device_views = packed_allocate(shape, 50283, "cuda:0")
    output = torch.empty(b * (k + 2), dtype=torch.float32, pin_memory=True)
    mapped = MappedPinned(output)
    logits = torch.empty((b, k), device="cuda")
    actions = torch.empty((b, 2), device="cuda")
    stream = owner.stream
    stream.wait_stream(torch.cuda.current_stream())
    host.copy_(torch.arange(host.numel()).remainder(251).to(torch.uint8))
    with torch.cuda.stream(stream):
        device.copy_(host, non_blocking=True)
        control(device, logits, actions)
        collect(logits, actions, mapped)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        device.copy_(host, non_blocking=True)
        control(device, logits, actions)
        collect(logits, actions, mapped)
    slot.native = laya_native_host.Session(
        host.data_ptr(),
        device.data_ptr(),
        host.numel(),
        b,
        s,
        k,
        50283,
        graph.raw_cuda_graph_exec(),
        stream.cuda_stream,
        logits.data_ptr(),
        actions.data_ptr(),
        output.data_ptr(),
        2,
        True,
    )
    slot.graph = graph
    slot.host, slot.host_views = host, host_views
    slot.device, slot.device_views = device, device_views
    slot.output, slot.outputs, slot.mapped = output, (logits, actions), mapped
    slot.numpy_output = output.numpy()
    return slot, {
        name: weakref.ref(value)
        for name, value in {
            "host": host,
            "output": output,
            "mapped": mapped,
            "interface_owner": mapped.owner,
            "alias": mapped.alias,
        }.items()
    }


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    before = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in Path("experiments/frontier").glob("mapped_io*.py")
    }
    rows = []
    for cycle in range(5):
        for shape in ((1, 64, 4), (1, 512, 4), (16, 64, 4)):
            slot, refs = make_slot(shape)
            snapshots = []
            for offset in (0, 31, 77):
                slot.host.copy_(
                    (torch.arange(slot.host.numel()) + offset)
                    .remainder(251)
                    .to(torch.uint8)
                )
                b, _, k = shape
                expected = torch.cat(
                    [slot.host[: b * k].float(), slot.host[-b * 2 :].flip(0).float()]
                ).numpy()
                slot.output.view(torch.int32).fill_(0x7FC01234)
                with torch.cuda.stream(slot.capture_owner.stream):
                    slot.device.fill_(253)
                    for item in slot.outputs:
                        item.fill_(float("nan"))
                del item
                slot.native.replay()
                actual = slot.numpy_output.copy()
                assert np.array_equal(actual.view(np.uint32), expected.view(np.uint32))
                snapshots.append((actual, actual.copy()))
            slot.close()
            slot.close()  # Idempotent close must not reuse destroyed streams.
            gc.collect()
            assert all(ref() is None for ref in refs.values())
            assert all(np.array_equal(a, b) for a, b in snapshots)
            rows.append(
                {
                    "cycle": cycle,
                    "shape": list(shape),
                    "owner_weakrefs_expired": True,
                    "owned_copies_unchanged": True,
                    "idempotent_close": True,
                }
            )
    after = {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in before}
    report = {
        "all_exact": True,
        "cycles": 5,
        "slots_closed_and_recreated": len(rows),
        "rows": rows,
        "source_before": before,
        "source_after": after,
        "sources_unchanged": before == after,
    }
    report["independent_cpu_expected_and_poison"] = True
    Path("results/frontier/mapped-io-lifetime-poison.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                k: v
                for k, v in report.items()
                if k not in {"rows", "source_before", "source_after"}
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
