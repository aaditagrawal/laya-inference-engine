"""Optional mapped-output capture adapter; retained sources remain unchanged.

Adapted from experiments/latency/serving/graph_adapter.py. Its packing and model
calls retain the repository's Apache-2.0-derived protocol/model notices.
"""

from time import perf_counter
from types import MethodType

import torch

from experiments.latency.serving.graph_adapter import CaptureStream, OwnedSlot
from experiments.native.host.adapter import laya_native_host, packed_allocate

from .mapped_io import MappedPinned, collect


@torch.inference_mode()
def mapped_slot(self, prepared):
    key = self.base._graph_key(prepared)
    if key in self.graphs:
        self.graphs.move_to_end(key)
        return self.graphs[key], False
    if len(self.graphs) >= self.max_graphs:
        self.graphs.popitem(last=False)[1].close()
    start = perf_counter()
    replay_stream = torch.cuda.current_stream(self.device)
    owner = CaptureStream(self.device, replay_stream.priority)
    slot = OwnedSlot(capture_owner=owner)
    try:
        shape = key[:3]
        b, s, k = shape
        host, inputs_host = packed_allocate(shape, self.agent.tok.pad_token_id, "cpu")
        device, inputs = packed_allocate(
            shape, self.agent.tok.pad_token_id, self.device
        )
        inputs["global_attention_unmasked"] = key[-1]
        slot.host, slot.host_storage, slot.device_storage = inputs_host, host, device
        slot.inputs = inputs
        slot.arrays = {name: value.numpy() for name, value in inputs_host.items()}
        self._fill_numpy(slot, prepared)
        device.copy_(host, non_blocking=True)
        stream = owner.stream
        stream.wait_stream(replay_stream)
        with torch.cuda.stream(stream):
            for _ in range(3):
                outputs = self.base._forward(inputs)
        replay_stream.wait_stream(stream)
        torch.cuda.synchronize(self.device)
        assert outputs[0].is_contiguous() and outputs[1].is_contiguous()
        assert outputs[0].dtype == outputs[1].dtype == torch.float32
        action_width = outputs[1].shape[1]
        host_out = torch.empty(
            b * (k + action_width), dtype=torch.float32, pin_memory=True
        )
        host_logits = host_out[: b * k].view(b, k)
        host_actions = host_out[b * k :].view(b, action_width)
        slot.mapped_output = mapped = MappedPinned(host_out, self.device)
        # Compile and warm the staging kernel outside stream capture.
        with torch.cuda.stream(stream):
            collect(outputs[0], outputs[1], mapped)
        stream.synchronize()
        slot.graph = graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            device.copy_(host, non_blocking=True)
            outputs = self.base._forward(inputs)
            collect(outputs[0], outputs[1], mapped)
        slot.native = laya_native_host.Session(
            host.data_ptr(),
            device.data_ptr(),
            host.numel(),
            b,
            s,
            k,
            self.agent.tok.pad_token_id,
            graph.raw_cuda_graph_exec(),
            replay_stream.cuda_stream,
            outputs[0].data_ptr(),
            outputs[1].data_ptr(),
            host_out.data_ptr(),
            action_width,
            True,
        )
        slot.outputs, slot.stream = outputs, replay_stream
        slot.host_out, slot.host_logits, slot.host_actions = (
            host_out,
            host_logits,
            host_actions,
        )
        slot.build_ms = (perf_counter() - start) * 1000
        self.graphs[key] = slot
        return slot, True
    except BaseException as error:
        # Local aliases must be released before OwnedSlot destroys its stream.
        # Keep the original error, including capture failures, for the caller.
        host = inputs_host = device = inputs = outputs = mapped = None
        host_out = host_logits = host_actions = graph = None
        try:
            slot.close()
        except BaseException as cleanup_error:  # noqa: BLE001 - preserve capture failure.
            error.add_note(f"Mapped capture cleanup also failed: {cleanup_error!r}")
        raise


def install(engine):
    adapter = engine.adapter
    if adapter.mode != "cpp-graph-io" or engine.base.graphs or adapter.graphs:
        raise RuntimeError(
            "Install mapped output before graph capture in cpp-graph-io mode"
        )
    if not engine.selection.get("host_runtime"):
        raise RuntimeError(
            "This adapter expects the retained host-runtime implementation"
        )
    previous = adapter._slot
    adapter._slot = MethodType(mapped_slot, adapter)
    # host_runtime closes over the capture function. Refresh that closure after
    # changing _slot; its packing, locking and owned output copies stay identical.
    from .host_runtime import install as install_host_runtime

    install_host_runtime(engine)
    return previous
