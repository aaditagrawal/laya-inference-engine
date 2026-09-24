"""Native host adapter with a uniquely owned CUDA capture stream per graph.

PyTorch 2.14 clears cuBLAS workspace entries for a graph's capture stream when
destroying that graph. Its ordinary Stream constructor round-robins over a pool:
keeping the Python Stream object alive does not prevent reuse. Distinct live
graphs must therefore not share that pooled capture-stream/workspace identity.
Each slot here creates a raw CUDA stream and destroys it after its graph and
buffers. Replay still uses the caller's chosen stream and the same native code.

The packing/capture implementation is adapted from the repository's host
experiment, which preserves its Apache-2.0-derived model and protocol notices.
"""

from time import perf_counter
from types import SimpleNamespace

import torch
from cuda.bindings import runtime

from experiments.native.host.adapter import (
    RustTokenizerHostAdapter,
    laya_native_host,
    packed_allocate,
)


def _checked(result):
    error, *values = result
    if int(error):
        raise RuntimeError(f"CUDA runtime call failed: {error}")
    return values[0] if len(values) == 1 else values


class CaptureStream:
    def __init__(self, device, priority):
        self.raw = None
        with torch.cuda.device(device):
            self.raw = _checked(runtime.cudaStreamCreateWithPriority(1, priority))
            try:
                self.stream = torch.cuda.ExternalStream(int(self.raw), device=device)
            except BaseException:
                _checked(runtime.cudaStreamDestroy(self.raw))
                self.raw = None
                raise

    def close(self):
        if self.raw is not None:
            _checked(runtime.cudaStreamDestroy(self.raw))
            self.raw = None

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001, S110 - destructors cannot raise during CUDA teardown.
            pass


class OwnedSlot(SimpleNamespace):
    def close(self):
        # Native Session only borrows pointers and has no CUDA destructor work.
        self.native = None
        graph = getattr(self, "graph", None)
        if graph is not None:
            graph.reset()
            self.graph = None
            del graph
        owner = getattr(self, "capture_owner", None)
        for name in list(vars(self)):
            if name != "capture_owner":
                delattr(self, name)
        if owner is not None:
            owner.close()
            self.capture_owner = None

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001, S110 - destructors cannot raise during CUDA teardown.
            pass


class StableHostAdapter(RustTokenizerHostAdapter):
    """Same input/output API and hot replay path; fixed capture-resource lifetime."""

    @torch.inference_mode()
    def _slot(self, prepared):
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
            host, inputs_host = packed_allocate(
                shape, self.agent.tok.pad_token_id, "cpu"
            )
            device, inputs = packed_allocate(
                shape, self.agent.tok.pad_token_id, self.device
            )
            inputs["global_attention_unmasked"] = key[-1]
            slot.host, slot.host_storage, slot.device_storage = (
                inputs_host,
                host,
                device,
            )
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
            captured_io = self.mode.endswith("graph-io")
            slot.graph = graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                if captured_io:
                    device.copy_(host, non_blocking=True)
                outputs = self.base._forward(inputs)
                if captured_io:
                    host_logits.copy_(outputs[0], non_blocking=True)
                    host_actions.copy_(outputs[1], non_blocking=True)
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
                captured_io,
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
            try:
                slot.close()
            except BaseException as cleanup_error:  # noqa: BLE001 - preserve the original capture failure.
                error.add_note(f"Capture cleanup also failed: {cleanup_error!r}")
            raise

    def close(self):
        with self.lock, torch.cuda.device(self.device):
            if self.closed:
                return
            torch.cuda.synchronize(self.device)
            for slot in self.graphs.values():
                slot.close()
            self.graphs.clear()
            self.closed = True


def replace_adapter(engine):
    """Replace before capture; the engine keeps its normal model/adapter ownership."""
    if engine.adapter.graphs:
        raise RuntimeError("Install stable adapter before graph capture")
    replacement = StableHostAdapter(
        engine.base, mode=engine.adapter.mode, max_graphs=engine.adapter.max_graphs
    )
    engine.adapter.close()
    engine.adapter = replacement
    return engine
