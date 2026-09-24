"""Owned CUDA graphs, packed transfers, and request-local Rust tokenization."""

import threading
from collections import OrderedDict
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import torch
from cuda.bindings import runtime

from laya_blackwell.engine import KEYS

from .host_prepare import prepare
from .native import load_python_extension
from .native_format import NativeFormatter


def packed_allocate(shape, pad, device):
    b, s, k = shape
    layout = [
        ("input_ids", (b, s), torch.int64),
        ("attention_mask", (b, s), torch.int64),
        ("marker_pos", (b, k), torch.int64),
        ("qtype", (b,), torch.int64),
        ("marker_mask", (b, k), torch.bool),
    ]
    size = sum(
        np.prod(dims).item() * torch.tensor([], dtype=dtype).element_size()
        for _, dims, dtype in layout
    )
    size = int((size + 7) // 8 * 8)
    kw = {"device": device}
    if str(device) == "cpu":
        kw["pin_memory"] = True
    storage = torch.empty(size, dtype=torch.uint8, **kw)
    offset = 0
    views = {}
    for key, dims, dtype in layout:
        nbytes = int(np.prod(dims)) * torch.tensor([], dtype=dtype).element_size()
        views[key] = storage[offset : offset + nbytes].view(dtype).view(dims)
        offset += nbytes
    return storage, views


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


class HostAdapter:
    """Serialize replay and return owned arrays from a bounded graph cache."""

    def __init__(self, base, max_graphs=8):
        from tokenizers import Tokenizer

        self.base, self.mode, self.max_graphs = base, "cpp-graph-io", max_graphs
        self.agent, self.device, self.backend = base.agent, base.device, "fast"
        self.lock = threading.RLock()
        self.graphs = OrderedDict()
        self.closed = False
        self.native_host = load_python_extension("host")
        self.formatter = NativeFormatter()
        self.rust_tokenizer = Tokenizer.from_str(
            base.agent.tok.backend_tokenizer.to_str()
        )
        self.rust_tokenizer.no_padding()
        self.rust_tokenizer.no_truncation()

    def prepare(self, state, questions):
        return prepare(
            self.agent.tok,
            self.rust_tokenizer,
            self.agent.cfg,
            state,
            questions,
            max_questions=self.base.max_questions,
        )

    def _fill_numpy(self, slot, prepared):
        a = slot.arrays
        a["input_ids"].fill(self.agent.tok.pad_token_id)
        for key in KEYS[1:]:
            a[key].fill(0)
        a["attention_mask"][:, 0] = 1
        a["marker_mask"][:, 0] = True
        for i, item in enumerate(prepared.items):
            n, k = len(item["ids"]), len(item["markers"])
            a["input_ids"][i, :n] = item["ids"]
            a["attention_mask"][i, :n] = 1
            a["marker_pos"][i, :k] = item["markers"]
            a["marker_mask"][i, :k] = True
            a["qtype"][i] = item["qtype"]

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
            slot.native = self.native_host.Session(
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
            # CUDA's pinned allocator can record release events on the capture
            # stream. Drop local aliases as well as slot-owned views before
            # destroying that stream, including partially constructed slots.
            outputs = inputs = inputs_host = host = device = None
            host_out = host_logits = host_actions = graph = None
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

    def run_prepared(self, prepared):
        # The native session serializes packing, graph replay and stream sync.
        # Keep the adapter lock until fresh output copies have been made, so a
        # concurrent caller can never overwrite the arrays being copied.
        with self.lock, torch.cuda.device(self.device):
            if self.closed:
                raise RuntimeError("Engine is closed")
            count = len(prepared.items)
            if not count:
                return np.empty((0, 0)), np.empty((0, 2)), {"graph_miss": False}
            key = self.base._graph_key(prepared)
            if key in self.graphs:
                self.graphs.move_to_end(key)
                slot, miss = self.graphs[key], False
            else:
                # Capture remains under the original inference-mode and device
                # guards. Only the warmed path avoids repeated Torch dispatch.
                slot, miss = self._slot(prepared)
            if not hasattr(slot, "runtime_numpy_views"):
                slot.runtime_numpy_views = (
                    slot.host_logits.numpy(),
                    slot.host_actions.numpy(),
                )
            self.graphs[key] = slot
            slot.native.run(prepared.items)
            logits, actions = slot.runtime_numpy_views
            return (
                logits[:count].copy(),
                actions[:count].copy(),
                {
                    "graph_miss": miss,
                    "graph_build_ms": slot.build_ms if miss else 0.0,
                    "shape": list(key[:3]),
                    "backend": "fast",
                },
            )

    def predict(self, state, questions):
        start = perf_counter()
        prepared = self.prepare(state, questions)
        logits, actions, metrics = self.run_prepared(prepared)
        result = self.formatter(
            prepared,
            logits,
            actions,
            self.agent.temperature,
            self.agent.temperature_by_options,
        )
        metrics["total_ms"] = (perf_counter() - start) * 1000
        result["engine"] = metrics
        return result
