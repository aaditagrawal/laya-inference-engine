"""Host execution experiments. Tokenization, model math, and formatting stay identical.

Each adapter owns its CUDA graphs, pinned/device storage, stream references, and
native pointer wrappers for the full replay lifetime. Public predict/run_prepared
serialize access and return copied arrays. The native replay releases the GIL
only after Python objects have been read and synchronizes before Python resumes.
"""

import threading
from collections import OrderedDict
from time import perf_counter
from types import SimpleNamespace

import numpy as np
import torch

from laya_blackwell.engine import KEYS
from laya_blackwell.protocol import format_response

from ._native import load_native

laya_native_host = load_native()

MODES = ("numpy", "cpp", "numpy-graph-io", "cpp-graph-io")


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


class HostAdapter:
    def __init__(self, base, mode="cpp-graph-io", max_graphs=4):
        if mode not in MODES:
            raise ValueError(mode)
        self.base, self.mode, self.max_graphs = base, mode, max_graphs
        self.agent, self.device, self.backend = base.agent, base.device, base.backend
        self.lock = threading.RLock()
        self.graphs = OrderedDict()
        self.closed = False

    def prepare(self, state, questions):
        return self.base.prepare(state, questions)

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
            self.graphs.popitem(last=False)
        start = perf_counter()
        shape = key[:3]
        b, s, k = shape
        host, inputs_host = packed_allocate(shape, self.agent.tok.pad_token_id, "cpu")
        device, inputs = packed_allocate(
            shape, self.agent.tok.pad_token_id, self.device
        )
        inputs["global_attention_unmasked"] = key[-1]
        slot = SimpleNamespace(
            host=inputs_host,
            host_storage=host,
            device_storage=device,
            inputs=inputs,
            arrays={key: v.numpy() for key, v in inputs_host.items()},
        )
        self._fill_numpy(slot, prepared)
        device.copy_(host, non_blocking=True)
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            for _ in range(3):
                outputs = self.base._forward(inputs)
        torch.cuda.current_stream(self.device).wait_stream(stream)
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
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            if captured_io:
                device.copy_(host, non_blocking=True)
            outputs = self.base._forward(inputs)
            if captured_io:
                host_logits.copy_(outputs[0], non_blocking=True)
                host_actions.copy_(outputs[1], non_blocking=True)
        replay_stream = torch.cuda.current_stream(self.device)
        native = laya_native_host.Session(
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
        slot.graph, slot.outputs, slot.stream = graph, outputs, replay_stream
        slot.host_out, slot.host_logits, slot.host_actions = (
            host_out,
            host_logits,
            host_actions,
        )
        slot.native, slot.build_ms = native, (perf_counter() - start) * 1000
        self.graphs[key] = slot
        return slot, True

    @torch.inference_mode()
    def run_prepared(self, prepared):
        with self.lock, torch.cuda.device(self.device):
            if self.closed:
                raise RuntimeError("Engine is closed")
            if not prepared.items:
                return np.empty((0, 0)), np.empty((0, 2)), {"graph_miss": False}
            slot, miss = self._slot(prepared)
            if self.mode.startswith("cpp"):
                slot.native.run(prepared.items)
            else:
                self._fill_numpy(slot, prepared)
                if not self.mode.endswith("graph-io"):
                    slot.device_storage.copy_(slot.host_storage, non_blocking=True)
                slot.graph.replay()
                if not self.mode.endswith("graph-io"):
                    slot.host_logits.copy_(slot.outputs[0], non_blocking=True)
                    slot.host_actions.copy_(slot.outputs[1], non_blocking=True)
                slot.stream.synchronize()
            count = len(prepared.items)
            # A result must remain owned after the next request overwrites staging.
            logits = slot.host_logits[:count].numpy().copy()
            actions = slot.host_actions[:count].numpy().copy()
            return (
                logits,
                actions,
                {
                    "graph_miss": miss,
                    "graph_build_ms": slot.build_ms if miss else 0.0,
                    "shape": list(self.base._shape(prepared)),
                    "backend": "host-" + self.mode,
                },
            )

    def predict(self, state, questions):
        start = perf_counter()
        prepared = self.prepare(state, questions)
        logits, actions, metrics = self.run_prepared(prepared)
        result = format_response(
            prepared,
            logits,
            actions,
            self.agent.temperature,
            self.agent.temperature_by_options,
        )
        metrics["total_ms"] = (perf_counter() - start) * 1000
        result["engine"] = metrics
        return result

    def close(self):
        with self.lock:
            torch.cuda.synchronize(self.device)
            self.graphs.clear()
            self.closed = True


class RustRequestTokenizer:
    """Call the tokenizer's existing Rust engine without the Transformers wrapper.

    The source tokenizer is cloned once, so disabling batch padding/truncation
    cannot mutate another caller. The SDK's sequence builder still controls its
    own truncation and marker sanitization. Memoization lasts one request.
    """

    def __init__(self, tokenizer, backend):
        self.tokenizer, self.backend = tokenizer, backend
        self.cache = {}
        # Avoid repeated Transformers special-token property lookups per option.
        for name in ("mask_token", "mask_token_id", "cls_token_id", "sep_token_id"):
            setattr(self, name, getattr(tokenizer, name))

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

    def __call__(self, text, **kwargs):
        if kwargs != {"add_special_tokens": False}:
            return self.tokenizer(text, **kwargs)
        if text not in self.cache:
            self.cache[text] = {
                "input_ids": self.backend.encode(text, add_special_tokens=False).ids
            }
        return self.cache[text]


class RustTokenizerHostAdapter(HostAdapter):
    def __init__(self, base, mode="cpp-graph-io", max_graphs=4):
        super().__init__(base, mode, max_graphs)
        from tokenizers import Tokenizer

        self.rust_tokenizer = Tokenizer.from_str(
            base.agent.tok.backend_tokenizer.to_str()
        )
        self.rust_tokenizer.no_padding()
        self.rust_tokenizer.no_truncation()

    def prepare(self, state, questions):
        from laya.common import serialize_state

        from laya_blackwell.protocol import prepare_request

        return prepare_request(
            RustRequestTokenizer(self.agent.tok, self.rust_tokenizer),
            self.agent.cfg,
            serialize_state(state),
            questions,
            max_questions=self.base.max_questions,
            truncate_left=isinstance(state, list),
        )
