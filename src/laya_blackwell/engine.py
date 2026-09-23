"""Bounded CUDA graph cache and serialized access to reusable device buffers."""
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
import threading
import time
import warnings

import numpy as np
import torch
from huggingface_hub import snapshot_download
from laya import Agent
from laya.common import serialize_state

from .model import FastDecisionModel
from .protocol import prepare_request, format_response

MODEL_ID = "convaiinnovations/laya"
REVISION = "5e7b2b1b8ca2ecdd3f2322d94069c9b6ce7e844b"
KEYS = ("input_ids", "attention_mask", "marker_pos", "marker_mask", "qtype")


class _RequestTokenizer:
    """Tokenize repeated state/option strings once per request, without an answer cache."""
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.cache = {}

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

    def __call__(self, text, **kwargs):
        if kwargs != {"add_special_tokens": False}:
            return self.tokenizer(text, **kwargs)
        if text not in self.cache:
            self.cache[text] = self.tokenizer(text, **kwargs)
        return self.cache[text]


def hardware_info(device="cuda:0"):
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA-enabled PyTorch build and NVIDIA Blackwell GPU are required")
    p = torch.cuda.get_device_properties(device)
    capability = (p.major, p.minor)
    if capability not in {(10, 0), (10, 3), (11, 0), (12, 0), (12, 1)}:
        raise RuntimeError(f"Expected Blackwell; found {p.name}, SM{p.major}{p.minor}")
    return {"name": p.name, "compute_capability": f"{p.major}.{p.minor}",
            "sm_count": p.multi_processor_count, "vram_bytes": p.total_memory,
            "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
            "cuda_architecture": f"sm_{p.major}{p.minor}"}


def model_path(model=MODEL_ID, revision=REVISION, subfolder=None):
    if Path(model).is_dir():
        root = Path(model)
    else:
        prefix = f"{subfolder}/" if subfolder else ""
        root = Path(snapshot_download(model, revision=revision, allow_patterns=[
            prefix + x for x in ("model.safetensors", "rl_agent_config.json", "encoder/*", "tokenizer/*")
        ]))
    return str(root / subfolder if subfolder else root)


@dataclass
class GraphSlot:
    host: dict
    inputs: dict
    graph: torch.cuda.CUDAGraph
    outputs: tuple
    build_ms: float


class BlackwellEngine:
    def __init__(self, model=MODEL_ID, *, revision=REVISION, subfolder=None,
                 device="cuda:0", max_graphs=8, max_questions=64,
                 sequence_buckets=(64, 128, 256, 384, 512, 768, 1024),
                 backend="fused"):
        self.device = torch.device(device)
        self.hardware = hardware_info(self.device)
        if max_graphs < 1 or max_questions < 1:
            raise ValueError("max_graphs and max_questions must be positive")
        if backend not in {"fused", "fp8", "eager"}:
            raise ValueError("backend must be fused, fp8 or eager")
        self.lock = threading.RLock()
        self.max_graphs, self.max_questions = max_graphs, max_questions
        self.sequence_buckets = tuple(sorted(set(sequence_buckets)))
        self.backend = backend
        self.graphs = OrderedDict()
        self.closed = False
        # Load on CPU first. This engine never silently migrates requests to CPU.
        self.agent = Agent(model_path(model, revision, subfolder), device="cpu")
        self.agent.dtype = torch.bfloat16
        self.agent.device = self.device
        self.agent.model.to(self.device).eval().requires_grad_(False)
        self.model = FastDecisionModel(self.agent.model) if backend != "eager" else self.agent.model
        if backend == "fp8":
            warnings.warn("FP8 is experimental: the bundled numerical check found changed decisions. "
                          "Use fused for the validated BF16 path.", RuntimeWarning, stacklevel=2)
            from .quantization import replace_encoder_linears
            replace_encoder_linears(self.model)
        if backend != "eager":
            # Release the original FP32 matrix weights after converting once.
            self.agent.model = None
        self.revision = revision

    def prepare(self, state, questions):
        return prepare_request(_RequestTokenizer(self.agent.tok), self.agent.cfg,
                               serialize_state(state), questions,
                               max_questions=self.max_questions,
                               truncate_left=isinstance(state, list))

    def _shape(self, prepared):
        count = len(prepared.items)
        length = max(len(x["ids"]) for x in prepared.items)
        options = max(2, max(len(x["markers"]) for x in prepared.items))
        max_len = self.agent.cfg.get("max_len", 512)
        length = next((x for x in self.sequence_buckets if length <= x <= max_len), max_len)
        batch = min(self.max_questions, 1 << (count - 1).bit_length())
        return batch, length, 1 << (options - 1).bit_length()

    def _allocate(self, shape, *, host):
        b, s, k = shape
        kw = {"device": "cpu", "pin_memory": True} if host else {"device": self.device}
        return {
            "input_ids": torch.full((b, s), self.agent.tok.pad_token_id, dtype=torch.long, **kw),
            "attention_mask": torch.zeros((b, s), dtype=torch.long, **kw),
            "marker_pos": torch.zeros((b, k), dtype=torch.long, **kw),
            "marker_mask": torch.zeros((b, k), dtype=torch.bool, **kw),
            "qtype": torch.zeros(b, dtype=torch.long, **kw),
        }

    def _graph_key(self, prepared):
        shape = self._shape(prepared)
        batch, length, _ = shape
        unmasked = len(prepared.items) == batch and all(
            len(item["ids"]) == length for item in prepared.items
        )
        return (*shape, unmasked)

    def _fill(self, host, prepared):
        host["input_ids"].fill_(self.agent.tok.pad_token_id)
        for key in KEYS[1:]:
            host[key].zero_()
        # Dummy batch rows have one valid token and option, avoiding all-masked attention.
        host["attention_mask"][:, 0] = 1
        host["marker_mask"][:, 0] = True
        for i, item in enumerate(prepared.items):
            n, k = len(item["ids"]), len(item["markers"])
            host["input_ids"][i, :n] = torch.as_tensor(item["ids"])
            host["attention_mask"][i, :n] = 1
            host["marker_pos"][i, :k] = torch.as_tensor(item["markers"])
            host["marker_mask"][i, :k] = True
            host["qtype"][i] = item["qtype"]

    def _forward(self, inputs):
        if self.backend == "eager":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                return self.model(**inputs)
        return self.model(**inputs)

    @torch.inference_mode()
    def _slot(self, shape, prepared):
        key = self._graph_key(prepared)
        if key in self.graphs:
            self.graphs.move_to_end(key)
            return self.graphs[key], False
        if len(self.graphs) >= self.max_graphs:
            # Callers hold the lock and the previous result copy has completed.
            self.graphs.popitem(last=False)
        start = time.perf_counter()
        host = self._allocate(shape, host=True)
        self._fill(host, prepared)
        inputs = {k: v.to(self.device) for k, v in host.items()}
        inputs["global_attention_unmasked"] = key[-1]
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(stream):
            for _ in range(3):
                outputs = self._forward(inputs)
        torch.cuda.current_stream(self.device).wait_stream(stream)
        torch.cuda.synchronize(self.device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            outputs = self._forward(inputs)
        slot = GraphSlot(host, inputs, graph, outputs, (time.perf_counter()-start)*1000)
        self.graphs[key] = slot
        return slot, True

    @torch.inference_mode()
    def run_prepared(self, prepared):
        """Returns owned CPU arrays; graph output buffers never escape the lock."""
        with self.lock, torch.cuda.device(self.device):
            if self.closed:
                raise RuntimeError("Engine is closed")
            if not prepared.items:
                return np.empty((0, 0)), np.empty((0, 2)), {"graph_miss": False}
            shape = self._shape(prepared)
            if self.backend == "eager":
                host = self._allocate(shape, host=True)
                self._fill(host, prepared)
                outputs = self._forward({k: v.to(self.device, non_blocking=True) for k, v in host.items()})
                miss, build_ms = False, 0.
            else:
                slot, miss = self._slot(shape, prepared)
                self._fill(slot.host, prepared)
                for key in KEYS:
                    slot.inputs[key].copy_(slot.host[key], non_blocking=True)
                slot.graph.replay()
                outputs = slot.outputs
                build_ms = slot.build_ms if miss else 0.
            count = len(prepared.items)
            logits, act = [o[:count].float().cpu().numpy() for o in outputs]
            return logits, act, {"graph_miss": miss, "graph_build_ms": build_ms,
                                 "shape": list(shape), "backend": self.backend}

    def predict(self, state, questions):
        start = time.perf_counter()
        prepared = self.prepare(state, questions)
        logits, act, metrics = self.run_prepared(prepared)
        result = format_response(prepared, logits, act, self.agent.temperature,
                                 self.agent.temperature_by_options)
        metrics["total_ms"] = (time.perf_counter()-start)*1000
        result["engine"] = metrics
        return result

    system_one = predict

    def warmup(self, state, questions):
        return self.predict(state, questions)["engine"]

    def close(self):
        with self.lock:
            torch.cuda.synchronize(self.device)
            self.graphs.clear()
            self.model = None
            self.agent.model = None
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
