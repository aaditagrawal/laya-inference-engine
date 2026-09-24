"""Opt-in exact BF16 preset extracted from the validated low-latency runtime."""

import json
import threading
import types
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch

from laya_blackwell.engine import MODEL_ID, REVISION, BlackwellEngine, hardware_info


class _ShortShapeCompiler(torch.nn.Module):
    def __init__(self, original, compiled):
        super().__init__()
        self.original = original
        self.compiled = compiled

    def forward(self, input_ids, *args, **kwargs):
        model = self.compiled if input_ids.numel() == 64 else self.original
        return model(input_ids, *args, **kwargs)


def _install_projections(model, selections):
    from .matmul import matmul
    from .tma import compiled_matmul

    for field, selected in selections.items():
        group, attr = field.split(".")
        config = tuple(selected["config"])
        for layer in model.net.encoder.layers:
            module = getattr(getattr(layer, group), attr)
            if module.bias is not None:
                raise ValueError("Fast mode requires bias-free encoder projections")
            original = module.forward
            if selected["variant"] == "tma":

                def forward(module, x, original=original, config=config):
                    if x.numel() // x.shape[-1] == 64 and x.is_contiguous():
                        return compiled_matmul(
                            x, module.weight, [int(v) for v in config]
                        )
                    return original(x)

            else:

                def forward(
                    module,
                    x,
                    original=original,
                    config=config,
                    partial_bf16=selected["variant"] == "bf16-partial",
                ):
                    if x.numel() // x.shape[-1] == 64 and x.is_contiguous():
                        return matmul(
                            x, module.weight, config, partial_bf16=partial_bf16
                        )
                    return original(x)

            module.forward = types.MethodType(forward, module)


class FastEngine:
    """The fixed SM120 preset, with owned graphs and copied public outputs.

    Build the native extensions with ``laya-blackwell build-fast`` first.
    Construction precomputes frozen token tables. The first short request
    compiles and captures its shape; subsequent requests replay the graph.
    This mode preserves the validated BF16 arithmetic and uses the original
    execution for shapes outside its specialization.
    """

    def __init__(
        self,
        model=MODEL_ID,
        *,
        revision=REVISION,
        subfolder=None,
        device="cuda:0",
        max_graphs=8,
        max_questions=64,
        sequence_buckets=(64, 128, 256, 384, 512, 768, 1024),
        backend="fused",
    ):
        if model != MODEL_ID or revision != REVISION or subfolder is not None:
            raise ValueError(
                "Fast mode requires the pinned convaiinnovations/laya checkpoint"
            )
        if backend != "fused":
            raise ValueError("Fast mode supports only the fused BF16 backend")
        if np.__version__ != "2.5.3":
            raise RuntimeError("Fast mode requires NumPy 2.5.3; install the fast extra")
        if max_graphs < 1 or max_questions < 1:
            raise ValueError("max_graphs and max_questions must be positive")
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("Fast mode requires a CUDA device")
        self.hardware = hardware_info(self.device)
        if self.hardware["compute_capability"] != "12.0":
            raise RuntimeError("Fast mode is validated only for SM120 Blackwell GPUs")
        from .paths import require_build

        self.build = require_build()
        self.selection = json.loads(Path(__file__).with_name("config.json").read_text())
        self.base = self.adapter = None
        self.closed = False
        self._close_lock = threading.RLock()
        self.backend = self.policy = "fast"
        self.max_graphs, self.max_questions = max_graphs, max_questions
        self.revision = revision
        try:
            with torch.cuda.device(self.device):
                self.base = BlackwellEngine(
                    model=model,
                    revision=revision,
                    device=self.device,
                    max_graphs=max_graphs,
                    max_questions=max_questions,
                    sequence_buckets=sequence_buckets,
                    backend="fused",
                )
                self._install()
                self.agent = self.base.agent
        except BaseException as error:
            try:
                self.close()
            except BaseException as cleanup_error:
                error.add_note(f"Constructor cleanup also failed: {cleanup_error!r}")
                raise error from cleanup_error
            raise

    def _install(self):
        from . import (
            attention,
            attention_special,
            global_attention,
            head,
            mlp_geglu,
            norm,
            reduce_norm,
            token_tables,
        )
        from .compiler import install_compiler
        from .host import HostAdapter
        from .window import install_window

        norm.install(self.base)
        install_window(self.base)
        self.adapter = HostAdapter(self.base, max_graphs=self.max_graphs)
        model = self.base.model
        _install_projections(model, self.selection["encoder"])
        self.selection["token_tables"] = token_tables.install(model)
        reduce_norm.install(model, self.selection["encoder"]["mlp.Wo"])
        mlp_geglu.install(model, self.selection["mlp_geglu"])
        head.install(model, self.selection["head"])
        install_compiler(self.base)
        self.base.model = _ShortShapeCompiler(model, self.base.model).eval()
        # Compiler preservation copies the functional namespace. Install the
        # attention dispatch afterward and before the first trace/capture.
        attention.install(model)
        attention_special.install(
            self, tuple(self.selection["local_attention"]), include_padding=True
        )
        global_attention.install(
            self, self.selection["global_attention"], include_padding=True
        )

    def _check_open(self):
        if self.closed:
            raise RuntimeError("Engine is closed")

    def prepare(self, state, questions):
        self._check_open()
        return self.adapter.prepare(state, questions)

    def run_prepared(self, prepared):
        self._check_open()
        return self.adapter.run_prepared(prepared)

    def predict(self, state, questions):
        self._check_open()
        return self.adapter.predict(state, questions)

    system_one = predict

    def warmup(self, state, questions):
        return self.predict(state, questions)["engine"]

    def close(self):
        with self._close_lock:
            if self.closed:
                return
            self.closed = True
            with ExitStack() as cleanup:
                if self.base is not None:
                    cleanup.callback(self.base.close)
                if self.adapter is not None:
                    cleanup.callback(self.adapter.close)

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, *_):
        self.close()
