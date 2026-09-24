"""One-shape request engine backed by an AOT package, without model loading."""

import json
import threading
from collections import OrderedDict
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace

import torch
from laya.common import clamp_temperature

from experiments.latency.serving.graph_adapter import StableHostAdapter
from experiments.native.compiler import pin_libdevice, preserve_ops  # noqa: F401
from experiments.native.kernels.candidates import build_vector
from laya_blackwell.engine import KEYS, BlackwellEngine, hardware_info

from .tokenizer import RuntimeTokenizer


class PackageBase(BlackwellEngine):
    """Reuse request packing rules but never invoke BlackwellEngine.__init__."""

    def __init__(self, package, *, loader="standard"):
        package = Path(package).resolve()
        manifest = json.loads(package.with_suffix(".manifest.json").read_text())
        runtime = package.parent / manifest["runtime_directory"]
        cfg = json.loads((runtime / "rl_agent_config.json").read_text())
        self.device = torch.device("cuda:0")
        self.hardware = hardware_info(self.device)
        if self.hardware["compute_capability"] != "12.0":
            raise RuntimeError("This AOT artifact was tested only on SM120")
        if manifest["torch"] != torch.__version__:
            raise RuntimeError("Use the PyTorch build recorded by the AOT artifact")
        self.fixed_graph_key = tuple(manifest["shape"]) + (manifest["unmasked"],)
        self.max_questions = 64
        self.sequence_buckets = tuple(manifest["sequence_buckets"])
        self.max_graphs = 1
        self.backend = "aot-package"
        self.lock = threading.RLock()
        self.graphs = OrderedDict()
        self.closed = False
        tokenizer = RuntimeTokenizer(runtime / "tokenizer")
        self.tokenizer_loader = "serialized-rust-tokenizer"
        self.agent = SimpleNamespace(
            tok=tokenizer,
            cfg=cfg,
            temperature=[
                clamp_temperature(t) for t in cfg.get("temperature", [1, 1, 1])
            ],
            temperature_by_options={
                key: clamp_temperature(value)
                for key, value in cfg.get("temperature_by_options", {}).items()
            },
            model=None,
        )
        library = pin_libdevice()
        if library["sha256"] != manifest["libdevice"]["sha256"]:
            raise RuntimeError("AOT math library differs from its build manifest")
        build_vector()
        self.loader = loader
        if loader == "standard":
            self.model = torch._inductor.aoti_load_package(
                str(package), run_single_threaded=True
            )
        elif loader == "checked-cpp":
            from .runtime import load_checked_package

            self.model, self.aot_target = load_checked_package(
                package, expected_torch=manifest["torch"]
            )
        else:
            raise ValueError("loader must be standard or checked-cpp")

    def _graph_key(self, prepared):
        key = super()._graph_key(prepared)
        if key != self.fixed_graph_key:
            raise ValueError(
                f"AOT package supports graph key {self.fixed_graph_key}; request needs {key}. "
                "Build the corresponding shape and mask specialization."
            )
        return key

    def _forward(self, inputs):
        if inputs["global_attention_unmasked"] != self.fixed_graph_key[-1]:
            raise ValueError("AOT global-attention specialization mismatch")
        return self.model(*(inputs[key] for key in KEYS))


class PackageEngine:
    def __init__(self, package, *, loader="standard"):
        self.base = self.adapter = None
        self.closed = False
        self._close_lock = threading.RLock()
        try:
            self.base = PackageBase(package, loader=loader)
            self.adapter = StableHostAdapter(
                self.base, mode="cpp-graph-io", max_graphs=1
            )
            self.agent = self.base.agent
            self.hardware = self.base.hardware
        except BaseException as error:
            try:
                self.close()
            except BaseException as cleanup_error:
                error.add_note(f"Constructor cleanup also failed: {cleanup_error!r}")
                raise error from cleanup_error
            raise

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
