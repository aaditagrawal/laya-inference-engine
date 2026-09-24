"""Exact NumPy arithmetic orchestrated from C++ with a guarded reference fallback.

The private NumPy loop capsule ABI is pinned to 2.5.3. The native path handles
ordinary contiguous FP32 outputs with finite bounded values and ordinary prepared
requests. Other values/shapes retain reference formatting and error behavior.
"""

import hashlib
import importlib.util
import json
from pathlib import Path
from time import perf_counter
from types import MethodType

import numpy as np

from laya_blackwell.protocol import PreparedRequest, format_response


class NativeFormatter:
    def __init__(self):
        if np.__version__ != "2.5.3":
            raise RuntimeError("Native formatter requires NumPy 2.5.3")
        root = Path(__file__).resolve().parents[2]
        record = json.loads(
            (root / ".research/frontier-native-format/build.json").read_text()
        )
        if record["returncode"] != 0 or record["numpy"] != np.__version__:
            raise RuntimeError(
                "Native formatter needs a successful build for this NumPy version"
            )
        source = Path(__file__).with_suffix(".cpp")
        library = Path(record["artifact"])
        if hashlib.sha256(source.read_bytes()).hexdigest() != record["source_sha256"]:
            raise RuntimeError("Rebuild native formatting after changing its source")
        if hashlib.sha256(library.read_bytes()).hexdigest() != record["library_sha256"]:
            raise RuntimeError("Native formatter library digest changed")
        spec = importlib.util.spec_from_file_location("laya_native_format", library)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        self.native = module.Formatter()
        self.build = record

    def __call__(self, prepared, logits, actions, temperature, buckets):
        if type(prepared) is PreparedRequest:
            result = self.native(prepared, logits, actions, temperature, buckets)
            if result is not None:
                return result
        return format_response(prepared, logits, actions, temperature, buckets)


def install(engine):
    formatter = NativeFormatter()
    previous = engine.adapter.predict

    def predict(adapter, state, questions):
        start = perf_counter()
        prepared = adapter.prepare(state, questions)
        logits, actions, metrics = adapter.run_prepared(prepared)
        result = formatter(
            prepared,
            logits,
            actions,
            adapter.agent.temperature,
            adapter.agent.temperature_by_options,
        )
        metrics["total_ms"] = (perf_counter() - start) * 1000
        result["engine"] = metrics
        return result

    engine.adapter.predict = MethodType(predict, engine.adapter)
    return previous
