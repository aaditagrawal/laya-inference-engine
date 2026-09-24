"""Pinned NumPy arithmetic in C++ with the reference-formatting fallback."""

import numpy as np

from laya_blackwell.protocol import PreparedRequest, format_response

from .native import load_python_extension


class NativeFormatter:
    def __init__(self):
        if np.__version__ != "2.5.3":
            raise RuntimeError("Fast mode requires NumPy 2.5.3")
        self.native = load_python_extension("format").Formatter()

    def __call__(self, prepared, logits, actions, temperature, buckets):
        if type(prepared) is PreparedRequest:
            result = self.native(prepared, logits, actions, temperature, buckets)
            if result is not None:
                return result
        return format_response(prepared, logits, actions, temperature, buckets)
