"""Reuse owned NumPy staging views and avoid Torch dispatch on graph-cache hits."""

from types import MethodType

import numpy as np
import torch


def install(engine):
    adapter = engine.adapter
    if adapter.mode != "cpp-graph-io":
        raise ValueError("Host runtime experiment requires cpp-graph-io")
    previous = adapter.run_prepared
    capture_slot = adapter._slot

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
                slot, miss = capture_slot(prepared)
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
                    "backend": "host-" + self.mode,
                },
            )

    adapter.run_prepared = MethodType(run_prepared, adapter)
    return previous
