"""Opt-in local engine. Build the native extensions before construction."""

import threading
from contextlib import ExitStack

from laya_blackwell.engine import BlackwellEngine


class ExperimentalEngine:
    """Own both the model and the host adapter, with deterministic cleanup.

    This research wrapper supports the tested English checkpoint on SM120.
    It does not change the installed laya_blackwell package or its defaults.
    """

    def __init__(
        self,
        *,
        mode="native-window",
        kernel="cuda_vector_norm_triton_geglu_corrected",
        max_graphs=8,
        **kwargs,
    ):
        if mode not in {"native", "native-window", "compiled", "autotuned"}:
            raise ValueError(
                "mode must be native, native-window, compiled, or autotuned"
            )
        if kwargs.get("backend", "fused") != "fused":
            raise ValueError("The experimental kernels require the fused BF16 backend")
        self.base = self.adapter = None
        self.mode = mode
        self.closed = False
        self._close_lock = threading.RLock()
        try:
            # Importing the host adapter verifies that its extension was built.
            from .host import RustTokenizerHostAdapter
            from .kernels import install

            self.base = BlackwellEngine(max_graphs=max_graphs, **kwargs)
            if self.base.hardware["compute_capability"] != "12.0":
                raise RuntimeError(
                    "These experiments have been validated only on SM120"
                )
            install(self.base, kernel)
            if mode in {"native-window", "compiled", "autotuned"}:
                from .compiler import install_padded_rope_window

                install_padded_rope_window(self.base)
            if mode == "compiled":
                from .compiler import install_precise_compile

                install_precise_compile(self.base)
            if mode == "autotuned":
                from .compiler import install_gemm_autotune

                install_gemm_autotune(self.base, tma=True)
            self.adapter = RustTokenizerHostAdapter(
                self.base, mode="cpp-graph-io", max_graphs=max_graphs
            )
            self.agent, self.device = self.base.agent, self.base.device
            self.backend = f"experimental-{mode}"
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

    system_one = predict

    def warmup(self, state, questions):
        return self.predict(state, questions)["engine"]

    def close(self):
        with self._close_lock:
            if self.closed:
                return
            self.closed = True
            # ExitStack attempts both cleanups even if the adapter raises.
            # Register the base first so its model outlives the adapter graphs.
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
