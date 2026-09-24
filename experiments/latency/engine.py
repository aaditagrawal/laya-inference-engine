"""Opt-in SM120 engine combining the accepted second-round experiments."""

import json
from pathlib import Path

from experiments.native.engine import ExperimentalEngine

from .fusion.adapter import install as install_fusion
from .serving.graph_adapter import replace_adapter


class V2Engine(ExperimentalEngine):
    """Unchanged batch geometry, owned streams, optional projection fusion.

    Denser padding and request microbatching remain separate research options.
    Numerical parity is measured on synthetic fixtures, not all possible inputs.
    """

    def __init__(self, *, optimization="fused", **kwargs):
        if optimization not in {"native", "fused"}:
            raise ValueError("optimization must be native or fused")
        if "mode" in kwargs:
            raise ValueError("Use optimization for this wrapper, not mode")
        if (
            kwargs.get("kernel", "cuda_vector_norm_triton_geglu_corrected")
            != "cuda_vector_norm_triton_geglu_corrected"
        ):
            raise ValueError(
                "V2Engine requires the validated CUDA vector norm and corrected GEGLU"
            )
        super().__init__(mode="native-window", **kwargs)
        try:
            replace_adapter(self)
            if optimization == "fused":
                config = json.loads(
                    (Path(__file__).parent / "fusion/best.json").read_text()
                )
                install_fusion(self, **config)
            self.optimization = optimization
        except BaseException as error:
            try:
                self.close()
            except BaseException as cleanup_error:  # noqa: BLE001 - preserve original error
                error.add_note(f"Constructor cleanup also failed: {cleanup_error!r}")
            raise
