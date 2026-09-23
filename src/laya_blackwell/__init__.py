"""Laya inference specialized for NVIDIA Blackwell."""
import os

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

__version__ = "0.1.0"


def __getattr__(name):
    if name == "BlackwellEngine":
        from .engine import BlackwellEngine
        return BlackwellEngine
    raise AttributeError(name)
