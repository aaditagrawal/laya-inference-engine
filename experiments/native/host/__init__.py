"""Experimental native host adapters; the supplied base engine remains caller-owned.

Build with host/build.py before importing. Close the adapter before the base.
"""

from .adapter import HostAdapter, RustTokenizerHostAdapter

__all__ = ["HostAdapter", "RustTokenizerHostAdapter"]
