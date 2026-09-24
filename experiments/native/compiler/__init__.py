"""Optional RTX 5070 Ti compiler and exact-window experiments."""

from .gemm import install_gemm_autotune
from .policy import install_compiler, pin_libdevice
from .window import install_window

install_padded_rope_window = install_window
install_precise_compile = install_compiler

__all__ = [
    "install_compiler",
    "install_gemm_autotune",
    "install_padded_rope_window",
    "install_precise_compile",
    "install_window",
    "pin_libdevice",
]
