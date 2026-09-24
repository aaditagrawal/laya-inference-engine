"""Load verified extensions without importing or compiling research code."""

import importlib.util
import sys

from .paths import artifact_path


def load_python_extension(kind):
    name = {"host": "laya_fast_host", "format": "laya_fast_format"}[kind]
    if name in sys.modules:
        return sys.modules[name]
    path = artifact_path(kind)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(
            f"Cannot load {name} from {path}; run laya-blackwell build-fast"
        )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[name] = module
    return module
