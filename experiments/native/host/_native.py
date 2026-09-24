"""Load the locally built extension while keeping binaries outside source trees."""

import importlib.util
import os
import sys
import sysconfig
from pathlib import Path


def default_build_directory():
    return Path(__file__).resolve().parents[3] / ".research" / "native-build" / "host"


def load_native():
    name = "laya_native_host"
    if name in sys.modules:
        return sys.modules[name]
    suffix = sysconfig.get_config_var("EXT_SUFFIX")
    path = Path(
        os.environ.get(
            "LAYA_HOST_EXTENSION",
            default_build_directory() / (name + suffix),
        )
    )
    if not path.is_file():
        raise ImportError(
            "The native host extension has not been built. Run "
            "uv run --no-sync python experiments/native/host/build.py "
            "under the shared experiment build lock."
        )
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load native host extension from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[name] = module
    return module
