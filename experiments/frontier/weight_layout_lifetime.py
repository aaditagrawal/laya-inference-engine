"""Check compiler/graph weight lifetime using torch-owned storage first."""

import fcntl
import gc
import json
from pathlib import Path

import torch

from experiments.native import common

from .engine import FrontierEngine
from .weight_layout import WeightLayout


def main():
    with Path("/tmp/laya-gpu-experiments.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        torch.set_num_threads(4)
        constructor = json.loads(Path("results/frontier/summary.json").read_text())[
            "recommended_constructor"
        ]
        layout = WeightLayout()
        engine = FrontierEngine(**constructor, max_graphs=2)
        layout.install(engine, "torch-bank")
        engine.predict(**common.workload(1, "short"))
        engine.close()
        del engine
        try:
            layout.close()
            print(json.dumps(layout.report["cleanup"]), flush=True)
        except RuntimeError:
            for name, ref in layout.references:
                parameter = ref()
                if parameter is not None:
                    refs = gc.get_referrers(parameter)
                    print(
                        "REFERENCE",
                        name,
                        [type(value).__name__ for value in refs],
                        flush=True,
                    )
                    for value in refs:
                        if isinstance(value, dict):
                            print("DICT_KEYS", [str(k) for k in value][:30], flush=True)
                            for owner in gc.get_referrers(value):
                                if isinstance(owner, dict) and "_parameters" in owner:
                                    for module in gc.get_referrers(owner):
                                        if isinstance(module, torch.nn.Module):
                                            print(
                                                "MODULE",
                                                type(module).__name__,
                                                flush=True,
                                            )
                                            for holder in gc.get_referrers(module):
                                                print(
                                                    "HOLDER",
                                                    type(holder).__name__,
                                                    flush=True,
                                                )
                                                if isinstance(holder, tuple):
                                                    for upper in gc.get_referrers(
                                                        holder
                                                    ):
                                                        if callable(upper):
                                                            print(
                                                                "CALLABLE",
                                                                getattr(
                                                                    upper,
                                                                    "__qualname__",
                                                                    type(
                                                                        upper
                                                                    ).__name__,
                                                                ),
                                                                flush=True,
                                                            )
                                                if isinstance(holder, dict):
                                                    print(
                                                        "HOLDER_KEYS",
                                                        [str(k) for k in holder][:20],
                                                        flush=True,
                                                    )
                        elif isinstance(value, (list, tuple)):
                            print(
                                "SEQUENCE",
                                len(value),
                                [type(v).__name__ for v in value[:8]],
                                flush=True,
                            )
                    break
            raise


if __name__ == "__main__":
    main()
