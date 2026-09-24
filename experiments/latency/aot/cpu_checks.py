"""Exercise constructor failure and teardown ownership without CUDA allocation."""

import json
from unittest.mock import patch

from . import engine as module


def main():
    events = []

    class Base:
        agent = hardware = None

        def __init__(self, *_args, **_kwargs):
            pass

        def close(self):
            events.append("base")

    class Adapter:
        def __init__(self, *_args, **_kwargs):
            pass

        def close(self):
            events.append("adapter")

    with patch.object(module, "PackageBase", Base):
        with patch.object(
            module, "StableHostAdapter", side_effect=RuntimeError("init failure")
        ):
            try:
                module.PackageEngine("unused")
            except RuntimeError as error:
                assert str(error) == "init failure"
            else:
                raise AssertionError("Expected adapter initialization failure")
            assert events == ["base"]
        events.clear()
        with patch.object(module, "StableHostAdapter", Adapter):
            engine = module.PackageEngine("unused")
            engine.close()
            engine.close()
            assert events == ["adapter", "base"]
            for name, args in [
                ("prepare", ("", {})),
                ("predict", ("", {})),
                ("run_prepared", (None,)),
            ]:
                try:
                    getattr(engine, name)(*args)
                except RuntimeError as error:
                    assert str(error) == "Engine is closed"
                else:
                    raise AssertionError(f"{name} accepted work after close")
        events.clear()
        with patch.object(module, "StableHostAdapter", Adapter):
            engine = module.PackageEngine("unused")
            with patch.object(
                engine.adapter, "close", side_effect=RuntimeError("close failure")
            ):
                try:
                    engine.close()
                except RuntimeError as error:
                    assert str(error) == "close failure"
                else:
                    raise AssertionError("Expected adapter close failure")
                assert events == ["base"]
                engine.close()
                assert events == ["base"]
    print(
        json.dumps(
            {
                "constructor_failure_cleanup": True,
                "idempotent_close": True,
                "use_after_close_rejected": True,
                "base_cleanup_after_adapter_failure": True,
            }
        )
    )


if __name__ == "__main__":
    main()
