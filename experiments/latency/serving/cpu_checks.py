"""CPU-only scheduler bounds, output ownership, and in-flight close checks."""

import json
import threading
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from laya_blackwell.protocol import PreparedRequest

from . import service


def prepared(n, value=0):
    return PreparedRequest(
        [], [], [{"ids": [value], "markers": [0], "qtype": 0} for _ in range(n)], n
    )


def main():
    report = {}
    scheduler = service.MicrobatchService.__new__(service.MicrobatchService)
    scheduler.max_requests, scheduler.max_questions = 8, 64
    key = (32, 64, 4, False)
    scheduler.queue = deque(
        service.Pending(prepared(n), key, 0, Future()) for n in (17, 32, 32)
    )
    selected, _ = scheduler._select()
    assert sum(len(x.prepared.items) for x in selected) == 49
    assert len(selected) == 2
    report["mixed_actual_17_32_rows_with_shared_batch32_key"] = (
        "49 rows selected; third 32-row request deferred"
    )

    key = (4, 512, 4, True)
    scheduler.queue = deque(
        service.Pending(prepared(4), key, 0, Future()) for _ in range(3)
    )
    selected, _ = scheduler._select()
    assert len(selected) == 2
    report["unmasked_power_of_two_batch_count"] = (
        "Three eligible requests select two to retain unmasked attention"
    )

    entered, release = threading.Event(), threading.Event()

    class FakeAdapter:
        def __init__(self, base, **_):
            self.lock = threading.RLock()
            self.graphs = {}
            self.closed = False

        def prepare(self, state, questions):
            return prepared(1, state)

        def run_prepared(self, p):
            entered.set()
            assert release.wait(timeout=5)
            return (
                np.array([[x["ids"][0]] for x in p.items], dtype=np.float32),
                np.zeros((len(p.items), 2)),
                {},
            )

        def close(self):
            self.closed = True

    base = SimpleNamespace(
        agent=SimpleNamespace(),
        device="cpu",
        max_questions=64,
        _graph_key=lambda p: (1, 64, 2, False),
    )
    for kind in ("microbatch", "streams"):
        entered.clear()
        release.clear()
        with (
            patch.object(service, "StableHostAdapter", FakeAdapter),
            patch.object(torch.cuda, "Stream", lambda **_: object()),
            patch.object(torch.cuda, "stream", lambda _: nullcontext()),
        ):
            target = (
                service.MicrobatchService(base, wait_ms=0)
                if kind == "microbatch"
                else service.StreamService(base, lanes=2)
            )
            with ThreadPoolExecutor(max_workers=3) as pool:
                request = pool.submit(target.run_prepared, prepared(1, 42))
                assert entered.wait(timeout=5)
                closed1 = pool.submit(target.close)
                with target.cv:
                    # A notify is not required here: the in-flight worker remains
                    # blocked, and the condition wait releases the GIL for close.
                    target.cv.wait_for(lambda target=target: target.closed, timeout=0.1)
                assert target.closed and not closed1.done()
                closed2 = pool.submit(target.close)
                assert not closed2.done()
                release.set()
                output = request.result(timeout=5)
                assert output[0][0, 0] == 42
                closed1.result(timeout=5)
                closed2.result(timeout=5)
            try:
                target.run_prepared(prepared(1))
            except RuntimeError:
                pass
            else:
                raise AssertionError("closed service accepted a request")
            report[kind + "_concurrent_close"] = (
                "Accepted request drained, two close callers synchronized, closed reuse rejected"
            )
            if kind == "streams":
                target = service.StreamService(base, lanes=2)
                with patch.object(target.cv, "wait_for", side_effect=KeyboardInterrupt):
                    try:
                        target.run_prepared(prepared(1, 43))
                    except KeyboardInterrupt:
                        pass
                    else:
                        raise AssertionError(
                            "Admission interruption was not propagated"
                        )
                assert target.active == 0 and all(
                    not q for q in target.waiters.values()
                )
                assert target.run_prepared(prepared(1, 43))[0][0, 0] == 43
                target.close()
                report["interrupted_lane_wait"] = (
                    "Interrupted ticket removed; later calls still complete"
                )
    from .graph_adapter import OwnedSlot

    events = []

    class Resource:
        def __del__(self):
            events.append("buffers released")

    slot = OwnedSlot(
        graph=SimpleNamespace(reset=lambda: events.append("graph reset")),
        outputs=Resource(),
        capture_owner=SimpleNamespace(close=lambda: events.append("stream destroyed")),
    )
    slot.close()
    assert events == ["graph reset", "buffers released", "stream destroyed"]
    report["capture_resource_release_order"] = events
    report["cuda_initialized"] = torch.cuda.is_initialized()
    assert not report["cuda_initialized"]
    report["status"] = "passed"
    path = (
        Path(__file__).resolve().parents[3]
        / "results/latency-optimizations/serving/cpu-checks.json"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
