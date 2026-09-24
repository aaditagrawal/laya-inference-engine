"""Serving experiments sharing an immutable model, with privately owned buffers.

The caller owns ``base`` and must keep it open until every service is closed.
Graph capture is globally exclusive within a StreamService; warmed graph replay
may overlap on separate streams. Separate services sharing one base must not be
used concurrently, because their capture gates do not coordinate with each other.
"""

import threading
from collections import deque
from concurrent.futures import Future
from contextlib import contextmanager
from dataclasses import dataclass
from time import perf_counter

import torch

from laya_blackwell.protocol import PreparedRequest, format_response

from .graph_adapter import StableHostAdapter


class CaptureGate:
    """Concurrent replay readers; captures/cleanup require an exclusive writer."""

    def __init__(self):
        self.cv = threading.Condition()
        self.readers = self.waiting_writers = 0
        self.writer = False

    @contextmanager
    def access(self, exclusive=False):
        with self.cv:
            if exclusive:
                self.waiting_writers += 1
                try:
                    self.cv.wait_for(lambda: not self.writer and not self.readers)
                    self.writer = True
                finally:
                    self.waiting_writers -= 1
            else:
                self.cv.wait_for(lambda: not self.writer and not self.waiting_writers)
                self.readers += 1
        try:
            yield
        finally:
            with self.cv:
                if exclusive:
                    self.writer = False
                else:
                    self.readers -= 1
                self.cv.notify_all()


class StreamService:
    """FIFO lane admission; each lane has a separate graph, stream and buffers.

        Preparation and response formatting run in the calling threads. The existing
    native graph launcher releases the GIL while synchronizing its own CUDA stream.
    No model weights are copied and no output is cached. New shapes safely drain
    active replays before capture; this is measured as part of an unwarmed request.
    """

    def __init__(self, base, lanes=2, max_graphs=16, small_lanes=0):
        if lanes < 1 or max_graphs < 1:
            raise ValueError("lanes and max_graphs must be positive")
        if small_lanes < 0 or small_lanes >= lanes:
            raise ValueError("small_lanes must leave at least one bulk lane")
        self.base, self.agent, self.device = base, base.agent, base.device
        self.cv = threading.Condition()
        self.close_lock = threading.Lock()
        self.closed = False
        self.active = 0
        self.small_lanes = small_lanes
        self.waiters = {0: deque(), 1: deque()}
        self.free = deque(range(lanes))
        self.gate = CaptureGate()
        self.adapters, self.streams = [], []
        try:
            for lane in range(lanes):
                self.adapters.append(StableHostAdapter(base, max_graphs=max_graphs))
                self.streams.append(
                    torch.cuda.Stream(
                        device=self.device, priority=-1 if lane < small_lanes else 0
                    )
                )
        except BaseException:
            for adapter in self.adapters:
                adapter.close()
            raise

    @contextmanager
    def _request(self):
        with self.cv:
            if self.closed:
                raise RuntimeError("Service is closed")
            self.active += 1
        try:
            yield
        finally:
            with self.cv:
                self.active -= 1
                self.cv.notify_all()

    def prepare(self, state, questions):
        with self._request():
            return self.adapters[0].prepare(state, questions)

    def _run_lane(self, prepared, lane):
        adapter = self.adapters[lane]
        with adapter.lock:
            miss = (
                bool(prepared.items)
                and self.base._graph_key(prepared) not in adapter.graphs
            )
            with (
                self.gate.access(exclusive=miss),
                torch.cuda.stream(self.streams[lane]),
            ):
                return adapter.run_prepared(prepared)

    def _run(self, prepared):
        start = perf_counter()
        ticket = object()
        group = int(bool(self.small_lanes) and len(prepared.items) > 1)
        allowed = self._lanes(prepared)
        waiters = self.waiters[group]
        with self.cv:
            waiters.append(ticket)
            try:
                self.cv.wait_for(
                    lambda: (
                        waiters[0] is ticket
                        and any(lane in allowed for lane in self.free)
                    )
                )
            except BaseException:
                waiters.remove(ticket)
                self.cv.notify_all()
                raise
            waiters.popleft()
            lane = next(lane for lane in self.free if lane in allowed)
            self.free.remove(lane)
            self.cv.notify_all()
        queued_ms = (perf_counter() - start) * 1000
        try:
            logits, actions, metrics = self._run_lane(prepared, lane)
            metrics.update(
                lane=lane,
                queue_ms=queued_ms,
                service="priority-streams" if self.small_lanes else "streams",
            )
            return logits, actions, metrics
        finally:
            with self.cv:
                self.free.append(lane)
                self.cv.notify_all()

    def run_prepared(self, prepared):
        with self._request():
            return self._run(prepared)

    def predict(self, state, questions):
        start = perf_counter()
        with self._request():
            prepared = self.adapters[0].prepare(state, questions)
            logits, actions, metrics = self._run(prepared)
            result = format_response(
                prepared,
                logits,
                actions,
                self.agent.temperature,
                self.agent.temperature_by_options,
            )
            metrics["total_ms"] = (perf_counter() - start) * 1000
            result["engine"] = metrics
            return result

    def warm(self, requests):
        with self._request():
            for request in requests:
                prepared = self.adapters[0].prepare(**request)
                for lane in self._lanes(prepared):
                    self._run_lane(prepared, lane)

    def _lanes(self, prepared):
        if not self.small_lanes:
            return range(len(self.adapters))
        return (
            range(self.small_lanes)
            if len(prepared.items) <= 1
            else range(self.small_lanes, len(self.adapters))
        )

    def close(self):
        with self.close_lock:
            with self.cv:
                if self.closed:
                    return
                self.closed = True
                self.cv.wait_for(lambda: self.active == 0)
            with self.gate.access(exclusive=True):
                for adapter in self.adapters:
                    adapter.close()


def merge_prepared(requests):
    """Only model rows are combined; each original request retains its IDs."""
    items = [item for request in requests for item in request.items]
    return PreparedRequest(
        [], [], items, sum(request.input_tokens for request in requests)
    )


@dataclass
class Pending:
    prepared: PreparedRequest
    key: tuple
    enqueued: float
    future: Future


class MicrobatchService:
    """One GPU worker; bounded FIFO waiting and batching by original graph key.

    The max-wait is only a batch-formation delay, not a request deadline. Queueing
    behind GPU work can exceed it. A bounded queue applies backpressure to callers.
    Full/unmasked batches use a power-of-two request count, retaining the original
    attention mask mode. GEMM geometry still changes and requires separate parity
    assessment. It is NOT asserted to produce bit-identical BF16 model outputs.
    """

    def __init__(
        self,
        base,
        max_requests=8,
        max_questions=64,
        wait_ms=0.25,
        max_pending=64,
        max_graphs=24,
    ):
        if min(max_requests, max_questions, max_pending, max_graphs) < 1 or wait_ms < 0:
            raise ValueError("positive bounds and a nonnegative wait are required")
        self.base, self.agent, self.device = base, base.agent, base.device
        self.max_requests = max_requests
        self.max_questions = min(max_questions, base.max_questions)
        self.wait_ms, self.max_pending = wait_ms, max_pending
        self.adapter = StableHostAdapter(base, max_graphs=max_graphs)
        self.cv = threading.Condition()
        self.close_lock = threading.Lock()
        self.queue = deque()
        self.closed = False
        self.active = 0
        self.worker = threading.Thread(
            target=self._work, name="laya-microbatch", daemon=True
        )
        self.worker.start()

    @contextmanager
    def _request(self):
        with self.cv:
            if self.closed:
                raise RuntimeError("Service is closed")
            self.active += 1
        try:
            yield
        finally:
            with self.cv:
                self.active -= 1
                self.cv.notify_all()

    def prepare(self, state, questions):
        with self._request():
            return self.adapter.prepare(state, questions)

    def _enqueue(self, prepared):
        key = tuple(self.base._graph_key(prepared)) if prepared.items else None
        pending = Pending(prepared, key, perf_counter(), Future())
        with self.cv:
            # Accepted in-flight callers complete even after close is requested.
            self.cv.wait_for(lambda: len(self.queue) < self.max_pending)
            self.queue.append(pending)
            self.cv.notify_all()
        return pending.future.result()

    def run_prepared(self, prepared):
        with self._request():
            return self._enqueue(prepared)

    def predict(self, state, questions):
        start = perf_counter()
        with self._request():
            prepared = self.adapter.prepare(state, questions)
            logits, actions, metrics = self._enqueue(prepared)
            result = format_response(
                prepared,
                logits,
                actions,
                self.agent.temperature,
                self.agent.temperature_by_options,
            )
            metrics["total_ms"] = (perf_counter() - start) * 1000
            result["engine"] = metrics
            return result

    def _select(self):
        first = self.queue[0]
        n = len(first.prepared.items)
        cap = min(self.max_requests, max(1, self.max_questions // max(n, 1)))
        same, questions = [], 0
        for item in self.queue:
            count = len(item.prepared.items)
            if item.key == first.key and (
                not same or questions + count <= self.max_questions
            ):
                same.append(item)
                questions += count
                if len(same) == cap:
                    break
        if first.key and first.key[-1]:
            # Equal original batch sizes and a power-of-two count preserve the
            # unmasked SDPA path. No dummy mask row is introduced here.
            count = 1 << (len(same).bit_length() - 1)
            same = same[:count]
        return same, cap

    def _work(self):
        while True:
            with self.cv:
                self.cv.wait_for(
                    lambda: self.queue or (self.closed and self.active == 0)
                )
                if not self.queue:
                    return
                while True:
                    selected, cap = self._select()
                    remaining = (
                        self.queue[0].enqueued + self.wait_ms / 1000 - perf_counter()
                    )
                    if len(selected) >= cap or remaining <= 0:
                        break
                    self.cv.wait(timeout=remaining)
                selected_ids = {id(item) for item in selected}
                self.queue = deque(
                    item for item in self.queue if id(item) not in selected_ids
                )
                self.cv.notify_all()
            start = perf_counter()
            try:
                joined = merge_prepared([item.prepared for item in selected])
                logits, actions, base_metrics = self.adapter.run_prepared(joined)
                offset = 0
                for item in selected:
                    count = len(item.prepared.items)
                    metrics = {
                        **base_metrics,
                        "service": "microbatch",
                        "batch_requests": len(selected),
                        "batch_questions": len(joined.items),
                        "queue_ms": (start - item.enqueued) * 1000,
                    }
                    # Slices are copied so a caller owns exactly its request's
                    # arrays and does not retain another caller's results.
                    item.future.set_result(
                        (
                            logits[offset : offset + count].copy(),
                            actions[offset : offset + count].copy(),
                            metrics,
                        )
                    )
                    offset += count
            except BaseException as error:  # noqa: BLE001 - accepted callers must receive worker failures.
                for item in selected:
                    item.future.set_exception(error)

    def warm(self, requests):
        # Use before concurrent traffic. The adapter's RLock also serializes this
        # operation against worker inference if a caller nevertheless races it.
        with self._request():
            for request in requests:
                prepared = self.adapter.prepare(**request)
                n = len(prepared.items)
                if not n:
                    continue
                cap = min(self.max_requests, max(1, self.max_questions // n))
                unmasked = self.base._graph_key(prepared)[-1]
                counts = [
                    i for i in range(1, cap + 1) if not unmasked or i & (i - 1) == 0
                ]
                for count in counts:
                    self.adapter.run_prepared(merge_prepared([prepared] * count))

    def close(self):
        with self.close_lock:
            with self.cv:
                if self.closed:
                    return
                self.closed = True
                self.cv.notify_all()
                self.cv.wait_for(lambda: self.active == 0)
                self.cv.notify_all()
            self.worker.join()
            self.adapter.close()
