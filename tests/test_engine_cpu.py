"""Engine shape and reusable-buffer boundaries without CUDA operations."""

from collections import OrderedDict
from contextlib import nullcontext
import threading
from types import SimpleNamespace

import pytest
import torch

from laya_blackwell.engine import BlackwellEngine, hardware_info
from laya_blackwell.protocol import PreparedRequest


def make_engine(*, max_questions=64, max_len=512):
    engine = BlackwellEngine.__new__(BlackwellEngine)
    engine.device = torch.device("cuda:0")
    engine.lock = threading.RLock()
    engine.max_questions = max_questions
    engine.sequence_buckets = (64, 128, 256, 384, 512, 768, 1024)
    engine.agent = SimpleNamespace(tok=SimpleNamespace(pad_token_id=0), cfg={"max_len": max_len}, model=object())
    engine.closed = False
    engine.graphs = OrderedDict()
    engine.model = object()
    return engine


def request_with_shapes(*shapes):
    items = [{"ids": [10] * length, "markers": list(range(options)), "qtype": index % 3}
             for index, (length, options) in enumerate(shapes)]
    return PreparedRequest([str(i) for i in range(len(items))], [{}] * len(items), items,
                           sum(len(item["ids"]) for item in items))


@pytest.mark.parametrize("shapes, limit, max_len, expected", [
    ([(1, 1)], 64, 512, (1, 64, 2)),
    ([(65, 3)], 64, 512, (1, 128, 4)),
    ([(65, 3), (80, 2), (20, 1)], 64, 512, (4, 128, 4)),
    ([(20, 2)] * 5, 6, 512, (6, 64, 2)),
    ([(80, 2)], 64, 96, (1, 96, 2)),
    ([(512, 65)], 64, 512, (1, 512, 128)),
])
def test_shape_buckets_cover_real_inputs(shapes, limit, max_len, expected):
    engine = make_engine(max_questions=limit, max_len=max_len)
    assert engine._shape(request_with_shapes(*shapes)) == expected


@pytest.mark.parametrize("shapes, unmasked", [
    ([(64, 2)], True),
    ([(63, 2)], False),
    ([(64, 2), (64, 2)], True),
    ([(64, 2), (63, 2)], False),
    ([(64, 2)] * 3, False),  # The fourth batch row is padding.
])
def test_graph_key_separates_global_attention_mask_modes(shapes, unmasked):
    engine = make_engine()
    prepared = request_with_shapes(*shapes)
    assert engine._graph_key(prepared) == (*engine._shape(prepared), unmasked)


def test_refill_clears_previous_request_and_pads_dummy_rows():
    engine = make_engine()
    buffers = {
        "input_ids": torch.full((4, 8), 99, dtype=torch.long),
        "attention_mask": torch.ones((4, 8), dtype=torch.long),
        "marker_pos": torch.full((4, 4), 7, dtype=torch.long),
        "marker_mask": torch.ones((4, 4), dtype=torch.bool),
        "qtype": torch.full((4,), 2, dtype=torch.long),
    }
    prepared = request_with_shapes((3, 2), (2, 1))
    prepared.items[0]["ids"] = [1, 3, 2]
    prepared.items[1]["ids"] = [1, 2]
    engine._fill(buffers, prepared)
    assert buffers["input_ids"].tolist() == [
        [1, 3, 2, 0, 0, 0, 0, 0], [1, 2, 0, 0, 0, 0, 0, 0],
        [0] * 8, [0] * 8,
    ]
    assert buffers["attention_mask"].tolist() == [
        [1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0, 0, 0],
    ]
    assert buffers["marker_pos"].tolist() == [[0, 1, 0, 0], [0] * 4, [0] * 4, [0] * 4]
    assert buffers["marker_mask"].tolist() == [
        [True, True, False, False], [True, False, False, False],
        [True, False, False, False], [True, False, False, False],
    ]
    assert buffers["qtype"].tolist() == [0, 1, 0, 0]


def test_closed_engine_rejects_empty_request_without_model_work(monkeypatch):
    engine = make_engine()
    engine.closed = True
    monkeypatch.setattr(torch.cuda, "device", lambda *_: nullcontext())
    with pytest.raises(RuntimeError, match="Engine is closed"):
        engine.run_prepared(request_with_shapes())


def test_empty_request_does_not_allocate_buffers_or_create_graphs(monkeypatch):
    engine = make_engine()
    monkeypatch.setattr(torch.cuda, "device", lambda *_: nullcontext())
    logits, act, metrics = engine.run_prepared(request_with_shapes())
    assert logits.shape == (0, 0)
    assert act.shape == (0, 2)
    assert metrics["graph_miss"] is False
    assert not engine.graphs


def test_close_releases_model_and_cached_graphs_after_synchronization(monkeypatch):
    engine = make_engine()
    engine.graphs[(1, 64, 2)] = object()
    events = []

    def synchronize(device):
        events.append(device)
        assert engine.model is not None
        assert engine.graphs

    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    engine.close()
    assert events == [torch.device("cuda:0")]
    assert engine.model is engine.agent.model is None
    assert engine.closed
    assert not engine.graphs


def test_hardware_rejects_cpu_runtime_without_device_query(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda *_: pytest.fail("unexpected device query"))
    with pytest.raises(RuntimeError, match="CUDA-enabled PyTorch"):
        hardware_info()


def test_hardware_rejects_non_blackwell(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda *_: SimpleNamespace(major=8, minor=9, name="Ada"))
    with pytest.raises(RuntimeError, match="Expected Blackwell; found Ada, SM89"):
        hardware_info()
