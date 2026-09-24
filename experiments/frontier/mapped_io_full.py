"""Full parity, lifecycle, concurrency and paired mapped-output comparison.

Run under exclusive /tmp/laya-gpu-experiments.lock. Graphs use DMA H2D and one
mapped output-store kernel. Model arithmetic and host preparation are unchanged.
"""

import gc
import hashlib
import json
import random
import statistics
import sys
import time
import weakref
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

from experiments.latency.serving.graph_adapter import StableHostAdapter
from experiments.native import common

from .engine import FrontierEngine
from .holdout import requests
from .host_prepare import install as install_prepare
from .host_runtime import install as install_runtime
from .mapped_io_adapter import install
from .native_format import install as install_format

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sources():
    paths = set(Path("experiments/frontier").glob("mapped_io*.py"))
    for module in list(sys.modules.values()):
        name = getattr(module, "__file__", None)
        if name is not None:
            p = Path(name).resolve()
            if p.suffix == ".py" and any(
                p.is_relative_to(ROOT / folder) for folder in ("src", "experiments")
            ):
                paths.add(p)
    return {str(p): digest(p) for p in paths}


def libraries():
    result = {}
    for line in Path("/proc/self/maps").read_text().splitlines():
        parts = line.split(maxsplit=5)
        if len(parts) == 6:
            path = Path(parts[-1])
            if path.is_relative_to(ROOT / ".research") and ".so" in path.name:
                result[str(path)] = digest(path)
    return result


def public(engine, fixture):
    answer = engine.predict(**fixture)
    metrics = answer.pop("engine")
    if set(metrics) != {"graph_miss", "graph_build_ms", "shape", "backend", "total_ms"}:
        raise RuntimeError("Changed engine metadata contract")
    return answer


def replace_closed_adapter(engine):
    old = engine.adapter
    if not old.closed or old.graphs:
        raise RuntimeError("Expected completely closed adapter")
    engine.adapter = StableHostAdapter(engine.base, mode=old.mode, max_graphs=16)
    install_prepare(engine, "batch")
    install_runtime(engine)
    install_format(engine)
    install(engine)


def lifecycle(engine, baseline, fixtures):
    engine.adapter.max_graphs = 2
    owners, buffers, copies, handles = [], [], [], []
    for fixture in fixtures + fixtures:
        prepared = engine.prepare(**fixture)
        logits, actions, _ = engine.run_prepared(prepared)
        copies.extend([(logits, logits.copy()), (actions, actions.copy())])
        key = engine.base._graph_key(prepared)
        slot = engine.adapter.graphs[key]
        owners.append(weakref.ref(slot.mapped_output.owner))
        buffers.append(weakref.ref(slot.host_out))
        handles.append(int(slot.capture_owner.raw))
        assert public(engine, fixture) == public(baseline, fixture)
        del slot
    # Three distinct geometries with a two-slot cache must evict four slots.
    gc.collect()
    expired = sum(ref() is None for ref in owners)
    if expired != 4:
        raise RuntimeError(f"Expected four evicted mapped owners, got {expired}")
    live_streams = [
        int(slot.capture_owner.raw) for slot in engine.adapter.graphs.values()
    ]
    assert len(live_streams) == len(set(live_streams))
    engine.adapter.close()
    engine.adapter.close()
    gc.collect()
    assert all(ref() is None for ref in owners + buffers)
    assert all(np.array_equal(a.view(np.uint8), b.view(np.uint8)) for a, b in copies)
    try:
        engine.run_prepared(prepared)
    except RuntimeError as error:
        assert str(error) == "Engine is closed"
    else:
        raise RuntimeError("Closed mapped adapter accepted a request")
    replace_closed_adapter(engine)
    assert public(engine, fixtures[0]) == public(baseline, fixtures[0])
    return {
        "evicted_owners_expired": expired,
        "all_six_owners_expired_after_close": True,
        "owned_arrays_unchanged": True,
        "unique_live_capture_streams": True,
        "idempotent_close": True,
        "closed_request_rejected": True,
        "recreated_adapter_exact": True,
        "capture_handles": handles,
    }


def main():
    torch.set_num_threads(4)
    probe = json.loads(Path("results/frontier/mapped-io-poison.json").read_text())
    life = json.loads(
        Path("results/frontier/mapped-io-lifetime-poison.json").read_text()
    )
    if (
        not probe["all_exact"]
        or not life["all_exact"]
        or not probe["sources_unchanged"]
        or not life["sources_unchanged"]
        or not life.get("independent_cpu_expected_and_poison")
        or not all(
            row.get("independent_cpu_expected_and_poison")
            for row in probe["visibility"]
        )
    ):
        raise RuntimeError("Mapped I/O visibility/lifetime gate failed")
    for gate in (probe, life):
        for path, expected_hash in gate["source_after"].items():
            if digest(path) != expected_hash:
                raise RuntimeError(f"Mapped gate source changed: {path}")
    if probe["summary"]["1-short"]["mapped-output"]["faster_rounds"] < 9:
        raise RuntimeError("Mapped output did not show a consistent isolated wall gain")
    constructor = json.loads(Path("results/frontier/summary.json").read_text())[
        "recommended_constructor"
    ]
    before = sources()
    report = {
        "metadata": common.metadata(),
        "constructor": constructor,
        "source_before": before,
        "isolated_probe_sha256": digest("results/frontier/mapped-io-poison.json"),
        "lifetime_probe_sha256": digest(
            "results/frontier/mapped-io-lifetime-poison.json"
        ),
        "mechanism": "Unchanged packed DMA H2D; one bit-preserving mapped-host output-store kernel replaces two D2H copies.",
    }
    fixtures = common.validation_requests() + requests()
    extras = [
        common.workload(n, length)
        for n, length in ((1, "short"), (1, "long"), (16, "short"))
    ]
    with (
        FrontierEngine(**constructor, max_graphs=16) as baseline,
        FrontierEngine(**constructor, max_graphs=16) as candidate,
    ):
        install(candidate)
        engines = {"retained": baseline, "mapped-output": candidate}
        report["lifecycle"] = lifecycle(candidate, baseline, extras)
        print("Full-model eviction, close and recreation gate passed", flush=True)
        parity = []
        retained_arrays = []
        for index, fixture in enumerate(fixtures + extras):
            values = {}
            for name, engine in engines.items():
                prepared = engine.prepare(**fixture)
                logits, actions, _ = engine.run_prepared(prepared)
                values[name] = (prepared, logits, actions, public(engine, fixture))
                if name == "mapped-output":
                    retained_arrays += [
                        (logits, logits.copy()),
                        (actions, actions.copy()),
                    ]
            expected, actual = values.values()
            checks = {
                "prepared": expected[0] == actual[0],
                "choice_bits": np.array_equal(
                    expected[1].view(np.uint8), actual[1].view(np.uint8)
                ),
                "action_bits": np.array_equal(
                    expected[2].view(np.uint8), actual[2].view(np.uint8)
                ),
                "public": json.dumps(expected[3]) == json.dumps(actual[3]),
            }
            if not all(checks.values()):
                raise RuntimeError(f"Mapped full parity failed at {index}: {checks}")
            parity.append(
                {"request": index, "decisions": len(expected[0].ids), **checks}
            )
            if index % 32 == 0:
                print(f"Validated request {index}", flush=True)
        assert all(
            np.array_equal(a.view(np.uint8), b.view(np.uint8))
            for a, b in retained_arrays
        )
        report["parity"] = {
            "all_exact": True,
            "requests": len(parity),
            "decisions": sum(row["decisions"] for row in parity),
            "returned_arrays_owned": True,
            "rows": parity,
        }
        concurrent = requests()[:8] + extras
        expected = [public(baseline, fixture) for fixture in concurrent]
        tasks = [(i % len(concurrent)) for i in range(44)]

        def call(index):
            return index, public(candidate, concurrent[index])

        with ThreadPoolExecutor(max_workers=4) as pool:
            actual = list(pool.map(call, tasks))
        assert all(answer == expected[index] for index, answer in actual)
        report["concurrent"] = {"workers": 4, "calls": len(tasks), "all_exact": True}
        print(
            "197-request parity, owned outputs and concurrent calls passed", flush=True
        )
        for path, current_hash in sources().items():
            if path in before and before[path] != current_hash:
                raise RuntimeError(f"Source changed during validation: {path}")
            before.setdefault(path, current_hash)
        report["native_libraries_before"] = libraries()
        report["fixed"] = common.benchmark_interleaved(
            engines,
            cases=[(1, "short"), (1, "long"), (16, "short")],
            rounds=11,
            repeats=150,
        )
        changing = requests()
        for engine in engines.values():
            for fixture in changing:
                engine.predict(**fixture)
        rows, orders = [], []
        rng = random.Random(157234)
        for round_id in range(11):
            names, indices = list(engines), list(range(len(changing)))
            rng.shuffle(names)
            rng.shuffle(indices)
            orders.append({"round": round_id, "variants": names, "indices": indices})
            for name in names:
                samples = []
                for index in indices:
                    start = time.perf_counter_ns()
                    engines[name].predict(**changing[index])
                    samples.append((time.perf_counter_ns() - start) / 1e6)
                rows.append(
                    {"variant": name, "round": round_id, **common.stats(samples)}
                )
        report["changing"] = {"rows": rows, "orders": orders}
        summary = {}
        for case in ("1-short", "1-long", "16-short", "changing"):
            selected = (
                rows
                if case == "changing"
                else [r for r in report["fixed"]["rows"] if r["case"] == case]
            )
            timing = {
                name: {
                    r["round"]: r["p50_ms"] for r in selected if r["variant"] == name
                }
                for name in engines
            }
            savings = [
                timing["retained"][i] - timing["mapped-output"][i] for i in range(11)
            ]
            summary[case] = {
                "p50_ms": {
                    name: statistics.median(
                        [
                            x
                            for r in selected
                            if r["variant"] == name
                            for x in r["samples_ms"]
                        ]
                    )
                    for name in engines
                },
                "paired_savings_ms": savings,
                "median_paired_saving_ms": statistics.median(savings),
                "faster_rounds": sum(x > 0 for x in savings),
            }
        report["summary"] = summary
        report["native_libraries_after"] = libraries()
        report["libraries_unchanged"] = (
            report["native_libraries_before"] == report["native_libraries_after"]
        )
    after = {path: digest(path) for path in before}
    report["source_after"] = after
    report["sources_unchanged"] = before == after
    Path("results/frontier/mapped-io-full.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                "parity": {k: v for k, v in report["parity"].items() if k != "rows"},
                "summary": summary,
                "sources_unchanged": report["sources_unchanged"],
                "libraries_unchanged": report["libraries_unchanged"],
            },
            indent=2,
        ),
        flush=True,
    )
    if not report["sources_unchanged"] or not report["libraries_unchanged"]:
        raise RuntimeError("Sources or native libraries changed during comparison")


if __name__ == "__main__":
    main()
