"""Check packaged fast mode against the retained research implementation.

Run from the repository under /tmp/laya-gpu-experiments.lock. This is a
publication check, not a runtime dependency. Both implementations stay resident
during the paired comparison; no answers or prepared requests are cached.
"""

import argparse
import gc
import hashlib
import json
import random
import statistics
import time
import weakref
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch

from experiments.frontier.engine import FrontierEngine
from experiments.frontier.holdout import requests
from experiments.native import common
from laya_blackwell import BlackwellEngine, FastEngine

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sources():
    paths = list((ROOT / "src/laya_blackwell").rglob("*.py"))
    paths += list((ROOT / "experiments").rglob("*.py"))
    paths += [ROOT / "src/laya_blackwell/fast/config.json", Path(__file__)]
    return {str(path.relative_to(ROOT)): digest(path) for path in sorted(paths)}


def bits_equal(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and a.tobytes() == b.tobytes()


def public(engine, fixture):
    response = engine.predict(**fixture)
    response.pop("engine")
    return response


def validate(reference, candidate):
    fixtures = common.validation_requests() + requests()
    fixtures += [
        common.workload(n, length)
        for n, length in ((1, "short"), (1, "long"), (16, "short"))
    ]
    fixtures += [
        {"state": "", "questions": {}},
        {"state": {"message": "hello"}, "questions": {}},
    ]
    rows, owned = [], []
    for index, fixture in enumerate(fixtures):
        expected = reference.prepare(**fixture)
        prepared = candidate.prepare(**fixture)
        a, b = reference.run_prepared(expected), candidate.run_prepared(prepared)
        checks = {
            "prepared": expected == prepared,
            "choice_bits": bits_equal(a[0], b[0]),
            "action_bits": bits_equal(a[1], b[1]),
            "public": public(reference, fixture) == public(candidate, fixture),
        }
        assert all(checks.values()), (index, checks)
        owned.extend([(b[0], b[0].copy()), (b[1], b[1].copy())])
        request_sha256 = hashlib.sha256(
            json.dumps(fixture, sort_keys=True).encode()
        ).hexdigest()
        rows.append(
            {
                "request": index,
                "request_sha256": request_sha256,
                "decisions": len(prepared.ids),
                **checks,
            }
        )
        if index % 32 == 0:
            print(f"Validated {index + 1}/{len(fixtures)} requests", flush=True)
    assert all(bits_equal(a, b) for a, b in owned)
    concurrent = requests()[:8] + fixtures[-5:]
    expected = [public(reference, fixture) for fixture in concurrent]
    tasks = [i % len(concurrent) for i in range(52)]

    def call(index):
        return index, public(candidate, concurrent[index])

    with ThreadPoolExecutor(max_workers=4) as pool:
        actual = list(pool.map(call, tasks))
    assert all(answer == expected[index] for index, answer in actual)
    return {
        "requests": len(rows),
        "decisions": sum(row["decisions"] for row in rows),
        "all_exact": True,
        "owned_outputs": True,
        "concurrent": {"workers": 4, "calls": len(tasks), "all_exact": True},
        "wheel_smoke": {
            "request": fixtures[-5],
            "expected_response": public(reference, fixtures[-5]),
        },
        "rows": rows,
    }


def lifecycle(candidate, reference):
    adapter = candidate.adapter
    for slot in adapter.graphs.values():
        slot.close()
    adapter.graphs.clear()
    adapter.max_graphs = 2
    fixtures = [
        common.workload(n, length)
        for n, length in ((1, "short"), (1, "long"), (16, "short"))
    ]
    owners = []
    for fixture in fixtures * 2:
        assert public(candidate, fixture) == public(reference, fixture)
        prepared = candidate.prepare(**fixture)
        slot = adapter.graphs[candidate.base._graph_key(prepared)]
        owners.append(weakref.ref(slot.capture_owner))
        del slot
    gc.collect()
    assert sum(ref() is None for ref in owners) == 4
    candidate.close()
    candidate.close()
    gc.collect()
    assert all(ref() is None for ref in owners)
    try:
        candidate.predict(**fixtures[0])
    except RuntimeError as error:
        assert "closed" in str(error).lower()
    else:
        raise AssertionError("Closed FastEngine accepted a request")
    return {
        "evictions": 4,
        "all_owners_released": True,
        "idempotent_close": True,
        "closed_request_rejected": True,
    }


def timing_summary(rows, names):
    result = {}
    for case in dict.fromkeys(row["case"] for row in rows):
        selected = [row for row in rows if row["case"] == case]
        result[case] = {
            name: common.stats(
                [
                    value
                    for row in selected
                    if row["variant"] == name
                    for value in row["samples_ms"]
                ]
            )
            for name in names
        }
        for value in result[case].values():
            value["samples"] = len(value.pop("samples_ms"))
        per_round = {
            name: {
                row["round"]: row["p50_ms"]
                for row in selected
                if row["variant"] == name
            }
            for name in names
        }
        savings = [
            per_round["retained"][i] - per_round["fast"][i]
            for i in sorted(per_round["retained"])
        ]
        result[case]["packaging_paired_savings_ms"] = savings
        result[case]["packaging_median_paired_saving_ms"] = statistics.median(savings)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=ROOT / "results/fast-package.json"
    )
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--repeats", type=int, default=100)
    args = parser.parse_args()
    torch.set_num_threads(4)
    constructor = json.loads((ROOT / "results/frontier/summary.json").read_text())[
        "recommended_constructor"
    ]
    before = sources()
    report = {
        "metadata": common.metadata(),
        "retained_constructor": constructor,
        "source_before": before,
    }
    with (
        FrontierEngine(**constructor, max_graphs=16) as reference,
        FastEngine(max_graphs=16) as candidate,
        BlackwellEngine(max_graphs=16) as balanced,
    ):
        report["native_build"] = {
            key: candidate.build[key]
            for key in (
                "fingerprint",
                "library_sha256",
                "fast_math",
                "flash_unfuse_fma",
            )
        }
        libraries = dict(candidate.build["artifacts"])
        report["parity"] = validate(reference, candidate)
        engines = {"retained": reference, "fast": candidate, "balanced": balanced}
        report["fixed"] = common.benchmark_interleaved(
            engines,
            cases=[(1, "short"), (1, "long"), (16, "short")],
            rounds=args.rounds,
            repeats=args.repeats,
            warmups=5,
        )
        changing = requests()
        for engine in engines.values():
            for fixture in changing:
                engine.predict(**fixture)
        rng = random.Random(402927)
        rows, orders = [], []
        for round_id in range(args.rounds):
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
                    {
                        "variant": name,
                        "case": "changing",
                        "round": round_id,
                        **common.stats(samples),
                    }
                )
        report["changing"] = {"rows": rows, "orders": orders}
        report["summary"] = timing_summary(report["fixed"]["rows"] + rows, engines)
        report["lifecycle"] = lifecycle(candidate, reference)
        report["native_libraries_after"] = {
            name: digest(Path(path)) for name, path in libraries.items()
        }
        assert (
            report["native_libraries_after"] == report["native_build"]["library_sha256"]
        )
    report["source_after"] = sources()
    assert before == report["source_after"], "Source changed during package validation"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {"summary": report["summary"], "parity": report["parity"]["all_exact"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
