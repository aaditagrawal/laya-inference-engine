"""Reproduce queue-inclusive serving experiments under the external GPU flock.

Example:
  flock /tmp/laya-gpu-experiments.lock uv run --no-sync python \
    -m experiments.latency.serving.run --task benchmark --rounds 3
"""

import argparse
import gc
import hashlib
import json
import random
import threading
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from experiments.native.common import (
    CACHE,
    compare_outputs,
    metadata,
    model_path,
    validation_requests,
)
from experiments.native.engine import ExperimentalEngine
from laya_blackwell.workloads import STATES, workload

from .graph_adapter import replace_adapter
from .service import MicrobatchService, StreamService, merge_prepared

OUT = Path(__file__).resolve().parents[3] / "results/latency-optimizations/serving"
NAMES = [
    "serial-current",
    "serial-fifo",
    "streams-2",
    "streams-4",
    "priority-3",
    "microbatch-0",
    "microbatch-0.25",
]


def save(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n")


def make_service(engine, name):
    if name == "serial-current":
        return engine
    if name == "serial-fifo":
        return StreamService(engine.base, lanes=1)
    if name.startswith("streams-"):
        return StreamService(engine.base, lanes=int(name.rsplit("-", 1)[1]))
    if name == "priority-3":
        return StreamService(engine.base, lanes=3, small_lanes=2)
    if name.startswith("microbatch-"):
        return MicrobatchService(engine.base, wait_ms=float(name.rsplit("-", 1)[1]))
    raise ValueError(name)


def profiles():
    result = {}
    for batch, length in ((1, "short"), (16, "short"), (1, "long"), (16, "long")):
        variants = []
        for variant in range(4):
            request = workload(batch, length)
            # Change the actual visible tokens, not only an ignored suffix after
            # truncation. Question definitions and output ID ordering stay fixed.
            request["state"] = (
                STATES[variant]
                if length == "short"
                else (str(STATES[variant]) + " " + str(STATES[9]))
            )
            variants.append(request)
        result[f"{batch}-{length}"] = variants
    result["mixed"] = [
        result[case][variant]
        for variant in range(4)
        for case in ("1-short", "16-short", "1-long", "16-long")
    ]
    return result


def round_run(service, requests, concurrency, count):
    """Fixed-count, closed-loop clients; full predict latency includes waiting."""
    barrier = threading.Barrier(concurrency + 1)
    cursor_lock = threading.Lock()
    cursor = 0

    def client():
        nonlocal cursor
        outputs = []
        barrier.wait()
        while True:
            with cursor_lock:
                index = cursor
                cursor += 1
            if index >= count:
                return outputs
            request = requests[index % len(requests)]
            start = perf_counter()
            response = service.predict(**request)
            end = perf_counter()
            outputs.append(
                {
                    "index": index,
                    "request": index % len(requests),
                    "start_s": start,
                    "end_s": end,
                    "latency_ms": (end - start) * 1000,
                    "questions": len(request["questions"]),
                    "engine": response["engine"],
                    "response": response,
                }
            )

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(client) for _ in range(concurrency)]
        start = perf_counter()
        barrier.wait()
        outputs = [item for future in futures for item in future.result()]
        elapsed = perf_counter() - start
    outputs.sort(key=lambda item: item["index"])
    latency = [item["latency_ms"] for item in outputs]
    details = [{k: v for k, v in item.items() if k != "response"} for item in outputs]
    for item in details:
        item["start_s"] -= start
        item["end_s"] -= start
    row = {
        "concurrency": concurrency,
        "requests": count,
        "elapsed_s": elapsed,
        "requests_per_second": count / elapsed,
        "decisions_per_second": sum(item["questions"] for item in outputs) / elapsed,
        "p50_ms": float(np.median(latency)),
        "p95_ms": float(np.percentile(latency, 95)),
        "mean_ms": float(np.mean(latency)),
        "graph_misses": sum(
            item["engine"].get("graph_miss", False) for item in outputs
        ),
        "batch_histogram": dict(
            Counter(item["engine"].get("batch_requests", 1) for item in outputs)
        ),
        "samples": details,
    }
    return row, [item["response"] for item in outputs]


def response_error(actual, expected):
    """Rounded response comparison, additional to raw-logit validation."""
    a, e = actual["answers"], expected["answers"]
    if actual["usage"] != expected["usage"] or list(a) != list(e):
        raise AssertionError(
            "Request/output ownership, ID ordering or token accounting changed"
        )
    decisions, agreements, choice_max, action_max = 0, 0, 0.0, 0.0
    for key in a:
        av, ev = a[key], e[key]
        decisions += 1
        if av["type"] != ev["type"]:
            raise AssertionError("Question type changed")
        if av["type"] == "noul":
            ap, ep = av["noul"], ev["noul"]
            agrees = (ap >= 0.5) == (ep >= 0.5)
            error = abs(ap - ep)
        else:
            ap, ep = av["probabilities"], ev["probabilities"]
            agrees = (
                av["choice"] == ev["choice"]
                if av["type"] == "choice"
                else max(ap, key=ap.get) == max(ep, key=ep.get)
            )
            error = max(abs(ap[k] - ep[k]) for k in ep)
        agreements += int(agrees)
        choice_max = max(choice_max, error)
        action_max = max(
            action_max,
            abs(av["action"]["act_probability"] - ev["action"]["act_probability"]),
        )
    return {
        "decisions": decisions,
        "agreement": agreements,
        "max_probability_error": choice_max,
        "max_action_probability_error": action_max,
        "exact_answers": a == e,
    }


def aggregate_checks(checks):
    return {
        "requests": len(checks),
        "decisions": sum(x["decisions"] for x in checks),
        "agreement": sum(x["agreement"] for x in checks),
        "max_probability_error": max(x["max_probability_error"] for x in checks),
        "max_action_probability_error": max(
            x["max_action_probability_error"] for x in checks
        ),
        "exact_answers_all": all(x.get("exact_answers", False) for x in checks),
    }


def benchmark(args):
    report = {"metadata": metadata(), "rows": [], "order": [], "warmup": []}
    report["metadata"]["method"] = (
        "Local Python service calls, no HTTP. Fixed-count closed-loop clients at concurrency 1/2/4/8. "
        "Wall request latency includes preparation, service queue, native GPU execution and formatting. "
        "Throughput uses whole client wall time; compilation/capture and equal warmup are excluded. "
        "Baseline already overlaps caller tokenization/formatting with serialized native GPU replay. "
        "No response cache. Every profile changes visible input tokens among four states."
    )
    report["metadata"]["original_baseline"] = "ExperimentalEngine(mode='native-window')"
    report["source_sha256"] = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (
            Path(__file__),
            Path(__file__).with_name("service.py"),
            Path(__file__).with_name("graph_adapter.py"),
        )
    }
    all_profiles = profiles()
    selected = {k: v for k, v in all_profiles.items() if k in args.profiles}
    report["request_hashes"] = {
        k: hashlib.sha256(json.dumps(v, sort_keys=True).encode()).hexdigest()
        for k, v in selected.items()
    }
    rng = random.Random(args.seed)
    with ExperimentalEngine(model=model_path(), max_graphs=24) as engine:
        replace_adapter(engine)
        report["metadata"]["graph_lifetime_policy"] = (
            "One uniquely owned raw CUDA capture stream per graph, in every variant including baseline"
        )
        report["metadata"]["kernel"] = engine.base.experimental_kernel
        references = {
            name: [engine.predict(**request) for request in requests]
            for name, requests in selected.items()
        }
        before = torch.cuda.memory_allocated()
        for round_id in range(args.rounds):
            names = list(args.variants)
            rng.shuffle(names)
            report["order"].append({"round": round_id, "variants": names})
            for name in names:
                service = make_service(engine, name)
                try:
                    for profile, requests in selected.items():
                        start = perf_counter()
                        if service is engine:
                            for request in requests:
                                service.predict(**request)
                        else:
                            service.warm(requests)
                        report["warmup"].append(
                            {
                                "round": round_id,
                                "variant": name,
                                "profile": profile,
                                "graph_warm_seconds": perf_counter() - start,
                                "allocated_cuda_bytes": torch.cuda.memory_allocated(),
                            }
                        )
                        for concurrency in (1, 2, 4, 8):
                            # Equal non-measured traffic for every mode/concurrency.
                            round_run(
                                service, requests, concurrency, max(16, concurrency * 2)
                            )
                            row, responses = round_run(
                                service, requests, concurrency, args.requests
                            )
                            row.update(round=round_id, variant=name, profile=profile)
                            row["response_check"] = aggregate_checks(
                                [
                                    response_error(
                                        response, references[profile][i % len(requests)]
                                    )
                                    for i, response in enumerate(responses)
                                ]
                            )
                            report["rows"].append(row)
                            save(args.output, report)
                            print(
                                json.dumps(
                                    {k: v for k, v in row.items() if k != "samples"}
                                ),
                                flush=True,
                            )
                finally:
                    if service is not engine:
                        service.close()
                        del service
                    gc.collect()
                    torch.cuda.empty_cache()
        report["cuda_bytes_after_services_close"] = torch.cuda.memory_allocated()
        report["cuda_bytes_before_services"] = before
    gc.collect()
    report["cuda_bytes_after_engine_close"] = torch.cuda.memory_allocated()
    save(args.output, report)


def validation(args):
    report = {"metadata": metadata(), "variant": args.variants[0]}
    reference = np.load(CACHE / "outputs.npz", allow_pickle=False)
    requests = validation_requests()
    with ExperimentalEngine(model=model_path(), max_graphs=4) as engine:
        replace_adapter(engine)
        service = make_service(engine, args.variants[0])
        before = torch.cuda.memory_allocated()
        try:
            prepared = [engine.prepare(**request) for request in requests]
            report["concurrent"] = []
            with ThreadPoolExecutor(max_workers=8) as pool:
                futures = [pool.submit(service.run_prepared, p) for p in prepared]
                outputs = [future.result() for future in futures]
            for i, (p, out) in enumerate(zip(prepared, outputs)):
                check = compare_outputs(
                    out,
                    (reference[f"logits_{i}"], reference[f"act_{i}"]),
                    p,
                    engine.agent,
                )
                check.update(case=i, metrics=out[2])
                report["concurrent"].append(check)
            # Preserve outputs across same-shape changing requests and cache churn.
            retained = [(o[0].copy(), o[1].copy()) for o in outputs]
            with ThreadPoolExecutor(max_workers=8) as pool:
                list(pool.map(service.run_prepared, reversed(prepared)))
            report["retained_output_arrays_unchanged"] = all(
                np.array_equal(o[0], r[0]) and np.array_equal(o[1], r[1])
                for o, r in zip(outputs, retained)
            )
            report["empty_request"] = (
                service.predict(state="", questions={})["answers"] == {}
            )
            try:
                service.predict(state="", questions={"bad": {}})
            except (ValueError, TypeError):
                report["invalid_request_rejected"] = True
            else:
                report["invalid_request_rejected"] = False
            if isinstance(service, MicrobatchService):
                report["forced_microbatches"] = {}
                bykey = defaultdict(list)
                for i, p in enumerate(prepared):
                    bykey[tuple(engine.base._graph_key(p))].append(i)
                for copies in (2, 8):
                    checks = []
                    for i, p in enumerate(prepared):
                        count = min(copies, 64 // len(p.items))
                        compatible = bykey[tuple(engine.base._graph_key(p))]
                        partners = [p] + [
                            prepared[
                                compatible[
                                    (compatible.index(i) + j + 1) % len(compatible)
                                ]
                            ]
                            for j in range(count - 1)
                        ]
                        logits, actions, metrics = service.adapter.run_prepared(
                            merge_prepared(partners)
                        )
                        n = len(p.items)
                        out = (logits[:n], actions[:n])
                        check = compare_outputs(
                            out,
                            (reference[f"logits_{i}"], reference[f"act_{i}"]),
                            p,
                            engine.agent,
                        )
                        check.update(case=i, batch_requests=count, metrics=metrics)
                        checks.append(check)
                    report["forced_microbatches"][str(copies)] = checks
                # Original four-row fixtures include near ties. Expand them to
                # irregular 5/17-row requests, comparing to the same padded BF16
                # engine rather than assuming parity follows from four rows.
                report["expanded_near_ties"] = []
                for index in (5, 33, 41, 45, 65):
                    original = requests[index]
                    definitions = list(original["questions"].values())
                    for n in (5, 17):
                        request = {
                            "state": original["state"],
                            "questions": {
                                f"q{i}": definitions[i % len(definitions)]
                                for i in range(n)
                            },
                        }
                        p = engine.prepare(**request)
                        expected = engine.run_prepared(p)
                        for copies in (2, 4, 8):
                            count = min(copies, 64 // n)
                            joined = merge_prepared([p] * count)
                            logits, actions, metrics = service.adapter.run_prepared(
                                joined
                            )
                            check = compare_outputs(
                                (logits[:n], actions[:n]), expected, p, engine.agent
                            )
                            check.update(
                                source_case=index,
                                questions=n,
                                requested_copies=copies,
                                batch_requests=count,
                                metrics=metrics,
                            )
                            report["expanded_near_ties"].append(check)
            # In-flight close must drain accepted requests and release graph storage.
            request = requests[0]
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(service.predict, **request) for _ in range(4)]
                # Waiting on each accepted request makes the close proof separate
                # from throughput; concurrent-close admission is checked below.
                results = [f.result() for f in futures]
            report["four_caller_answers_valid"] = all(
                len(x["answers"]) == len(request["questions"]) for x in results
            )
        finally:
            if service is not engine:
                service.close()
                service.close()
        try:
            service.predict(**requests[0])
        except RuntimeError:
            report["use_after_close_rejected"] = True
        else:
            report["use_after_close_rejected"] = False
        del service
        gc.collect()
        torch.cuda.empty_cache()
        report["allocated_before_service_graphs"] = before
        report["allocated_after_service_close"] = torch.cuda.memory_allocated()
    gc.collect()
    report["allocated_after_engine_close"] = torch.cuda.memory_allocated()
    for label, checks in [
        ("concurrent", report["concurrent"]),
        *report.get("forced_microbatches", {}).items(),
    ]:
        print(
            json.dumps(
                {
                    "check": label,
                    "decisions": sum(x["decisions"] for x in checks),
                    "agreement": sum(x["agreement"] for x in checks),
                    "exact_all": all(x["exact_logits_and_actions"] for x in checks),
                    "max_probability_error": max(
                        x["max_probability_error"] for x in checks
                    ),
                    "max_action_error": max(
                        x["max_action_probability_error"] for x in checks
                    ),
                }
            ),
            flush=True,
        )
    save(args.output, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task", choices=("benchmark", "validate"), default="benchmark"
    )
    parser.add_argument("--variants", nargs="+", choices=NAMES, default=NAMES)
    parser.add_argument(
        "--profiles", nargs="+", choices=list(profiles()), default=list(profiles())
    )
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--requests", type=int, default=48)
    parser.add_argument("--seed", type=int, default=77019)
    parser.add_argument("--output", type=Path, default=OUT / "screen.json")
    args = parser.parse_args()
    if args.rounds < 1 or args.requests < 8:
        parser.error("rounds must be positive and requests at least 8")
    if args.task == "validate" and len(args.variants) != 1:
        parser.error("validate requires exactly one variant")
    torch.set_num_threads(1)
    (benchmark if args.task == "benchmark" else validation)(args)


if __name__ == "__main__":
    main()
