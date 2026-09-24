"""Full changing-input parity and paired request timing for safe PDL variants."""

import argparse
import contextlib
import hashlib
import json
import random
import statistics
import time
from pathlib import Path
from unittest.mock import patch

import torch
from cuda.bindings import driver
from cuda.bindings import runtime as cuda

from experiments.native import common

from .engine import FrontierEngine
from .holdout import requests
from .pdl_kernels import evidence, installed

ROOT = Path(__file__).resolve().parents[2]


def checked(result):
    status, *values = result
    if int(status) != 0:
        raise RuntimeError(f"CUDA graph inspection failed: {status}")
    return values[0] if len(values) == 1 else tuple(values)


class RetainedGraph(torch.cuda.CUDAGraph):
    def __init__(self, *args, **kwargs):
        kwargs["keep_graph"] = True
        super().__init__(*args, **kwargs)

    def capture_end(self):
        super().capture_end()
        self.instantiate()


def graph_edges(engine, *, expect_pdl=True):
    reports = []
    for key, slot in engine.adapter.graphs.items():
        graph = slot.graph.raw_cuda_graph()
        _, _, _, count = checked(cuda.cudaGraphGetEdges(graph))
        sources, targets, data, _ = checked(cuda.cudaGraphGetEdges(graph, count))
        names = {}

        def name(node, names=names):
            node_id = int(node)
            if node_id not in names:
                kind = checked(cuda.cudaGraphNodeGetType(node))
                if kind == cuda.cudaGraphNodeType.cudaGraphNodeTypeKernel:
                    params = checked(driver.cuGraphKernelNodeGetParams(node))
                    value = checked(driver.cuFuncGetName(params.func))
                    names[node_id] = (
                        value.decode() if isinstance(value, bytes) else str(value)
                    )
                else:
                    names[node_id] = str(kind)
            return names[node_id]

        rows = []
        for source, target, edge in zip(sources, targets, data):
            if int(edge.type):
                row = {
                    "source": name(source),
                    "target": name(target),
                    "type": int(edge.type),
                    "from_port": int(edge.from_port),
                    "to_port": int(edge.to_port),
                }
                if "pdl_" not in row["target"]:
                    raise RuntimeError(
                        f"Programmatic edge into a kernel without a proven wait: {row}"
                    )
                rows.append(row)
        all_names = sorted({name(node) for node in [*sources, *targets]})
        pdl_names = [value for value in all_names if value.startswith("pdl_")]
        reports.append(
            {
                "key": list(key),
                "edges": count,
                "programmatic_edges": rows,
                "kernel_names": all_names,
                "pdl_kernel_names": pdl_names,
            }
        )
        if not expect_pdl and (rows or pdl_names):
            raise RuntimeError("Retained baseline was contaminated by PDL patches")
    if expect_pdl and not any(r["programmatic_edges"] for r in reports):
        raise RuntimeError("Captured model graph has no programmatic dependency edges")
    return reports


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="+", default=["late", "early", "preload"])
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--output", type=Path, default=Path("results/frontier/pdl.json")
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    ctor = json.loads((ROOT / "results/frontier/summary.json").read_text())[
        "recommended_constructor"
    ]
    fixtures = requests()
    if args.smoke:
        fixtures = fixtures[:4]
    report = {
        "metadata": common.metadata(),
        "baseline_constructor": ctor,
        "fixtures": fixtures,
        "fixture_sha256": hashlib.sha256(
            json.dumps(fixtures, sort_keys=True).encode()
        ).hexdigest(),
        "method": "Unmodified retained constructor versus same-arithmetic PDL kernel clones. Full warm requests include tokenization through formatting. Programmatic graph edges are created by Triton launch_pdl, never rewritten. Every PDL consumer has a wait before dependent activation loads.",
        "variants": {},
        "source_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [
                Path(__file__),
                *(
                    ROOT / "experiments/frontier" / (name + ".py")
                    for name in [
                        "pdl_kernels",
                        "engine",
                        "matmul",
                        "tma",
                        "mlp_geglu",
                        "head_gemm",
                        "head_install",
                        "reduce_norm",
                        "holdout",
                    ]
                ),
                ROOT / "src/laya_blackwell/kernels.py",
                ROOT / "src/laya_blackwell/model.py",
                ROOT / "experiments/latency/serving/graph_adapter.py",
            ]
        },
    }
    with contextlib.ExitStack() as stack:
        stack.enter_context(patch.object(torch.cuda, "CUDAGraph", RetainedGraph))
        prior = stack.enter_context(FrontierEngine(**ctor, max_graphs=8))
        expected = []
        for fixture in fixtures:
            output = prior.run_prepared(prior.prepare(**fixture))
            expected.append(
                tuple(v.copy() if hasattr(v, "copy") else v for v in output)
            )
        fixed = common.workload(1, "short")
        prior.predict(**fixed)
        report["baseline_graphs"] = graph_edges(prior, expect_pdl=False)
        baseline_graph_ids = {
            key: id(slot.graph) for key, slot in prior.adapter.graphs.items()
        }
        for mode in args.modes:
            with installed(mode), FrontierEngine(**ctor, max_graphs=8) as candidate:
                item = {"parity": []}
                report["variants"][mode] = item
                for index, (fixture, reference) in enumerate(zip(fixtures, expected)):
                    prepared = candidate.prepare(**fixture)
                    actual = candidate.run_prepared(prepared)
                    check = common.compare_outputs(
                        actual, reference, prepared, candidate.agent
                    )
                    item["parity"].append(
                        {
                            "index": index,
                            "shape": candidate.base._shape(prepared),
                            **check,
                        }
                    )
                    if not check["exact_logits_and_actions"]:
                        raise RuntimeError(
                            f"PDL exact-output parity failed: {mode}:{index}: {check}"
                        )
                candidate.predict(**fixed)
                item["instructions"] = evidence(
                    mode, ROOT / ".research/frontier-pdl" / mode
                )
                item["graphs"] = graph_edges(candidate)
                item["fixed"] = common.benchmark_interleaved(
                    {"retained": prior, "pdl": candidate},
                    cases=[(1, "short")],
                    rounds=3 if args.smoke else 9,
                    repeats=30 if args.smoke else 100,
                )
                item["changing_inputs"] = {"rows": [], "order": []}
                rng = random.Random(520926)
                rounds = 3 if args.smoke else 9
                for round_id in range(rounds):
                    indices = list(range(len(fixtures)))
                    rng.shuffle(indices)
                    engines = [("retained", prior), ("pdl", candidate)]
                    rng.shuffle(engines)
                    item["changing_inputs"]["order"].append(
                        {"indices": indices, "variants": [name for name, _ in engines]}
                    )
                    for name, current in engines:
                        samples = []
                        for index in indices:
                            start = time.perf_counter()
                            current.predict(**fixtures[index])
                            samples.append((time.perf_counter() - start) * 1000)
                        item["changing_inputs"]["rows"].append(
                            {
                                "variant": name,
                                "round": round_id,
                                **common.stats(samples),
                            }
                        )
                item["summary"] = {}
                if baseline_graph_ids != {
                    key: id(slot.graph) for key, slot in prior.adapter.graphs.items()
                }:
                    raise RuntimeError(
                        "Baseline graph cache changed under candidate patches"
                    )
                item["baseline_graphs_after"] = graph_edges(prior, expect_pdl=False)
                for case, rows in [
                    ("fixed", item["fixed"]["rows"]),
                    ("changing", item["changing_inputs"]["rows"]),
                ]:
                    med = {
                        name: statistics.median(
                            [
                                value
                                for row in rows
                                if row["variant"] == name
                                for value in row["samples_ms"]
                            ]
                        )
                        for name in ["retained", "pdl"]
                    }
                    by_round = {
                        name: {
                            r["round"]: r["p50_ms"]
                            for r in rows
                            if r["variant"] == name
                        }
                        for name in ["retained", "pdl"]
                    }
                    diffs = [
                        by_round["retained"][i] - by_round["pdl"][i]
                        for i in range(rounds)
                    ]
                    item["summary"][case] = {
                        "median_ms": med,
                        "paired_savings_ms": diffs,
                        "faster_rounds": sum(x > 0 for x in diffs),
                    }
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print(mode, json.dumps(item["summary"]), flush=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
