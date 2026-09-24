"""Request-ordered dense-weight streaming, separately from inference work."""

import collections
import fcntl
import hashlib
import json
import random
import statistics
import types
from pathlib import Path

import numpy as np
import torch
from cuda.bindings import driver
from cuda.bindings import runtime as cuda

from experiments.native import common

from .engine import FrontierEngine
from .weight_stream_kernel import BLOCK, COMPILED, Operation

ROOT = Path(__file__).resolve().parents[2]
DIRECTORY = ROOT / ".research/frontier-weight-stream"
OUTPUT = ROOT / "results/frontier/weight-stream.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes():
    return {
        str(path.relative_to(ROOT)): digest(path)
        for path in sorted(Path(__file__).parent.glob("weight_stream*.py"))
    }


def checked(result):
    status, *values = result
    if int(status):
        raise RuntimeError(f"CUDA graph audit failed: {status}")
    return values[0] if len(values) == 1 else tuple(values)


def cloned_method(method, replacements):
    old = method.__func__
    function = types.FunctionType(
        old.__code__,
        dict(old.__globals__, **replacements),
        old.__name__,
        old.__defaults__,
        old.__closure__,
    )
    function.__kwdefaults__ = old.__kwdefaults__
    return types.MethodType(function, method.__self__)


def trace_projections(engine):
    model = engine.base.model
    modules = dict(model.named_modules())
    names = {id(module): name for name, module in modules.items()}
    records = []

    def record(module, x, weight, bias, mechanism, suffix=""):
        records.append(
            {
                "name": names[id(module)] + suffix,
                "weight": weight,
                "input_shape": list(x.shape),
                "weight_shape": list(weight.shape),
                "weight_stride": list(weight.stride()),
                "dtype": str(weight.dtype),
                "bytes": weight.nbytes,
                "bias_bytes_excluded": bias.nbytes if bias is not None else 0,
                "capture_mechanism": mechanism,
                "original_address": weight.data_ptr(),
            }
        )

    def hook(module, args):
        record(module, args[0], module.weight, module.bias, "Linear pre-hook")

    handles = [
        module.register_forward_pre_hook(hook)
        for module in modules.values()
        if isinstance(module, torch.nn.Linear)
    ]
    original_forward, original_head = model.forward, model._head_layer
    mlp = original_forward.__func__.__globals__["frontier_mlp_geglu"]
    qkv = original_head.__func__.__globals__["head_qkv"]

    def record_mlp(x, module):
        # Retained unpacked GEGLU uses the actual module weight, without packing.
        record(module, x, module.weight, module.bias, "fused GEGLU helper")
        return mlp(x, module)

    def record_qkv(x, module):
        record(
            module,
            x,
            module.in_proj_weight,
            module.in_proj_bias,
            "packed head QKV helper",
            ".in_proj",
        )
        return qkv(x, module)

    model.forward = cloned_method(original_forward, {"frontier_mlp_geglu": record_mlp})
    model._head_layer = cloned_method(original_head, {"head_qkv": record_qkv})
    request = common.workload(1, "short")
    prepared = engine.prepare(**request)
    key = engine.base._graph_key(prepared)
    assert key[:2] == (1, 64)
    host = engine.base._allocate(key[:3], host=True)
    engine.base._fill(host, prepared)
    inputs = {name: tensor.to(engine.base.device) for name, tensor in host.items()}
    inputs["global_attention_unmasked"] = key[-1]
    try:
        engine.base._forward(inputs)
        torch.cuda.synchronize()
    finally:
        model.forward, model._head_layer = original_forward, original_head
        for handle in handles:
            handle.remove()
    captured = collections.Counter(row["name"] for row in records)
    # There are four dense encoder projections per layer, except precomputed QKV0.
    expected = {
        f"net.encoder.layers.{i}.{field}"
        for i in range(28)
        for field in ["attn.Wqkv", "attn.Wo", "mlp.Wi", "mlp.Wo"]
        if i != 0 or field != "attn.Wqkv"
    }
    for name, module in modules.items():
        if isinstance(module, torch.nn.Linear) and not name.startswith("net.encoder."):
            expected.add(name)
    expected.update(
        f"net.head.layers.{i}.self_attn.in_proj"
        for i in range(len(model.net.head.layers))
    )
    assert set(captured) == expected, (
        set(captured) - expected,
        expected - set(captured),
    )
    assert all(value == 1 for value in captured.values())
    return records, {
        "request": request,
        "shape_key": list(key),
        "input_tokens": prepared.input_tokens,
        "expected_projection_count": len(expected),
        "capture_complete_against_architecture": True,
        "bypasses_of_Linear_hooks": [
            "28 MLPWi calls in frontier_mlp_geglu, read from each module.weight",
            "Packed head QKV in head_qkv, read from self_attn.in_proj_weight",
        ],
        "excluded": [
            "First encoder QKV, precomputed by the token-table adapter before tracing",
            "Token/type embeddings and token-table gathers, not dense projections",
            "Normalization parameters and dense biases; only matrix weights are streamed",
        ],
    }


def reference_checksum(words):
    offsets = np.arange(0, len(words), BLOCK)
    return np.add.reduceat(words, offsets, dtype=np.uint64).astype(np.uint32)


def capture_graph(operations):
    for operation in operations:
        operation()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph(keep_graph=True)
    with torch.cuda.graph(graph):
        for operation in operations:
            operation()
    graph.instantiate()
    raw = graph.raw_cuda_graph()
    _, count = checked(cuda.cudaGraphGetNodes(raw))
    nodes, actual = checked(cuda.cudaGraphGetNodes(raw, count))
    assert count == actual == len(operations)
    names = []
    for node in nodes:
        kind = checked(cuda.cudaGraphNodeGetType(node))
        assert kind == cuda.cudaGraphNodeType.cudaGraphNodeTypeKernel
        params = checked(driver.cuGraphKernelNodeGetParams(node))
        name = checked(driver.cuFuncGetName(params.func))
        names.append(name.decode() if isinstance(name, bytes) else str(name))
    return graph, {"kernel_nodes": count, "kernel_names": sorted(set(names))}


def measure(graph, repeats=60):
    # Untimed complete-bank replays displace anything remaining from another case.
    for _ in range(3):
        graph.replay()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / repeats


def binary_evidence():
    rows = []
    for n, kernel in sorted(COMPILED.items()):
        paths = {}
        for kind in ["ptx", "cubin"]:
            path = DIRECTORY / f"weight_stream-{n}.{kind}"
            payload = kernel.asm[kind]
            path.write_bytes(
                payload if isinstance(payload, bytes) else payload.encode()
            )
            paths[str(path.relative_to(ROOT))] = digest(path)
        rows.append(
            {
                "words": n,
                "registers": kernel.n_regs,
                "spills": kernel.n_spills,
                "shared_bytes": kernel.metadata.shared,
                "hashes": paths,
            }
        )
    return rows


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    DIRECTORY.mkdir(parents=True, exist_ok=True)
    constructor_path = ROOT / "results/frontier/summary.json"
    retained = json.loads(constructor_path.read_text())["recommended_constructor"]
    constructor = dict(retained)
    constructor["policy"] = constructor["policy"].removesuffix("-short-compiled")
    before = source_hashes()
    report = {
        "metadata": common.metadata(),
        "scope": "Optimistic matrix-weight streaming diagnostic. Not inference, model-output parity, or a rigorous latency floor. No GEMM arithmetic, activation reads, repeated CTA weight reads or non-projection kernels are included.",
        "retained_constructor": retained,
        "trace_constructor": constructor,
        "summary_sha256_at_start": digest(constructor_path),
        "source_sha256_before": before,
        "checksum": "Each CTA sums 4096 consecutive uint32 weight words modulo 2^32 and writes one uint32. Every input word is read once per replay; CPU NumPy uint64 sums independently verify every block.",
        "rounds": 9,
        "replays_per_round": 60,
        "random_seed": 92931,
        "timing": "CUDA events around graph replays, three untimed whole-bank replays before each sample. Exclusive experiment flock covers capture, checks and all timing. Nine randomized interleaved rounds, with no simultaneous timing.",
    }
    with FrontierEngine(**constructor, max_graphs=1) as engine:
        records, trace = trace_projections(engine)
        report["trace"] = trace
        weights = [row.pop("weight") for row in records]
        total = sum(row["bytes"] for row in records)
        l2 = checked(
            driver.cuDeviceGetAttribute(
                driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_L2_CACHE_SIZE, 0
            )
        )
        assert total > 8 * l2
        assert len({weight.data_ptr() for weight in weights}) == len(weights)
        report.update(
            projections=records,
            projection_launch_count=len(weights),
            total_logical_bytes=total,
            dense_bias_bytes_excluded=sum(
                row["bias_bytes_excluded"] for row in records
            ),
            l2_bytes=l2,
            logical_bytes_over_l2=total / l2,
            distinct_source_allocations=len(weights),
        )
        bank = torch.cat([weight.view(torch.uint32).flatten() for weight in weights])
        references = []
        bank_host = bank.cpu().numpy()
        offset = 0
        slices = []
        for row, weight in zip(records, weights):
            original = weight.view(torch.uint32).flatten().cpu().numpy()
            end = offset + len(original)
            assert np.array_equal(original, bank_host[offset:end])
            row["bank_byte_offset"] = offset * 4
            row["weight_sha256"] = hashlib.sha256(original).hexdigest()
            ref = reference_checksum(original)
            row["checksum_sha256"] = hashlib.sha256(ref).hexdigest()
            references.append(ref)
            slices.append(bank[offset:end])
            offset = end
        assert offset == bank.numel() and bank.nbytes == total
        report["combined_bank_sha256"] = hashlib.sha256(bank_host).hexdigest()
        combined_reference = reference_checksum(bank_host)
        del bank_host
        cases = {
            "per_projection_original_allocations": [
                Operation(weight) for weight in weights
            ],
            "per_projection_contiguous_slices": [
                Operation(weight) for weight in slices
            ],
            "combined_contiguous_one_kernel": [Operation(bank)],
        }
        graphs = {}
        report["cases"] = {}

        def check_outputs():
            checks = {}
            for name, operations in cases.items():
                expected = [combined_reference] if len(operations) == 1 else references
                mismatches = sum(
                    np.count_nonzero(operation.output.cpu().numpy() != reference)
                    for operation, reference in zip(operations, expected)
                )
                assert mismatches == 0, (name, mismatches)
                checks[name] = {
                    "checked_output_words": sum(len(value) for value in expected),
                    "mismatches": int(mismatches),
                }
            return checks

        for name, operations in cases.items():
            graph, audit = capture_graph(operations)
            graphs[name] = graph
            report["cases"][name] = {
                "graph": audit,
                "checksum_output_bytes": sum(
                    operation.output.nbytes for operation in operations
                ),
                "samples_ms": [],
            }
        report["checks_before"] = check_outputs()
        report["binaries_before"] = binary_evidence()
        print(
            json.dumps(
                {"projections": len(weights), "logical_bytes": total, "l2_bytes": l2}
            ),
            flush=True,
        )
        rng = random.Random(report["random_seed"])
        report["round_order"] = []
        for round_index in range(9):
            order = list(cases)
            rng.shuffle(order)
            report["round_order"].append(order)
            for name in order:
                value = measure(graphs[name])
                report["cases"][name]["samples_ms"].append(value)
            print(
                json.dumps(
                    {
                        "round": round_index,
                        "ms": {
                            name: report["cases"][name]["samples_ms"][-1]
                            for name in cases
                        },
                    }
                ),
                flush=True,
            )
        report["checks_after"] = check_outputs()
        report["binaries_after"] = binary_evidence()
        assert report["binaries_before"] == report["binaries_after"]
        for case in report["cases"].values():
            case["median_ms"] = statistics.median(case["samples_ms"])
            case["logical_GB_per_second"] = total / case["median_ms"] / 1e6
        ordered = report["cases"]["per_projection_contiguous_slices"]["median_ms"]
        combined = report["cases"]["combined_contiguous_one_kernel"]["median_ms"]
        report["contiguous_per_projection_minus_one_kernel_ms"] = ordered - combined
        report["interpretation_limit"] = (
            "Launch granularity changes grid partitioning and GPU scheduling as well as graph-node count. The difference is not pure launch overhead and neither control is a latency floor for fused inference. Replays cycle a bank much larger than L2; no explicit L2 flush occurs between projections. Matrix bias reads are excluded and separately counted."
        )
        for graph in graphs.values():
            graph.reset()
    report["source_sha256_after"] = source_hashes()
    assert report["source_sha256_before"] == report["source_sha256_after"]
    OUTPUT.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"result": str(OUTPUT), "cases": report["cases"]}), flush=True)


if __name__ == "__main__":
    with open("/tmp/laya-gpu-experiments.lock", "a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        main()
