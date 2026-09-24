"""Mapped pinned visibility and graph-I/O screen on real packed request sizes.

Run under exclusive /tmp/laya-gpu-experiments.lock. No model weights are loaded.
Every mode includes the same tiny device kernel between input and output I/O,
so the dependency chain matches full inference instead of independent copies.
"""

import argparse
import hashlib
import json
import random
import statistics
import time
from pathlib import Path

import numpy as np
import torch
import triton as tr
import triton.language as tl
from huggingface_hub.constants import HF_HUB_CACHE
from laya.common import serialize_state
from transformers import AutoTokenizer

from experiments.latency.serving.graph_adapter import CaptureStream
from experiments.native import common
from experiments.native.host.adapter import laya_native_host, packed_allocate
from laya_blackwell.engine import REVISION, BlackwellEngine
from laya_blackwell.protocol import prepare_request

from .mapped_io import MappedPinned, collect, stage


@tr.jit
def _control(
    INPUT,
    LOGITS,
    ACTIONS,
    SIZE: tl.constexpr,
    NL: tl.constexpr,
    NA: tl.constexpr,
    BLOCK: tl.constexpr,
):
    i = tl.arange(0, BLOCK)
    left = tl.load(INPUT + i, i < NL, 0).to(tl.float32)
    right = tl.load(INPUT + SIZE - 1 - i, i < NA, 0).to(tl.float32)
    tl.store(LOGITS + i, left, i < NL)
    tl.store(ACTIONS + i, right, i < NA)


def control(device, logits, actions):
    _control[(1,)](
        device,
        logits,
        actions,
        device.numel(),
        logits.numel(),
        actions.numel(),
        tr.next_power_of_2(max(logits.numel(), actions.numel())),
        num_warps=4,
    )


def hashes():
    return {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(Path("experiments/frontier").glob("mapped_io*.py"))
    }


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("results/frontier/mapped-io-poison.json")
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    before = hashes()
    report = {
        "metadata": common.metadata(),
        "source_before": before,
        "method": "Same input/output sizes and dependency-control kernel; graph contains I/O; native Session.replay launches and synchronizes without packing.",
    }
    root = Path(HF_HUB_CACHE) / "models--convaiinnovations--laya/snapshots" / REVISION
    cfg = json.loads((root / "rl_agent_config.json").read_text())
    tok = AutoTokenizer.from_pretrained(root / "tokenizer", local_files_only=True)
    # Only the shape helper is used. Do not initialize an engine or load weights.
    shape_helper = object.__new__(BlackwellEngine)
    shape_helper.agent = type("Config", (), {"cfg": cfg})()
    shape_helper.max_questions = 64
    shape_helper.sequence_buckets = (64, 128, 256, 384, 512, 768, 1024)
    rows, checks, all_orders = [], [], []
    rng = random.Random(272938)
    for count, length in ((1, "short"), (1, "long"), (16, "short")):
        case = f"{count}-{length}"
        request = common.workload(count, length)
        prepared = prepare_request(
            tok, cfg, serialize_state(request["state"]), request["questions"]
        )
        shape = shape_helper._shape(prepared)
        b, s, k = shape
        host, _ = packed_allocate(shape, tok.pad_token_id, "cpu")
        device, _ = packed_allocate(shape, tok.pad_token_id, "cuda:0")
        host.copy_(torch.arange(host.numel()).remainder(251).to(torch.uint8))
        output = torch.empty(b * (k + 2), dtype=torch.float32, pin_memory=True)
        host_logits, host_actions = (
            output[: b * k].view(b, k),
            output[b * k :].view(b, 2),
        )
        logits, actions = (
            torch.empty((b, k), device="cuda"),
            torch.empty((b, 2), device="cuda"),
        )
        mapped_in, mapped_out = MappedPinned(host), MappedPinned(output)
        owners = {
            mode: CaptureStream(torch.device("cuda:0"), 0)
            for mode in ("dma", "mapped-output", "mapped-both")
        }
        stream = owners["mapped-both"].stream
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            stage(mapped_in, device)
            control(device, logits, actions)
            collect(logits, actions, mapped_out)
        stream.synchronize()
        visibility = {
            "case": case,
            "shape": list(shape),
            "input_bytes": host.numel(),
            "output_bytes": output.nbytes,
            "input_alias": mapped_in.alias.data_ptr() == mapped_in.device_pointer,
            "output_alias": mapped_out.alias.data_ptr() == mapped_out.device_pointer,
            "input_visible": torch.equal(device.cpu(), host),
            "logits_visible": torch.equal(host_logits, logits.cpu()),
            "actions_visible": torch.equal(host_actions, actions.cpu()),
        }
        if not all(
            visibility[key]
            for key in (
                "input_alias",
                "output_alias",
                "input_visible",
                "logits_visible",
                "actions_visible",
            )
        ):
            raise RuntimeError(f"Mapped visibility failed: {visibility}")
        graphs, sessions = {}, {}
        for mode in ("dma", "mapped-output", "mapped-both"):
            stream = owners[mode].stream
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                if mode == "mapped-both":
                    stage(mapped_in, device)
                else:
                    device.copy_(host, non_blocking=True)
                control(device, logits, actions)
                if mode == "dma":
                    host_logits.copy_(logits, non_blocking=True)
                    host_actions.copy_(actions, non_blocking=True)
                else:
                    collect(logits, actions, mapped_out)
            graphs[mode] = graph
            sessions[mode] = laya_native_host.Session(
                host.data_ptr(),
                device.data_ptr(),
                host.numel(),
                b,
                s,
                k,
                tok.pad_token_id,
                graph.raw_cuda_graph_exec(),
                stream.cuda_stream,
                logits.data_ptr(),
                actions.data_ptr(),
                output.data_ptr(),
                2,
                True,
            )
        # Refill after capture to catch stale pointers or incorrect graph inputs.
        parity = []
        for iteration in range(16):
            host.copy_(
                (torch.arange(host.numel()) + iteration * 19)
                .remainder(251)
                .to(torch.uint8)
            )
            expected = (
                torch.cat([host[: b * k].float(), host[-b * 2 :].flip(0).float()])
                .numpy()
                .view(np.uint32)
                .copy()
            )
            for mode, session in sessions.items():
                # Independent CPU expected values plus fresh poison ensure a
                # missing stage/collect cannot borrow the preceding DMA result.
                output.view(torch.int32).fill_(0x7FC01234)
                with torch.cuda.stream(owners[mode].stream):
                    device.fill_(253)
                    logits.fill_(float("nan"))
                    actions.fill_(float("nan"))
                session.replay()
                actual = output.numpy().view(np.uint32).copy()
                parity.append(
                    np.array_equal(actual, expected) and torch.equal(device.cpu(), host)
                )
        visibility["changing_input_parity"] = all(parity)
        visibility["independent_cpu_expected_and_poison"] = True
        checks.append(visibility)
        if not all(parity):
            raise RuntimeError("Captured mapped I/O changed input/output bits")
        for session in sessions.values():
            for _ in range(30):
                session.replay()
        for round_id in range(11):
            names = list(sessions)
            rng.shuffle(names)
            all_orders.append({"case": case, "round": round_id, "variants": names})
            for mode in names:
                stream = owners[mode].stream
                session = sessions[mode]
                samples = []
                for _ in range(300):
                    start = time.perf_counter_ns()
                    session.replay()
                    samples.append((time.perf_counter_ns() - start) / 1e6)
                start, end = (
                    torch.cuda.Event(enable_timing=True),
                    torch.cuda.Event(enable_timing=True),
                )
                with torch.cuda.stream(stream):
                    start.record()
                    for _ in range(300):
                        graphs[mode].replay()
                    end.record()
                end.synchronize()
                rows.append(
                    {
                        "case": case,
                        "mode": mode,
                        "round": round_id,
                        "gpu_ms": start.elapsed_time(end) / 300,
                        **common.stats(samples),
                    }
                )
        sessions.clear()
        del session
        for graph in graphs.values():
            graph.reset()
        graphs.clear()
        del graph
        # PyTorch's pinned allocator records release events on streams that
        # touched each allocation. Release aliases and all buffer views before
        # destroying those external streams, as StableHostAdapter does.
        del mapped_in, mapped_out, host_logits, host_actions, host, output
        del logits, actions, device, _
        torch.cuda.synchronize()
        for owner in owners.values():
            owner.close()
        print(
            f"Finished {case}: {visibility['input_bytes']} input bytes, "
            f"{visibility['output_bytes']} output bytes",
            flush=True,
        )
    summary = {}
    for case in ("1-short", "1-long", "16-short"):
        selected = [r for r in rows if r["case"] == case]
        timing = {
            mode: {r["round"]: r["p50_ms"] for r in selected if r["mode"] == mode}
            for mode in ("dma", "mapped-output", "mapped-both")
        }
        summary[case] = {
            mode: {
                "p50_ms": statistics.median(
                    [v for r in selected if r["mode"] == mode for v in r["samples_ms"]]
                ),
                "gpu_ms": statistics.median(
                    [r["gpu_ms"] for r in selected if r["mode"] == mode]
                ),
                "faster_rounds": sum(
                    timing[mode][i] < timing["dma"][i] for i in range(11)
                ),
                "median_paired_saving_ms": statistics.median(
                    [timing["dma"][i] - timing[mode][i] for i in range(11)]
                ),
            }
            for mode in timing
        }
    report.update(
        visibility=checks,
        all_exact=True,
        rows=rows,
        orders=all_orders,
        summary=summary,
        source_after=hashes(),
    )
    report["sources_unchanged"] = report["source_before"] == report["source_after"]
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "visibility": checks,
                "summary": summary,
                "sources_unchanged": report["sources_unchanged"],
            },
            indent=2,
        ),
        flush=True,
    )
    if not report["sources_unchanged"]:
        raise RuntimeError("Probe source changed during execution")


if __name__ == "__main__":
    main()
