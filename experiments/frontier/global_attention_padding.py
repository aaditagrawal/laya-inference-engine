"""Screen constant CUTLASS attention for padded global masks on real inputs."""

import json
from pathlib import Path

import torch
from torch.nn import functional as F

from experiments.native import common

from .attention_special_adapter import load
from .engine import FrontierEngine
from .holdout import requests
from .tune import timing


@torch.inference_mode()
def capture(engine, request):
    prepared = engine.prepare(**request)
    key = engine.base._graph_key(prepared)
    if key[:2] != (1, 64) or key[-1]:
        return None
    host = engine.base._allocate(key[:3], host=True)
    engine.base._fill(host, prepared)
    inputs = {k: v.to(engine.base.device) for k, v in host.items()}
    inputs["global_attention_unmasked"] = False
    original = F.scaled_dot_product_attention
    tensors = []

    def intercept(q, k, v, attn_mask=None, **kwargs):
        if (
            q.shape == (1, 16, 64, 64)
            and attn_mask is not None
            and attn_mask.shape[-2] == 1
        ):
            tensors.append((q, k, v, attn_mask))
        return original(q, k, v, attn_mask=attn_mask, **kwargs)

    F.scaled_dot_product_attention = intercept
    try:
        engine.base._forward(inputs)
    finally:
        F.scaled_dot_product_attention = original
    # The first head layer has the same geometry and broadcast mask. It runs
    # after all 10 global encoder calls and is outside this installer's scope.
    if len(tensors) != 11:
        raise RuntimeError(f"Expected 10 encoder plus 1 head call, got {len(tensors)}")
    return tensors[:10]


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    load()
    report = {
        "metadata": common.metadata(),
        "scope": "Padded global attention on all eligible original and holdout fixtures; real broadcast masks; kernel timings use bias prepared outside graph",
        "build": json.loads(
            Path(".research/frontier-attention-special/build.json").read_text()
        ),
        "rows": [],
    }
    with FrontierEngine(policy="bf16-exact", max_graphs=2) as engine:
        groups = []
        expected = []
        sources = []
        for suite, fixtures in [
            ("original", common.validation_requests()),
            ("holdout", requests()),
        ]:
            for index, request in enumerate(fixtures):
                group = capture(engine, request)
                if group is None:
                    continue
                expected.append(
                    [
                        F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
                        for q, k, v, mask in group
                    ]
                )
                groups.append(
                    [
                        (
                            q,
                            k,
                            v,
                            torch.zeros_like(mask, dtype=q.dtype).masked_fill_(
                                ~mask, float("-inf")
                            ),
                        )
                        for q, k, v, mask in group
                    ]
                )
                sources.append({"suite": suite, "index": index, "request": request})
        report["eligible_requests"] = sources
        first = groups[0]

        def baseline():
            return [
                F.scaled_dot_product_attention(q, k, v, attn_mask=bias)
                for q, k, v, bias in first
            ]

        check = baseline()
        assert all(torch.equal(a, b) for a, b in zip(check, expected[0]))
        report["baseline_ms"], report["baseline_samples_ms"] = timing(
            baseline, repeats=100, rounds=7
        )
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ]
        ) as profiler:
            baseline()
            torch.cuda.synchronize()
        trace = Path(".research/global-attention-padding.trace.json")
        profiler.export_chrome_trace(str(trace))
        report["baseline_kernels"] = sorted(
            {
                e["name"]
                for e in json.loads(trace.read_text())["traceEvents"]
                if e.get("cat") == "kernel"
            }
        )
        for config in [
            (32, True, False),
            (64, True, False),
            (32, True, True),
            (64, True, True),
        ]:
            mismatch = []
            for group, ref in zip(groups, expected):
                actual = [
                    torch.ops.laya_frontier_attention_special.forward(
                        q, k, v, *config, bias
                    )
                    for q, k, v, bias in group
                ]
                mismatch.append(sum(int((a != b).sum()) for a, b in zip(actual, ref)))

            def candidate(config=config):
                return [
                    torch.ops.laya_frontier_attention_special.forward(
                        q, k, v, *config, bias
                    )
                    for q, k, v, bias in first
                ]

            ms, samples = timing(candidate, repeats=100, rounds=7)
            row = {
                "configuration": config,
                "mismatches": sum(mismatch),
                "request_mismatches": mismatch,
                "ms": ms,
                "samples_ms": samples,
                "speedup": report["baseline_ms"] / ms,
            }
            report["rows"].append(row)
            print(
                {k: v for k, v in row.items() if k != "request_mismatches"}, flush=True
            )
            Path("results/frontier/global-attention-padding.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )


if __name__ == "__main__":
    main()
