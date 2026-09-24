"""Full-request screen for prefetch hints fused into existing GPU kernels."""

import argparse
import json
from pathlib import Path

import torch

from experiments.native import common

from .engine import FrontierEngine
from .holdout import requests
from .prefetch import COMPILED


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--modes", nargs="+", default=["rope", "geglu", "both"])
    parser.add_argument("--chunks", nargs="+", type=int, default=[16384, 65536])
    parser.add_argument("--instructions-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    if args.instructions_only:
        report = {"metadata": common.metadata(), "kernels": {}}
        for chunk in args.chunks:
            with FrontierEngine(
                policy="bf16-splitk-exact",
                attention="native",
                fuse_reduce_norm=True,
                token_tables=True,
                prefetch="both",
                prefetch_chunk=chunk,
                max_graphs=1,
            ) as engine:
                engine.predict(**common.workload(1, "short"))
                for kind, kernel in COMPILED.items():
                    instructions = [
                        line.strip()
                        for line in kernel.asm["ptx"].splitlines()
                        if "cp.async.bulk.prefetch" in line
                    ]
                    if not instructions:
                        raise RuntimeError(
                            "Prefetch instruction missing from compiled kernel"
                        )
                    report["kernels"][f"{kind}-{chunk}"] = {
                        "instructions": instructions,
                        "shared_bytes": kernel.metadata.shared,
                    }
        Path("results/frontier/prefetch-instructions.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
        print(json.dumps(report, indent=2), flush=True)
        return
    options = {
        "policy": "bf16-splitk-exact-short-compiled",
        "attention": "native",
        "fuse_reduce_norm": True,
        "token_tables": True,
        "max_graphs": 4,
    }
    report = {"metadata": common.metadata(), "variants": {}}
    path = Path("results/frontier/prefetch.json")
    with FrontierEngine(**options) as baseline:
        for mode in args.modes:
            for chunk in args.chunks:
                label = f"{mode}-{chunk}"
                with FrontierEngine(
                    **options, prefetch=mode, prefetch_chunk=chunk
                ) as candidate:
                    row = {"configuration": {"mode": mode, "chunk": chunk}}
                    row["timings"] = common.benchmark_interleaved(
                        {"prior": baseline, label: candidate},
                        cases=[(1, "short")],
                        rounds=7,
                        repeats=100,
                    )
                    row["parity"] = []
                    for request in requests():
                        prepared = candidate.prepare(**request)
                        a = candidate.run_prepared(prepared)
                        b = baseline.run_prepared(baseline.prepare(**request))
                        row["parity"].append(
                            common.compare_outputs(a, b, prepared, candidate.agent)
                        )
                    row["all_exact"] = all(
                        r["exact_logits_and_actions"] for r in row["parity"]
                    )
                    row["instructions"] = {
                        kind: [
                            line.strip()
                            for line in kernel.asm["ptx"].splitlines()
                            if "cp.async.bulk.prefetch" in line
                        ]
                        for kind, kernel in COMPILED.items()
                    }
                    report["variants"][label] = row
                    path.write_text(json.dumps(report, indent=2) + "\n")
                    print(label, "exact", row["all_exact"], flush=True)


if __name__ == "__main__":
    main()
