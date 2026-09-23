"""Time the upstream Laya FlashAttention path, not our custom graph engine.

Run with: uv run --extra hub-kernels python scripts/probe_flash_attention.py
"""
from importlib.metadata import version
import json
from pathlib import Path
import time

import numpy as np

from laya_blackwell.engine import model_path
from laya_blackwell.workloads import workload
from laya import Agent


def main():
    result = {"packages": {p: version(p) for p in ("torch", "transformers", "kernels")},
              "kernel_revision": "81fb77c12b2ad5d69380669b46739d5868614502",
              "scope": "Upstream SDK with FlashAttention, 30 serial timed requests after 5 warmups. Not a custom graph-engine attention comparison."}
    try:
        agent = Agent(model_path(), device="cuda:0")
        agent.model.encoder.set_attn_implementation(
            "kernels-community/flash-attn2@" + result["kernel_revision"]
        )
        rows = []
        for count, length in ((1, "short"), (16, "long")):
            request = workload(count, length)
            for _ in range(5):
                agent.predict(**request)
            samples = []
            for _ in range(30):
                start = time.perf_counter()
                agent.predict(**request)
                samples.append((time.perf_counter()-start)*1000)
            if agent.device.type != "cuda":
                raise RuntimeError("Upstream fell back to CPU; invalid GPU comparison")
            rows.append({"questions": count, "length": length,
                         "p50_ms": float(np.median(samples)),
                         "p95_ms": float(np.percentile(samples, 95)), "samples_ms": samples})
        result["rows"] = rows
    except Exception as exc:
        result["error"] = repr(exc)
    path = Path("results/upstream-flash2-probe.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if "error" in result:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
