"""Attribute retained request-preparation CPU work without loading the model.

Use the exclusive experiment lock. cProfile values diagnose Python overhead;
they are not uninstrumented latency measurements.
"""

import cProfile
import hashlib
import json
import pstats
from datetime import UTC, datetime
from pathlib import Path

from huggingface_hub.constants import HF_HUB_CACHE
from tokenizers import Tokenizer
from transformers import AutoTokenizer

from laya_blackwell.engine import REVISION
from laya_blackwell.workloads import workload

from .holdout import requests
from .host_prepare import prepare


def main():
    root = Path(HF_HUB_CACHE) / "models--convaiinnovations--laya/snapshots" / REVISION
    config = json.loads((root / "rl_agent_config.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(root / "tokenizer", local_files_only=True)
    backend = Tokenizer.from_str(tokenizer.backend_tokenizer.to_str())
    backend.no_padding()
    backend.no_truncation()
    fixtures = [workload(1, "short"), *requests()]

    def run():
        for fixture in fixtures:
            prepare(tokenizer, backend, config, **fixture, mode="batch")

    run()
    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(20):
        run()
    profiler.disable()
    stats = pstats.Stats(profiler)
    rows = []
    for (filename, line, function), values in stats.stats.items():
        primitive, total, own, cumulative, _ = values
        rows.append(
            {
                "file": filename,
                "line": line,
                "function": function,
                "calls": total,
                "primitive_calls": primitive,
                "own_seconds": own,
                "cumulative_seconds": cumulative,
            }
        )
    report = {
        "created_utc": datetime.now(UTC).isoformat(),
        "scope": "Instrumented CPU preparation, no model load, inference or HTTP",
        "model_revision": REVISION,
        "requests": len(fixtures) * 20,
        "total_seconds": stats.total_tt,
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (
                Path(__file__).relative_to(Path.cwd()),
                Path("experiments/frontier/host_prepare.py"),
                Path("experiments/native/host/adapter.py"),
                Path("src/laya_blackwell/protocol.py"),
            )
        },
        "rows": sorted(rows, key=lambda row: -row["own_seconds"]),
    }
    Path("results/frontier/prepare-cpu-profile.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps({**report, "rows": report["rows"][:20]}, indent=2))


if __name__ == "__main__":
    main()
