"""CPU-only parity and timing for offset-free request tokenization.

Use the exclusive /tmp/laya-gpu-experiments.lock for timing: other full-request
benchmarks also measure host work, so CPU-only timing must not overlap them.
"""

import argparse
import copy
import hashlib
import json
import random
import statistics
import time
from dataclasses import asdict
from datetime import UTC, datetime
from functools import partial
from pathlib import Path

from huggingface_hub.constants import HF_HUB_CACHE
from laya.common import serialize_state
from tokenizers import Tokenizer
from transformers import AutoTokenizer

from experiments.native.common import validation_requests
from experiments.native.host.adapter import RustRequestTokenizer
from laya_blackwell.engine import REVISION
from laya_blackwell.protocol import prepare_request
from laya_blackwell.workloads import workload

from .holdout import requests
from .host_prepare import prepare, template_tokens


def outcome(function):
    try:
        return {"prepared": asdict(function())}
    except (
        AttributeError,
        TypeError,
        ValueError,
        OverflowError,
        UnicodeError,
    ) as error:
        return {"error_type": type(error).__name__, "error": str(error)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, default=Path("results/frontier/host-prepare.json")
    )
    args = parser.parse_args()
    root = Path(HF_HUB_CACHE) / "models--convaiinnovations--laya/snapshots" / REVISION
    cfg = json.loads((root / "rl_agent_config.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(root / "tokenizer", local_files_only=True)
    backend = Tokenizer.from_str(tokenizer.backend_tokenizer.to_str())
    backend.no_padding()
    backend.no_truncation()
    templates = template_tokens(backend)
    assert len(templates) == 3

    def run(mode, fixture, config=None, max_questions=64):
        config = cfg if config is None else config
        if mode == "baseline":
            return prepare_request(
                RustRequestTokenizer(tokenizer, backend),
                config,
                serialize_state(fixture["state"]),
                fixture["questions"],
                max_questions=max_questions,
                truncate_left=isinstance(fixture["state"], list),
            )
        return prepare(
            tokenizer,
            backend,
            config,
            **fixture,
            mode=mode,
            max_questions=max_questions,
            templates=templates,
        )

    modes = ["baseline", "single", "batch", "template"]
    cases = [(fixture, cfg, 64) for fixture in validation_requests() + requests()]
    rng = random.Random(46292)
    fragments = [
        "",
        " ",
        "  ",
        "\t",
        "\n",
        "é",
        "e\u0301",
        "नमस्ते",
        "🙂",
        "\u200d",
        "[MASK]",
        "[SEP]",
        "[CLS]",
        "[unused0]",
        "|||IP_ADDRESS|||",
        "<|endoftext|>",
        "\0",
        ":",
        "'",
        "¿",
        "中文",
        "\u2028",
        "\ufeff",
        "\U0010ffff",
    ]
    for index in range(384):
        text = "".join(rng.choices(fragments, k=rng.randrange(1, 30)))
        count = rng.choice([1, 2, 3, 4, 5, 8, 17, 33])
        kind = rng.choice(["choice", "score", "noul"])
        question = {"type": kind, "instructions": text}
        if kind == "choice":
            question["criteria"] = {str(i): rng.choice(fragments) for i in range(count)}
        elif kind == "score":
            question["criteria"] = rng.choices(fragments, k=count)
        elif index % 2:
            question["criteria"] = {False: text, True: {"text": text}}
            question["labels"] = {"false": "No", "true": "Yes"}
        state = rng.choice([text, {"body": text}, [{"content": text * 50}]])
        config = {
            **cfg,
            "max_len": rng.choice([1, 8, 16, 64, 127, 512]),
            "head_max_len": rng.choice([1, 8, 16, 80, 192]),
        }
        cases.append(({"state": state, "questions": {"q": question}}, config, 64))
    invalid = [
        None,
        {"type": "unknown", "instructions": "?"},
        {"type": "choice", "criteria": ["a"]},
        {"type": "choice", "instructions": "?", "criteria": []},
        {"type": "score", "instructions": "?", "criteria": {}},
        {"type": "noul", "instructions": "?", "criteria": []},
        {"type": [], "instructions": "?"},
        {"type": "choice", "instructions": "?", "criteria": [["a"]]},
        {
            "type": "noul",
            "instructions": "?",
            "labels": {"false": "same", "true": "same"},
        },
    ]
    for bad in invalid:
        cases.append(
            (
                {
                    "state": "state",
                    "questions": {
                        "valid": {"type": "noul", "instructions": "?"},
                        "broken": bad,
                    },
                },
                cfg,
                64,
            )
        )
    for value in ([], None, "bad", {}):
        cases.append(({"state": "", "questions": value}, cfg, 64))
    for limit in (0, -1, True, 1.5):
        cases.append((workload(1, "short"), cfg, limit))
        for key in ("max_len", "head_max_len"):
            cases.append((workload(1, "short"), {**cfg, key: limit}, 64))
    for count in (64, 65):
        cases.append(
            (
                {
                    "state": "",
                    "questions": {
                        str(i): {"type": "noul", "instructions": "?"}
                        for i in range(count)
                    },
                },
                cfg,
                64,
            )
        )
    for text in ("\ud800", "x\udfff", "choice question: \ud800"):
        cases.append(
            (
                {
                    "state": text,
                    "questions": {"q": {"type": "noul", "instructions": text}},
                },
                cfg,
                64,
            )
        )
    for instruction in ("valid", "\ud800"):
        cases.append(
            (
                {
                    "state": "",
                    "questions": {
                        "q": {
                            "type": "choice",
                            "instructions": instruction,
                            "criteria": {1: None},
                        }
                    },
                },
                cfg,
                64,
            )
        )
    parity = []
    for index, (fixture, config, limit) in enumerate(cases):
        before = copy.deepcopy(fixture)
        outcomes = {
            mode: outcome(partial(run, mode, fixture, config, limit)) for mode in modes
        }
        equal = {mode: outcomes[mode] == outcomes["baseline"] for mode in modes[1:]}
        parity.append(
            {
                "case": index,
                "exact": equal,
                "input_unchanged": before == fixture,
                "rejected": "error" in outcomes["baseline"],
            }
        )
        if not all(equal.values()):
            print(json.dumps({"case": index, "outcomes": outcomes}, ensure_ascii=True))
    print(f"Checked {len(parity)} prepared-request and error cases", flush=True)

    timing = []
    for case in ("1-short", "1-long", "16-short", "changing"):
        count, length = case.split("-") if case != "changing" else ("1", "short")
        fixtures = requests() if case == "changing" else [workload(int(count), length)]
        for mode in modes:
            for fixture in fixtures[:10]:
                run(mode, fixture)
        for round_id in range(9):
            order = list(modes)
            rng.shuffle(order)
            indices = list(range(256))
            rng.shuffle(indices)
            for mode in order:
                samples = []
                for index in indices:
                    fixture = fixtures[index % len(fixtures)]
                    start = time.perf_counter_ns()
                    run(mode, fixture)
                    samples.append((time.perf_counter_ns() - start) / 1e6)
                timing.append(
                    {
                        "case": case,
                        "mode": mode,
                        "round": round_id,
                        "p50_ms": statistics.median(samples),
                        "samples_ms": samples,
                    }
                )
    summary = {}
    for case in ("1-short", "1-long", "16-short", "changing"):
        rows = [row for row in timing if row["case"] == case]
        summary[case] = {
            mode: statistics.median(
                [
                    sample
                    for row in rows
                    if row["mode"] == mode
                    for sample in row["samples_ms"]
                ]
            )
            for mode in modes
        }
    report = {
        "created_utc": datetime.now(UTC).isoformat(),
        "scope": "CPU preparation only; no GPU inference, request cache, or output-format changes",
        "revision": REVISION,
        "tokenizer_sha256": hashlib.sha256(backend.to_str().encode()).hexdigest(),
        "source_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(Path("experiments/frontier").glob("host_*.py"))
        },
        "parity": parity,
        "all_exact": all(
            all(row["exact"].values()) and row["input_unchanged"] for row in parity
        ),
        "summary_ms": summary,
        "timing": timing,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({"all_exact": report["all_exact"], "summary_ms": summary}, indent=2),
        flush=True,
    )
    if not report["all_exact"]:
        raise RuntimeError("Preparation or error parity failed")


if __name__ == "__main__":
    main()
