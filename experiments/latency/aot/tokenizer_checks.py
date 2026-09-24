"""Compare lightweight tokenizer preparation with AutoTokenizer on all fixtures."""

import argparse
import json
from pathlib import Path

from laya.common import serialize_state
from tokenizers import Tokenizer
from transformers import AutoTokenizer

from experiments.native.host.adapter import RustRequestTokenizer
from laya_blackwell.protocol import prepare_request

from .tokenizer import RuntimeTokenizer


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    cfg = json.loads((args.runtime / "rl_agent_config.json").read_text())
    requests_path = (
        Path(__file__).resolve().parents[3]
        / "results/native-optimizations/reference/requests.json"
    )
    requests = json.loads(requests_path.read_text())
    tokenizers = [
        AutoTokenizer.from_pretrained(
            args.runtime / "tokenizer", local_files_only=True
        ),
        RuntimeTokenizer(args.runtime / "tokenizer"),
    ]
    backends = [
        Tokenizer.from_str(tok.backend_tokenizer.to_str()) for tok in tokenizers
    ]
    for backend in backends:
        backend.no_padding()
        backend.no_truncation()
    cases = []
    for index, request in enumerate(requests):
        prepared = [
            prepare_request(
                RustRequestTokenizer(tok, backend),
                cfg,
                serialize_state(request["state"]),
                request["questions"],
                max_questions=64,
                truncate_left=isinstance(request["state"], list),
            )
            for tok, backend in zip(tokenizers, backends)
        ]
        cases.append(
            {
                "case": index,
                "exact": prepared[0] == prepared[1],
                "decisions": len(prepared[0].items),
            }
        )
    report = {
        "requests": len(cases),
        "decisions": sum(c["decisions"] for c in cases),
        "all_prepared_requests_exact": all(c["exact"] for c in cases),
        "scope": "All cached reference requests, comparing token IDs, markers, masks, IDs, questions, and input-token counts. Includes literal special markers, conversation left truncation and many-option cases. No model inference.",
        "cases": cases,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "cases"}), flush=True)
    if not report["all_prepared_requests_exact"]:
        raise RuntimeError("Lightweight tokenizer changed a prepared request")


if __name__ == "__main__":
    main()
