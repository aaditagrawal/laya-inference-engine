"""Mutation, fallback, logging and exception checks for header preparation.

Run with uv under the experiment lock. This probe does not load model weights,
execute CUDA, or measure latency. It changes only its local tokenizer objects
and restores temporary class overrides before the next case.
"""

import copy
import hashlib
import json
import logging
import warnings
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from types import MethodType
from unittest.mock import patch

from huggingface_hub.constants import HF_HUB_CACHE
from tokenizers import AddedToken, Tokenizer
from transformers import AutoTokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.tokenization_utils_tokenizers import TokenizersBackend

from laya_blackwell.engine import REVISION
from laya_blackwell.workloads import workload

from .header_prepare import prepare as candidate
from .host_prepare import FastRequestTokenizer
from .host_prepare import prepare as reference


class Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.rows = []

    def emit(self, record):
        self.rows.append([record.name, record.levelname, record.getMessage()])


class PlainSubclass(TokenizersBackend):
    pass


class ConverterSubclass(TokenizersBackend):
    def convert_tokens_to_ids(self, value):
        return 42


class NoDictSubclass(TokenizersBackend):
    mask_token = "[MASK]"
    mask_token_id = 50284
    cls_token_id = 50281
    sep_token_id = 50282

    def __getattribute__(self, name):
        if name == "__dict__":
            raise RuntimeError("subclass disallows dictionary access")
        return super().__getattribute__(name)


class CustomMap(dict):
    pass


class CustomString(str):
    pass


def hashes():
    paths = list(Path("experiments/frontier").glob("header_prepare*.py"))
    paths += [
        Path("experiments/frontier/host_prepare.py"),
        Path("experiments/native/host/adapter.py"),
        Path("src/laya_blackwell/protocol.py"),
        Path(
            ".venv/lib/python3.12/site-packages/transformers/tokenization_utils_base.py"
        ),
        Path(
            ".venv/lib/python3.12/site-packages/transformers/tokenization_utils_tokenizers.py"
        ),
    ]
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}


def main():
    before = hashes()
    root = Path(HF_HUB_CACHE) / "models--convaiinnovations--laya/snapshots" / REVISION
    cfg = json.loads((root / "rl_agent_config.json").read_text())
    original = AutoTokenizer.from_pretrained(root / "tokenizer", local_files_only=True)
    assert type(original) is TokenizersBackend
    cases = [("ordinary", lambda tok: None, nullcontext, True)]

    def special(name, value):
        return lambda tok: setattr(tok, name, value)

    for name in ("mask_token", "cls_token", "sep_token"):
        for label, value, fast in (
            ("unset", None, False),
            ("unknown", "<absent-header-token>", False),
            ("changed", "[SEP]" if name != "sep_token" else "[CLS]", True),
            ("added-value", AddedToken("[MASK]", lstrip=True, normalized=False), True),
            ("empty", "", False),
            ("str-subclass", CustomString("[MASK]"), False),
        ):
            cases.append((f"{name}-{label}", special(name, value), nullcontext, fast))
        cases.append(
            (
                f"{name}-deleted-definition",
                lambda tok, name=name: tok._special_tokens_map.pop(name),
                nullcontext,
                False,
            )
        )
        for verbose in (True, False):

            def unset_verbose(tok, name=name, verbose=verbose):
                tok.verbose = verbose
                setattr(tok, name, None)

            cases.append(
                (f"{name}-unset-verbose-{verbose}", unset_verbose, nullcontext, False)
            )
        cases.append(
            (
                f"{name}-invalid-definition",
                lambda tok, name=name: tok._special_tokens_map.__setitem__(name, 37),
                nullcontext,
                False,
            )
        )

    def added(tok):
        tok.add_special_tokens({"mask_token": AddedToken("<new-mask>", lstrip=True)})

    cases.append(("added-new-vocabulary-token", added, nullcontext, True))
    cases.append(
        (
            "custom-special-token-map",
            lambda tok: setattr(
                tok, "_special_tokens_map", CustomMap(tok._special_tokens_map)
            ),
            nullcontext,
            False,
        )
    )
    default_getattribute = TokenizersBackend.__getattribute__

    def custom_getattribute(self, name):
        if name == "mask_token_id":
            return 42
        return default_getattribute(self, name)

    cases.append(
        (
            "class-getattribute-override",
            lambda tok: None,
            lambda: patch.object(
                TokenizersBackend, "__getattribute__", custom_getattribute
            ),
            False,
        )
    )
    for cls in (PlainSubclass, ConverterSubclass, NoDictSubclass):
        cases.append(
            (
                cls.__name__,
                lambda tok, cls=cls: object.__setattr__(tok, "__class__", cls),
                nullcontext,
                False,
            )
        )

    def changed_converter(self, value):
        logging.getLogger("header_prepare_extra.converter").warning("convert %s", value)
        return 42

    def raising_converter(self, value):
        raise ValueError(f"custom converter rejects {value}")

    for method in ("convert_tokens_to_ids", "_convert_token_to_id_with_added_voc"):
        for label, function in (
            ("changed", changed_converter),
            ("raises", raising_converter),
        ):
            cases.append(
                (
                    f"instance-{method}-{label}",
                    lambda tok, method=method, function=function: setattr(
                        tok, method, MethodType(function, tok)
                    ),
                    nullcontext,
                    False,
                )
            )
            cases.append(
                (
                    f"class-{method}-{label}",
                    lambda tok: None,
                    lambda method=method, function=function: patch.object(
                        TokenizersBackend, method, function
                    ),
                    False,
                )
            )
    for name in ("mask_token", "mask_token_id", "cls_token_id", "sep_token_id"):
        value = "[SEP]" if name == "mask_token" else 42
        cases.append(
            (
                f"instance-shadow-{name}",
                lambda tok, name=name, value=value: tok.__dict__.__setitem__(
                    name, value
                ),
                nullcontext,
                False,
            )
        )
        cases.append(
            (
                f"class-shadow-{name}",
                lambda tok: None,
                lambda name=name, value=value: patch.object(
                    TokenizersBackend, name, value, create=True
                ),
                False,
            )
        )
    cases.append(
        (
            "instance-special-attributes-removed",
            lambda tok: setattr(
                tok, "SPECIAL_TOKENS_ATTRIBUTES", ["cls_token", "sep_token"]
            ),
            nullcontext,
            False,
        )
    )
    cases.append(
        (
            "class-special-attributes-removed",
            lambda tok: None,
            lambda: patch.object(
                TokenizersBackend,
                "SPECIAL_TOKENS_ATTRIBUTES",
                ["cls_token", "sep_token"],
            ),
            False,
        )
    )
    default_getattr = PreTrainedTokenizerBase.__getattr__

    def custom_getattr(self, name):
        if name == "mask_token_id":
            return 42
        return default_getattr(self, name)

    cases.append(
        (
            "class-getattr-override",
            lambda tok: None,
            lambda: patch.object(TokenizersBackend, "__getattr__", custom_getattr),
            False,
        )
    )

    fixtures = {
        "ordinary": workload(1, "short"),
        "special-token-content": {
            **workload(1, "short"),
            "state": "Quoted [MASK], [SEP], and <new-mask> content.",
        },
        "empty-questions": {"state": "", "questions": {}},
        "invalid-question": {"state": "", "questions": {"q": {"type": "unknown"}}},
    }
    results = []
    # Capture both Transformers' nonpropagating logger and ordinary custom logs.
    loggers = [logging.getLogger("transformers"), logging.getLogger()]
    original_init = FastRequestTokenizer.__init__
    for name, mutate, context, expected_fast in cases:
        tok = copy.deepcopy(original)
        backend = Tokenizer.from_str(tok.backend_tokenizer.to_str())
        backend.no_padding()
        backend.no_truncation()
        mutate(tok)
        with context():
            for fixture_name, fixture in fixtures.items():
                outcomes = {}
                calls = {}
                for mode in ("reference", "candidate"):
                    counter = [0]

                    def counted(self, *args, counter=counter, **kwargs):
                        counter[0] += 1
                        return original_init(self, *args, **kwargs)

                    capture = Capture()
                    for logger in loggers:
                        logger.addHandler(capture)
                    try:
                        with warnings.catch_warnings(record=True) as caught:
                            warnings.simplefilter("always")
                            with patch.object(
                                FastRequestTokenizer, "__init__", counted
                            ):
                                try:
                                    kwargs = (
                                        {"mode": "batch"} if mode == "reference" else {}
                                    )
                                    fn = reference if mode == "reference" else candidate
                                    value = fn(tok, backend, cfg, **fixture, **kwargs)
                                    outcome = {"prepared": asdict(value)}
                                except Exception as error:  # noqa: BLE001 - compare public errors.
                                    outcome = {
                                        "error_type": type(error).__name__,
                                        "error": str(error),
                                    }
                            outcome["warnings"] = [
                                [w.category.__name__, str(w.message)] for w in caught
                            ]
                            outcome["logs"] = capture.rows
                    finally:
                        for logger in loggers:
                            logger.removeHandler(capture)
                    outcomes[mode] = outcome
                    calls[mode] = counter[0]
                exact = outcomes["reference"] == outcomes["candidate"]
                fallback_correct = calls["candidate"] == int(not expected_fast)
                results.append(
                    {
                        "mutation": name,
                        "fixture": fixture_name,
                        "exact": exact,
                        "fallback_expected": not expected_fast,
                        "fallback_observed": calls["candidate"] == 1,
                        "fallback_correct": fallback_correct,
                        **({"outcomes": outcomes} if not exact else {}),
                    }
                )
    after = hashes()
    report = {
        "method": "CPU-only mutation/error/logging probe; no timings or CUDA",
        "cases": len(results),
        "all_exact": all(r["exact"] for r in results),
        "all_fallbacks_as_expected": all(r["fallback_correct"] for r in results),
        "source_before": before,
        "source_after": after,
        "sources_unchanged": before == after,
        "candidate_initial_sha256": hashlib.sha256(
            Path(".research/header_prepare.initial.py").read_bytes()
        ).hexdigest(),
        "candidate_hardened_sha256": after["experiments/frontier/header_prepare.py"],
        "rows": results,
    }
    Path("results/frontier/header-prepare-extra.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key not in {"rows", "source_before", "source_after"}
            },
            indent=2,
        )
    )
    failures = [r for r in results if not r["exact"] or not r["fallback_correct"]]
    print(json.dumps(failures, indent=2))
    if not report["all_exact"] or not report["sources_unchanged"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
