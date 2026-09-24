"""Omit unused tokenizer offsets and optionally encode a question in one Rust call.

Sequence assembly is adapted from the pinned Apache-2.0 Laya build_sequence;
see the repository NOTICE. Request validation remains the existing implementation.
All token memoization belongs to one request and is discarded afterward.
"""

from hashlib import sha256
from types import FunctionType, MethodType

from laya.common import build_sequence as original_build_sequence
from laya.common import render_options, serialize_state

from experiments.native.host.adapter import RustRequestTokenizer
from laya_blackwell.protocol import prepare_request


class FastRequestTokenizer(RustRequestTokenizer):
    def encode_ids(self, text):
        try:
            return self.backend.encode_batch_fast([text], add_special_tokens=False)[
                0
            ].ids
        except TypeError:
            # The batch binding words malformed-Unicode errors differently.
            # The original binding supplies the established exception contract.
            return self.backend.encode(text, add_special_tokens=False).ids

    def __call__(self, text, **kwargs):
        if kwargs != {"add_special_tokens": False}:
            return self.tokenizer(text, **kwargs)
        if text not in self.cache:
            self.cache[text] = {"input_ids": self.encode_ids(text)}
        return self.cache[text]

    def encode_many(self, texts):
        missing = list(dict.fromkeys(text for text in texts if text not in self.cache))
        if missing:
            try:
                encoded = self.backend.encode_batch_fast(
                    missing, add_special_tokens=False
                )
            except TypeError:
                encoded = [
                    self.backend.encode(text, add_special_tokens=False)
                    for text in missing
                ]
            self.cache.update(
                (text, {"input_ids": encoding.ids})
                for text, encoding in zip(missing, encoded)
            )
        return [self.cache[text]["input_ids"] for text in texts]


TOKENIZER_SHA256 = "f64652c0d4292921662f8a34068ed38c9db4a8e78daae667eaff88a30494ef8b"


def template_tokens(backend):
    """Three immutable grammar fragments, gated to the inspected tokenizer.

    Its NFC normalizer cannot join a word through the following ASCII colon.
    The ByteLevel regex splits words from punctuation, and no added token spans
    this boundary. The colon and following space remain with the variable text.
    """
    if sha256(backend.to_str().encode()).hexdigest() != TOKENIZER_SHA256:
        return {}
    return {
        kind: tuple(backend.encode(kind + " question", add_special_tokens=False).ids)
        for kind in ("choice", "score", "noul")
    }


class TemplateRequestTokenizer(FastRequestTokenizer):
    def __init__(self, tokenizer, backend, templates):
        super().__init__(tokenizer, backend)
        self.templates = templates

    def __call__(self, text, **kwargs):
        if kwargs != {"add_special_tokens": False}:
            return self.tokenizer(text, **kwargs)
        if text not in self.cache:
            kind, separator, tail = text.partition(" question")
            prefix = (
                self.templates.get(kind)
                if separator and tail.startswith(": ")
                else None
            )
            self.cache[text] = {
                "input_ids": (
                    list(prefix) + self.encode_ids(tail)
                    if prefix is not None
                    else self.encode_ids(text)
                )
            }
        return self.cache[text]


def _build_sequence(
    tok,
    state,
    q,
    max_len=512,
    head_max_len=192,
    option_order=None,
    truncate_left=False,
):
    mask_tok = tok.mask_token
    opts = render_options(q)
    order = option_order if option_order is not None else list(range(len(opts)))
    ins = str(q["ins"]).replace(mask_tok, " ")
    try:
        texts = [f"{q['t']} question: {ins}"]
        texts.extend(" " + opts[i].replace(mask_tok, " ") for i in order)
        texts.append(serialize_state(state).replace(mask_tok, " "))
    except (AttributeError, TypeError, ValueError):
        # Invalid option labels can fail before a bad earlier head is encoded.
        # Replay the upstream operation order to preserve which error is raised.
        original = RustRequestTokenizer(tok.tokenizer, tok.backend)
        original.cache = tok.cache
        return original_build_sequence(
            original, state, q, max_len, head_max_len, option_order, truncate_left
        )
    encoded = tok.encode_many(texts)
    head_ids = encoded[0]
    opt_ids = [[tok.mask_token_id] + ids[:48] for ids in encoded[1:-1]]
    opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    if opt_budget < 16:
        per = max(4, (head_max_len - 16) // max(1, len(opt_ids)))
        opt_ids = [o[:per] for o in opt_ids]
        opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    head_ids = head_ids[: max(8, opt_budget)]
    ids = [tok.cls_token_id] + head_ids + [tok.sep_token_id]
    markers = []
    for option in opt_ids:
        markers.append(len(ids))
        ids.extend(option)
    ids.append(tok.sep_token_id)
    room = max(0, max_len - len(ids) - 1)
    state_ids = encoded[-1]
    state_ids = (
        state_ids[max(0, len(state_ids) - room) :]
        if truncate_left
        else state_ids[:room]
    )
    ids = ids + state_ids + [tok.sep_token_id]
    return ids[:max_len], [marker for marker in markers if marker < max_len]


# Clone the function namespace once, so production globals are never changed.
# This retains the complete validation order, messages, and normalization code.
_prepare_batched = FunctionType(
    prepare_request.__code__,
    {**prepare_request.__globals__, "build_sequence": _build_sequence},
    name="prepare_batched",
    argdefs=prepare_request.__defaults__,
    closure=prepare_request.__closure__,
)
_prepare_batched.__kwdefaults__ = prepare_request.__kwdefaults__


def prepare(
    tokenizer,
    backend,
    cfg,
    state,
    questions,
    *,
    mode="single",
    max_questions=64,
    templates=None,
):
    if mode not in {"single", "batch", "template"}:
        raise ValueError(mode)
    function = _prepare_batched if mode == "batch" else prepare_request
    return function(
        TemplateRequestTokenizer(tokenizer, backend, templates or {})
        if mode == "template"
        else FastRequestTokenizer(tokenizer, backend),
        cfg,
        serialize_state(state),
        questions,
        max_questions=max_questions,
        truncate_left=isinstance(state, list),
    )


def install(engine, mode="single"):
    """Replace only preparation on an existing adapter, preserving its ownership."""
    if mode not in {"single", "batch", "template"}:
        raise ValueError(mode)
    adapter = engine.adapter
    previous = adapter.prepare
    templates = template_tokens(adapter.rust_tokenizer) if mode == "template" else {}

    def optimized(adapter, state, questions):
        return prepare(
            adapter.agent.tok,
            adapter.rust_tokenizer,
            adapter.agent.cfg,
            state,
            questions,
            mode=mode,
            max_questions=adapter.base.max_questions,
            templates=templates,
        )

    adapter.prepare = MethodType(optimized, adapter)
    return previous
