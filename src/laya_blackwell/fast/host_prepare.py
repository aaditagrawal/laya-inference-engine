"""Omit unused tokenizer offsets and optionally encode a question in one Rust call.

Sequence assembly is adapted from the pinned Apache-2.0 Laya build_sequence;
see the repository NOTICE. Request validation remains the existing implementation.
All token memoization belongs to one request and is discarded afterward.
"""

from types import FunctionType

from laya.common import build_sequence as original_build_sequence
from laya.common import render_options, serialize_state

from laya_blackwell.protocol import prepare_request


class RustRequestTokenizer:
    """Call the tokenizer's existing Rust engine without the Transformers wrapper.

    The source tokenizer is cloned once, so disabling batch padding/truncation
    cannot mutate another caller. The SDK's sequence builder still controls its
    own truncation and marker sanitization. Memoization lasts one request.
    """

    def __init__(self, tokenizer, backend):
        self.tokenizer, self.backend = tokenizer, backend
        self.cache = {}
        # Avoid repeated Transformers special-token property lookups per option.
        for name in ("mask_token", "mask_token_id", "cls_token_id", "sep_token_id"):
            setattr(self, name, getattr(tokenizer, name))

    def __getattr__(self, name):
        return getattr(self.tokenizer, name)

    def __call__(self, text, **kwargs):
        if kwargs != {"add_special_tokens": False}:
            return self.tokenizer(text, **kwargs)
        if text not in self.cache:
            self.cache[text] = {
                "input_ids": self.backend.encode(text, add_special_tokens=False).ids
            }
        return self.cache[text]


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


def prepare(tokenizer, backend, cfg, state, questions, *, max_questions=64):
    return _prepare_batched(
        FastRequestTokenizer(tokenizer, backend),
        cfg,
        serialize_state(state),
        questions,
        max_questions=max_questions,
        truncate_left=isinstance(state, list),
    )
