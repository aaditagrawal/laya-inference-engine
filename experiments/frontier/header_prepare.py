"""Avoid generic special-token attribute dispatch for the pinned fast tokenizer.

Read current token definitions and vocabulary on every request. Unknown tokenizer
types, unset definitions and unsupported values keep the original initialization.
No request or tokenization result is cached across calls.
"""

from types import MethodType

from laya.common import serialize_state
from tokenizers import AddedToken, Tokenizer
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from transformers.tokenization_utils_tokenizers import TokenizersBackend

from .host_prepare import FastRequestTokenizer, _prepare_batched

_GETATTR = PreTrainedTokenizerBase.__getattr__
_CONVERT = PreTrainedTokenizerBase.convert_tokens_to_ids
_LOOKUP = TokenizersBackend._convert_token_to_id_with_added_voc
_MISSING = object()


class DirectHeaderTokenizer(FastRequestTokenizer):
    def __init__(self, tokenizer, backend):
        if (
            type(tokenizer) is not TokenizersBackend
            or TokenizersBackend.__getattribute__ is not object.__getattribute__
            or TokenizersBackend.__getattr__ is not _GETATTR
            or TokenizersBackend.convert_tokens_to_ids is not _CONVERT
            or TokenizersBackend._convert_token_to_id_with_added_voc is not _LOOKUP
            or any(
                getattr(TokenizersBackend, name, _MISSING) is not _MISSING
                for name in (
                    "mask_token",
                    "mask_token_id",
                    "cls_token_id",
                    "sep_token_id",
                )
            )
        ):
            super().__init__(tokenizer, backend)
            return
        state = tokenizer.__dict__
        special = state.get("_special_tokens_map")
        original = state.get("_tokenizer")
        recognized = tokenizer.SPECIAL_TOKENS_ATTRIBUTES
        if (
            type(special) is not dict
            or type(original) is not Tokenizer
            or type(recognized) not in (list, tuple)
            or any(
                name not in recognized
                for name in ("mask_token", "cls_token", "sep_token")
            )
            or any(
                name in state
                for name in (
                    "mask_token",
                    "mask_token_id",
                    "cls_token_id",
                    "sep_token_id",
                    "convert_tokens_to_ids",
                    "_convert_token_to_id_with_added_voc",
                )
            )
        ):
            super().__init__(tokenizer, backend)
            return
        values = [
            special.get(name) for name in ("mask_token", "cls_token", "sep_token")
        ]
        if any(type(value) not in (str, AddedToken) for value in values):
            super().__init__(tokenizer, backend)
            return
        strings = [str(value) for value in values]
        ids = [original.token_to_id(value) for value in strings]
        if any(value is None for value in ids):
            super().__init__(tokenizer, backend)
            return
        self.tokenizer, self.backend, self.cache = tokenizer, backend, {}
        self.mask_token = strings[0]
        self.mask_token_id, self.cls_token_id, self.sep_token_id = ids


def prepare(tokenizer, backend, cfg, state, questions, *, max_questions=64):
    return _prepare_batched(
        DirectHeaderTokenizer(tokenizer, backend),
        cfg,
        serialize_state(state),
        questions,
        max_questions=max_questions,
        truncate_left=isinstance(state, list),
    )


def install(engine):
    adapter = engine.adapter
    previous = adapter.prepare

    def optimized(adapter, state, questions):
        return prepare(
            adapter.agent.tok,
            adapter.rust_tokenizer,
            adapter.agent.cfg,
            state,
            questions,
            max_questions=adapter.base.max_questions,
        )

    adapter.prepare = MethodType(optimized, adapter)
    return previous
