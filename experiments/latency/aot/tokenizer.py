"""Load the serialized Rust tokenizer without initializing Transformers."""

import json
from pathlib import Path

from tokenizers import Tokenizer


class RuntimeTokenizer:
    """Expose the tokenizer interface consumed by Laya's sequence builder."""

    def __init__(self, directory):
        directory = Path(directory)
        self.backend_tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
        config = json.loads((directory / "tokenizer_config.json").read_text())
        for name in ("cls", "sep", "pad", "mask", "unk"):
            token = config[f"{name}_token"]
            if isinstance(token, dict):
                token = token["content"]
            identifier = self.backend_tokenizer.token_to_id(token)
            if identifier is None:
                raise ValueError(
                    f"Special token {token!r} is absent from the serialized tokenizer"
                )
            setattr(self, f"{name}_token", token)
            setattr(self, f"{name}_token_id", identifier)

    def __call__(self, text, *, add_special_tokens=True):
        return {
            "input_ids": self.backend_tokenizer.encode(
                text, add_special_tokens=add_special_tokens
            ).ids
        }
