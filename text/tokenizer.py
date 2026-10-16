"""Tokenizer for the structured Cartoon Set captions."""

import json
from pathlib import Path


class CaptionTokenizer:
    """Small deterministic tokenizer built from training captions only."""

    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"

    def __init__(self):
        self.token_to_id = {self.PAD_TOKEN: 0, self.UNK_TOKEN: 1}
        self.id_to_token = {0: self.PAD_TOKEN, 1: self.UNK_TOKEN}
        self._fitted = False

    @property
    def vocab_size(self):
        return len(self.token_to_id)

    @property
    def pad_id(self):
        return self.token_to_id[self.PAD_TOKEN]

    @property
    def unk_id(self):
        return self.token_to_id[self.UNK_TOKEN]

    def _split(self, caption):
        if not isinstance(caption, str):
            raise TypeError("caption must be a string")

        tokens = caption.strip().split()
        if not tokens:
            raise ValueError("caption must contain at least one token")

        return tokens

    def fit(self, captions):
        """Build vocabulary from the supplied training captions."""
        unique_tokens = set()

        for caption in captions:
            unique_tokens.update(self._split(caption))

        regular_tokens = sorted(
            token for token in unique_tokens
            if token not in {self.PAD_TOKEN, self.UNK_TOKEN}
        )

        self.token_to_id = {
            self.PAD_TOKEN: 0,
            self.UNK_TOKEN: 1,
        }

        for token in regular_tokens:
            self.token_to_id[token] = len(self.token_to_id)

        self.id_to_token = {
            token_id: token
            for token, token_id in self.token_to_id.items()
        }

        self._fitted = True
        return self

    def _check_fitted(self):
        if not self._fitted:
            raise RuntimeError("Call fit() before encoding captions.")

    def encode(self, caption, max_length=None, padding=False, truncation=False):
        """Convert a caption into token IDs."""
        self._check_fitted()
        tokens = self._split(caption)

        if max_length is not None:
            if max_length <= 0:
                raise ValueError("max_length must be greater than zero")

            if len(tokens) > max_length:
                if not truncation:
                    raise ValueError(
                        "Caption is longer than max_length; "
                        "use truncation=True."
                    )
                tokens = tokens[:max_length]

            if padding and len(tokens) < max_length:
                tokens += [self.PAD_TOKEN] * (max_length - len(tokens))

        elif padding:
            raise ValueError("max_length is required when padding=True")

        return [
            self.token_to_id.get(token, self.unk_id)
            for token in tokens
        ]

    def attention_mask(self, token_ids):
        """Return 1 for real tokens and 0 for <PAD>."""
        return [0 if token_id == self.pad_id else 1 for token_id in token_ids]

    def decode(self, token_ids, skip_padding=True):
        """Convert token IDs back into tokens."""
        self._check_fitted()
        tokens = []

        for token_id in token_ids:
            if token_id not in self.id_to_token:
                raise ValueError(f"Unknown token ID: {token_id}")

            token = self.id_to_token[token_id]
            if skip_padding and token == self.PAD_TOKEN:
                continue
            tokens.append(token)

        return " ".join(tokens)

    def save(self, path):
        """Save the vocabulary as JSON."""
        self._check_fitted()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as file:
            json.dump(self.token_to_id, file, indent=2, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        """Load a previously saved vocabulary."""
        path = Path(path)

        with path.open("r", encoding="utf-8") as file:
            token_to_id = json.load(file)

        tokenizer = cls()
        tokenizer.token_to_id = {
            str(token): int(token_id)
            for token, token_id in token_to_id.items()
        }
        tokenizer.id_to_token = {
            token_id: token
            for token, token_id in tokenizer.token_to_id.items()
        }
        tokenizer._fitted = True

        return tokenizer
