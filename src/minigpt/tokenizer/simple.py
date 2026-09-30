"""Word-level tokenizer for the Stage 1 baseline.

Splits text into words (``[\\w']+``) and single symbols (punctuation, emoji pieces),
lowercases, and maps tokens to ids with a vocabulary built from the training split.
Words not in the vocabulary become ``<unk>`` - the weakness BPE fixes in Stage 2.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

PAD, UNK = "<pad>", "<unk>"
_TOKEN_RE = re.compile(r"[\w']+|[^\w\s]")


class WordTokenizer:
    """Word-level tokenizer with a frequency-cut vocabulary.

    Ids 0 and 1 are reserved for ``<pad>`` and ``<unk>``.
    """

    def __init__(self, itos: list[str], lowercase: bool = True) -> None:
        self.itos = itos
        self.stoi = {tok: i for i, tok in enumerate(itos)}
        self.lowercase = lowercase

    @property
    def pad_id(self) -> int:
        return self.stoi[PAD]

    @property
    def unk_id(self) -> int:
        return self.stoi[UNK]

    def __len__(self) -> int:
        return len(self.itos)

    def tokenize(self, text: str) -> list[str]:
        """Split text into word and symbol tokens."""
        return _TOKEN_RE.findall(text.lower() if self.lowercase else text)

    @classmethod
    def build(cls, texts: Iterable[str], min_freq: int, max_size: int, lowercase: bool = True) -> WordTokenizer:
        """Build a vocabulary from texts: keep tokens seen >= ``min_freq`` times, most frequent first."""
        counter: Counter[str] = Counter()
        tmp = cls([PAD, UNK], lowercase)
        for text in texts:
            counter.update(tmp.tokenize(text))
        kept = [tok for tok, c in counter.most_common(max_size - 2) if c >= min_freq]
        return cls([PAD, UNK, *kept], lowercase)

    def encode(self, text: str, max_len: int | None = None) -> list[int]:
        """Map text to ids, truncated to ``max_len``. Empty text becomes ``[<unk>]``."""
        ids = [self.stoi.get(tok, self.unk_id) for tok in self.tokenize(text)][:max_len]
        return ids or [self.unk_id]

    def decode(self, ids: Iterable[int]) -> str:
        """Map ids back to space-joined tokens (lossy: case and spacing are not restored)."""
        return " ".join(self.itos[i] for i in ids if i != self.pad_id)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps({"itos": self.itos, "lowercase": self.lowercase}, ensure_ascii=False),
                              encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> WordTokenizer:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["itos"], data["lowercase"])
