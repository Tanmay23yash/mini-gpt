"""Byte-level Byte Pair Encoding (BPE) tokenizer, GPT-2 style.   [LEARN FILE]

Idea
----
Start with the 256 possible byte values as the base vocabulary (so any text, emoji
or script can be encoded and nothing is ever <unk>). Then repeat:

    1. count every adjacent pair of token ids in the training text
    2. take the most frequent pair (a, b)
    3. give it a new id  new = 256 + (number of merges so far)
    4. replace every (a, b) with new, and record  merges[(a, b)] = new

After N merges the vocabulary is 256 + N tokens, and ``vocab[new] = vocab[a] + vocab[b]``
(bytes concatenation). Frequent words like "bhai" end up as one token; rare words
are split into a few pieces ("bhaiyaaa" -> "bhai" + "ya" + "aa"), so spelling
variants share most of their tokens - the thing Stage 1's word vocab could not do.

Encoding new text applies the learned merges **in the order they were learned**
(lowest new id first), because later merges were learned on top of earlier ones.

Text is first split into chunks by a regex (``pretokenize``) and merges never
cross chunk boundaries, so a token never glues the end of one word to the next.

Tests: ``pytest tests/test_bpe.py -v``.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

# --------------------------------------------------------------------------- provided

# GPT-2's split pattern, rewritten for Python's built-in ``re`` (no \p{L}):
#   contractions | optional space + letters (incl. Devanagari vowel signs) | optional space + digits
#   | optional space + other symbols (punctuation, emoji) | whitespace runs
# The leading optional space means " bhai" is one chunk: word tokens carry their space.
PRETOKENIZE_RE = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?(?:[^\W\d_]|[\u0900-\u097F])+| ?\d+| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+"""
)

ENDOFTEXT = "<|endoftext|>"


def pretokenize(text: str) -> list[str]:
    """Split text into chunks that BPE merges may not cross.

    >>> pretokenize("bhai kya scene hai?? 😂")
    ['bhai', ' kya', ' scene', ' hai', '??', ' 😂']
    """
    return PRETOKENIZE_RE.findall(text)


# --------------------------------------------------------------------------- helpers


def get_stats(
    ids: list[int],
    counts: dict[tuple[int, int], int] | None = None,
    weight: int = 1
) -> dict[tuple[int, int], int]:
    """Count adjacent pairs in ``ids``, adding into ``counts`` if given.

    Args:
        ids: a sequence of token ids, e.g. [1, 2, 3, 1, 2].
        counts: an existing dict to update in place (lets you accumulate over many chunks).
        weight: how much each occurrence counts. Useful when ``ids`` is a chunk that
            appears ``weight`` times in the corpus - you count it once, not N times.

    Returns:
        The counts dict, e.g. {(1, 2): 2, (2, 3): 1, (3, 1): 1}.
    """
    if counts is None:
        counts = {}

    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + weight

    return counts


def merge(
    ids: list[int],
    pair: tuple[int, int],
    new_id: int
) -> list[int]:
    """Return a new list where every occurrence of ``pair`` is replaced by ``new_id``.

    Scan left to right; occurrences do not overlap:
        merge([1, 2, 3, 1, 2], (1, 2), 99) -> [99, 3, 99]
        merge([5, 5, 5], (5, 5), 7)        -> [7, 5]
    """
    out = []
    i = 0

    while i < len(ids):
        if i + 1 < len(ids) and (ids[i], ids[i + 1]) == pair:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1

    return out


# --------------------------------------------------------------------------- tokenizer


class BPETokenizer:
    """Byte-level BPE tokenizer.

    Attributes:
        merges: {(id_a, id_b): new_id}, in the order they were learned (dicts keep order).
        vocab: {id: bytes} for the 256 byte tokens and every merged token.
        special_tokens: {"<|endoftext|>": id, ...}; ids come right after the last merge.
    """

    def __init__(
        self,
        merges: dict[tuple[int, int], int] | None = None,
        special_tokens: Iterable[str] = (ENDOFTEXT,)
    ) -> None:

        self.merges: dict[tuple[int, int], int] = dict(merges or {})

        self.vocab: dict[int, bytes] = self._build_vocab()

        first_special = 256 + len(self.merges)

        self.special_tokens: dict[str, int] = {
            tok: first_special + i
            for i, tok in enumerate(special_tokens)
        }

        self.inverse_special_tokens: dict[int, str] = {
            i: tok
            for tok, i in self.special_tokens.items()
        }

        self._cache: dict[str, list[int]] = {}

    # ---- provided ------------------------------------------------------------------

    def __len__(self) -> int:
        """Total vocabulary size including special tokens."""
        return 256 + len(self.merges) + len(self.special_tokens)

    @property
    def eot_id(self) -> int:
        """Id of <|endoftext|>, used to separate documents (tweets) when pre-training."""
        return self.special_tokens[ENDOFTEXT]

    def save(self, path: str | Path) -> None:
        """Save merges and special tokens as JSON."""

        data = {
            "merges": [
                [a, b, idx]
                for (a, b), idx in self.merges.items()
            ],
            "special_tokens": list(self.special_tokens)
        }

        Path(path).write_text(
            json.dumps(data),
            encoding="utf-8"
        )

    @classmethod
    def load(cls, path: str | Path) -> "BPETokenizer":
        """Load a tokenizer saved with ``save``."""

        data = json.loads(
            Path(path).read_text(encoding="utf-8")
        )

        merges = {
            (a, b): idx
            for a, b, idx in data["merges"]
        }

        return cls(
            merges,
            data["special_tokens"]
        )

    def token_str(self, idx: int) -> str:
        """Human-readable form of one token (for inspecting the vocab)."""

        if idx in self.inverse_special_tokens:
            return self.inverse_special_tokens[idx]

        return self.vocab[idx].decode(
            "utf-8",
            errors="replace"
        )

    # ---- core algorithm ------------------------------------------------------------

    def _build_vocab(self) -> dict[int, bytes]:
        """Build {id: bytes} from ``self.merges``.

        Ids 0-255 map to the single byte ``bytes([i])``. Each merge (a, b) -> new
        maps to ``vocab[a] + vocab[b]``. Iterate merges in insertion order so a and b
        are always already in the vocab.
        """

        vocab: dict[int, bytes] = {
            i: bytes([i])
            for i in range(256)
        }

        for (a, b), new_id in self.merges.items():
            vocab[new_id] = vocab[a] + vocab[b]

        return vocab

    def train(self, texts: Iterable[str], vocab_size: int, verbose: bool = False) -> None:
        """Learn ``vocab_size - 256`` merges from ``texts`` (special tokens not counted).

        Works on distinct chunks weighted by their frequency: " hai" may occur 50k
        times but is counted once with weight 50k, which is what makes this fast enough.
        """
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256")

        # 1. Distinct chunks and how often each occurs, e.g. {" hai": 51234, " bhai": 20876, ...}
        chunk_counts: Counter[str] = Counter()
        for text in texts:
            chunk_counts.update(pretokenize(text))

        # 2. Each distinct chunk as its UTF-8 byte ids, paired with its count.
        chunks: list[tuple[list[int], int]] = [
            (list(chunk.encode("utf-8")), count) for chunk, count in chunk_counts.items()
        ]

        # 3. Learn merges one at a time. Vocab is kept up to date so tokens can be printed.
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        num_merges = vocab_size - 256
        for i in range(num_merges):
            stats: dict[tuple[int, int], int] = {}
            for ids, count in chunks:
                get_stats(ids, stats, weight=count)
            if not stats:  # every chunk is a single token: nothing left to merge
                break

            pair = max(stats, key=stats.get)  # most frequent pair (first seen wins ties)
            new_id = 256 + i
            chunks = [(merge(ids, pair, new_id), count) for ids, count in chunks]
            self.merges[pair] = new_id
            self.vocab[new_id] = self.vocab[pair[0]] + self.vocab[pair[1]]

            if verbose and (i % 500 == 0 or i == num_merges - 1):
                # Decode vocab bytes directly: token_str() would check the special-token ids,
                # which still point at 256 until training finishes.
                token = self.vocab[new_id].decode("utf-8", errors="replace")
                print(f"merge {i + 1:>5}/{num_merges}: {pair} -> {new_id} = {token!r}"
                      f" (seen {stats[pair]:,}x)")

        # 4. Special tokens get the ids right after the last merge.
        self.__init__(self.merges, list(self.special_tokens))

    def _encode_chunk(self, chunk: str) -> list[int]:
        """Encode one pre-tokenized chunk by applying merges earliest-learned first."""
        ids = list(chunk.encode("utf-8"))
        while len(ids) >= 2:
            # Among adjacent pairs that have a merge, take the one learned earliest (lowest id).
            mergeable = [(self.merges[pair], pair) for pair in zip(ids, ids[1:]) if pair in self.merges]
            if not mergeable:
                break
            new_id, pair = min(mergeable)
            ids = merge(ids, pair, new_id)
        return ids

    def encode(self, text: str) -> list[int]:
        """Encode text to ids: pre-tokenize, encode each chunk (memoized), concatenate.

        Special-token strings inside ``text`` are treated as normal text (safer:
        user input can't inject <|endoftext|>). Add ``self.eot_id`` yourself where needed.
        """
        ids: list[int] = []
        for chunk in pretokenize(text):
            if chunk not in self._cache:
                self._cache[chunk] = self._encode_chunk(chunk)
            ids.extend(self._cache[chunk])
        return ids

    def decode(self, ids: Iterable[int]) -> str:
        """Decode ids back to text.

        Token bytes are joined first and decoded as UTF-8 once, because one character
        (an emoji is 4 bytes) can be split across several tokens.
        """
        parts: list[bytes] = []
        for idx in ids:
            if idx in self.inverse_special_tokens:
                parts.append(self.inverse_special_tokens[idx].encode("utf-8"))
            elif idx in self.vocab:
                parts.append(self.vocab[idx])
            else:
                raise ValueError(f"unknown token id: {idx}")
        return b"".join(parts).decode("utf-8", errors="replace")
