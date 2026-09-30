"""Byte-level Byte Pair Encoding (BPE) tokenizer, GPT-2 style.   [LEARN FILE]

Idea
----
Start with the 256 possible byte values as the base vocabulary (so any text, emoji
or script can be encoded and nothing is ever <unk>). Then repeat:

    1. count every adjacent pair of token ids in the training text
    2. take the most frequent pair (a, b)
    3. give it a new id  new = 256 + (number of merges so far)
    4. replace every (a, b) with new, and record  merges[(a, b)] = new

After N merges the vocabulary has 256 + N tokens, and ``vocab[new] = vocab[a] + vocab[b]``
(bytes concatenation). Frequent words like "bhai" end up as one token; rare words
are split into a few pieces ("bhaiyaaa" -> "bhai" + "ya" + "aa"), so spelling
variants share most of their tokens - the thing Stage 1's word vocab could not do.

Encoding new text applies the learned merges **in the order they were learned**
(lowest new id first), because later merges were learned on top of earlier ones.

Text is first split into chunks by a regex (``pretokenize``) and merges never
cross chunk boundaries, so a token never glues the end of one word to the next.

What you implement: every function/method marked ``TODO``.
Run ``pytest tests/test_bpe.py -v`` to check your work.
"""

from __future__ import annotations

import json
import re
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


# --------------------------------------------------------------------------- helpers (TODO)


def get_stats(ids: list[int], counts: dict[tuple[int, int], int] | None = None,
              weight: int = 1) -> dict[tuple[int, int], int]:
    """Count adjacent pairs in ``ids``, adding into ``counts`` if given.

    Args:
        ids: a sequence of token ids, e.g. [1, 2, 3, 1, 2].
        counts: an existing dict to update in place (lets you accumulate over many chunks).
        weight: how much each occurrence counts. Useful when ``ids`` is a chunk that
            appears ``weight`` times in the corpus - you count it once, not N times.

    Returns:
        The counts dict, e.g. {(1, 2): 2, (2, 3): 1, (3, 1): 1}.
    """
    # TODO: implement
    raise NotImplementedError


def merge(ids: list[int], pair: tuple[int, int], new_id: int) -> list[int]:
    """Return a new list where every occurrence of ``pair`` is replaced by ``new_id``.

    Scan left to right; occurrences do not overlap:
        merge([1, 2, 3, 1, 2], (1, 2), 99) -> [99, 3, 99]
        merge([5, 5, 5], (5, 5), 7)        -> [7, 5]
    """
    # TODO: implement
    raise NotImplementedError


# --------------------------------------------------------------------------- tokenizer


class BPETokenizer:
    """Byte-level BPE tokenizer.

    Attributes:
        merges: {(id_a, id_b): new_id}, in the order they were learned (dicts keep order).
        vocab: {id: bytes} for the 256 byte tokens and every merged token.
        special_tokens: {"<|endoftext|>": id, ...}; ids come right after the last merge.
    """

    def __init__(self, merges: dict[tuple[int, int], int] | None = None,
                 special_tokens: Iterable[str] = (ENDOFTEXT,)) -> None:
        self.merges: dict[tuple[int, int], int] = dict(merges or {})
        self.vocab: dict[int, bytes] = self._build_vocab()
        first_special = 256 + len(self.merges)
        self.special_tokens: dict[str, int] = {tok: first_special + i for i, tok in enumerate(special_tokens)}
        self.inverse_special_tokens: dict[int, str] = {i: tok for tok, i in self.special_tokens.items()}
        self._cache: dict[str, list[int]] = {}  # optional: chunk -> ids, speeds up encode()

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
        data = {"merges": [[a, b, idx] for (a, b), idx in self.merges.items()],
                "special_tokens": list(self.special_tokens)}
        Path(path).write_text(json.dumps(data), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> BPETokenizer:
        """Load a tokenizer saved with ``save``."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        merges = {(a, b): idx for a, b, idx in data["merges"]}
        return cls(merges, data["special_tokens"])

    def token_str(self, idx: int) -> str:
        """Human-readable form of one token (for inspecting the vocab)."""
        if idx in self.inverse_special_tokens:
            return self.inverse_special_tokens[idx]
        return self.vocab[idx].decode("utf-8", errors="replace")

    # ---- TODO ----------------------------------------------------------------------

    def _build_vocab(self) -> dict[int, bytes]:
        """Build {id: bytes} from ``self.merges``.

        Ids 0-255 map to the single byte ``bytes([i])``. Each merge (a, b) -> new
        maps to ``vocab[a] + vocab[b]``. Iterate merges in insertion order so a and b
        are always already in the vocab.
        """
        # TODO: implement
        raise NotImplementedError

    def train(self, texts: Iterable[str], vocab_size: int, verbose: bool = False) -> None:
        """Learn ``vocab_size - 256`` merges from ``texts`` (special tokens not counted).

        Steps:
            1. Pre-tokenize every text into chunks and count how often each distinct
               chunk occurs (a Counter of strings). Many chunks repeat (" hai", " ki"),
               so working on distinct chunks with weights is much faster.
            2. Turn each distinct chunk into its list of UTF-8 byte values.
            3. For each merge i in range(vocab_size - 256):
                 - count pairs over all chunks (use ``get_stats`` with ``weight``)
                 - stop early if there are no pairs left
                 - pick the most frequent pair, new_id = 256 + i
                 - apply ``merge`` to every chunk
                 - record it in ``self.merges``
            4. Rebuild ``self.vocab`` and the special-token ids (they must come
               after the last merge) - re-running ``self.__init__(self.merges, ...)``
               with the current special tokens is the easy way.

        Tip: ``print`` the new token every few hundred merges when ``verbose`` -
        watching " bhai", " nahi", " hai" appear is the best check that it works.
        """
        # TODO: implement
        raise NotImplementedError

    def _encode_chunk(self, chunk: str) -> list[int]:
        """Encode one pre-tokenized chunk.

        Start from its UTF-8 bytes. While at least 2 ids remain:
            - among the adjacent pairs, find the one with the LOWEST merge id
              (the earliest learned merge); pairs not in ``self.merges`` don't count
            - if none of the pairs can be merged, stop
            - otherwise merge that pair everywhere in the chunk
        """
        # TODO: implement
        raise NotImplementedError

    def encode(self, text: str) -> list[int]:
        """Encode text to ids: pre-tokenize, encode each chunk, concatenate.

        Special-token strings inside ``text`` are treated as normal text (safer:
        user input can't inject <|endoftext|>). Add ``self.eot_id`` yourself where needed.
        Optional speed-up: memoize ``_encode_chunk`` results in ``self._cache``.
        """
        # TODO: implement
        raise NotImplementedError

    def decode(self, ids: Iterable[int]) -> str:
        """Decode ids back to text.

        Concatenate the BYTES of all tokens first, then decode UTF-8 once with
        ``errors="replace"``. (Decoding token by token breaks characters whose bytes
        are split across tokens, e.g. an emoji is 4 bytes.) Special-token ids map to
        their string, e.g. "<|endoftext|>".
        """
        # TODO: implement
        raise NotImplementedError
