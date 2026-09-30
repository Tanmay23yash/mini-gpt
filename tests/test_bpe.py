"""Tests for the byte-level BPE tokenizer (src/minigpt/tokenizer/bpe.py)."""

from pathlib import Path

import pytest

from minigpt.tokenizer.bpe import ENDOFTEXT, BPETokenizer, get_stats, merge, pretokenize

CORPUS = [
    "bhai kya scene hai aaj match dekhne chalein?",
    "bhai match toh kal hai, aaj nahi",
    "nahi yaar aaj mood nahi hai 😭",
    "kal pakka chalenge bhai, match dekhenge 🏏",
    "yeh match bahut accha tha bhai!!",
    "nhi bhai aaj nahi, kal pakka",
] * 20

ROUND_TRIP_TEXTS = [
    "bhai kya scene hai?",
    "Kal ka match 🔥🔥 dekha kya?? #cwc19",
    "नमस्ते दोस्त, kaise ho?",
    "  multiple   spaces\nand\n\nnewlines\t",
    "unseen wordz: zxqv bhaiyaaaa",
    "<|endoftext|> is plain text here",
    "",
]


# --------------------------------------------------------------------------- provided parts


def test_pretokenize_is_lossless() -> None:
    for text in ROUND_TRIP_TEXTS:
        assert "".join(pretokenize(text)) == text


# --------------------------------------------------------------------------- helpers


def test_get_stats_counts_adjacent_pairs() -> None:
    assert get_stats([1, 2, 3, 1, 2]) == {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    assert get_stats([7]) == {}


def test_get_stats_accumulates_with_weight() -> None:
    counts = get_stats([1, 2], weight=3)
    get_stats([1, 2, 3], counts, weight=2)
    assert counts == {(1, 2): 5, (2, 3): 2}


def test_merge_replaces_all_occurrences() -> None:
    assert merge([1, 2, 3, 1, 2], (1, 2), 99) == [99, 3, 99]
    assert merge([1, 2, 3], (4, 5), 99) == [1, 2, 3]
    assert merge([1, 2], (1, 2), 99) == [99]


def test_merge_does_not_overlap() -> None:
    assert merge([5, 5, 5], (5, 5), 7) == [7, 5]
    assert merge([5, 5, 5, 5], (5, 5), 7) == [7, 7]


def test_merge_does_not_modify_input() -> None:
    ids = [1, 2, 1, 2]
    merge(ids, (1, 2), 9)
    assert ids == [1, 2, 1, 2]


# --------------------------------------------------------------------------- vocab / encode order


def test_build_vocab_from_merges() -> None:
    tok = BPETokenizer({(97, 98): 256, (256, 99): 257})  # "ab", then "ab"+"c"
    assert tok.vocab[97] == b"a"
    assert tok.vocab[256] == b"ab"
    assert tok.vocab[257] == b"abc"
    assert len(tok.vocab) == 258


def test_encode_applies_earliest_merge_first() -> None:
    """With merges ab -> 256 then bc -> 257, "abc" must become [ab, c], not [a, bc]."""
    tok = BPETokenizer({(97, 98): 256, (98, 99): 257})
    assert tok.encode("abc") == [256, 99]
    tok = BPETokenizer({(98, 99): 256, (97, 98): 257})  # same pairs, opposite order
    assert tok.encode("abc") == [97, 256]


def test_encode_without_merges_is_utf8_bytes() -> None:
    tok = BPETokenizer()
    assert tok.encode("hi 😂") == list("hi 😂".encode("utf-8"))


# --------------------------------------------------------------------------- training


@pytest.fixture(scope="module")
def trained() -> BPETokenizer:
    tok = BPETokenizer()
    tok.train(CORPUS, vocab_size=320)
    return tok


def test_train_learns_requested_number_of_merges(trained: BPETokenizer) -> None:
    assert len(trained.merges) == 320 - 256
    assert sorted(trained.merges.values()) == list(range(256, 320))  # new ids in order


def test_train_first_merge_is_most_frequent_pair() -> None:
    tok = BPETokenizer()
    tok.train(["aa aa aa b"], vocab_size=257)  # chunks: "aa", " aa", " aa", " b"
    assert tok.merges == {(97, 97): 256}      # (a, a) x3 beats (" ", a) x2


def test_train_stops_when_no_pairs_left() -> None:
    tok = BPETokenizer()
    tok.train(["ab"], vocab_size=300)  # only one pair exists -> only one merge possible
    assert tok.merges == {(97, 98): 256}


def test_train_is_deterministic() -> None:
    a, b = BPETokenizer(), BPETokenizer()
    a.train(CORPUS, vocab_size=300)
    b.train(CORPUS, vocab_size=300)
    assert a.merges == b.merges


def test_merges_do_not_cross_chunk_boundaries(trained: BPETokenizer) -> None:
    """A space may only start a token, never sit inside it (" bhai" ok, "i b" not)."""
    for idx in trained.merges.values():
        assert b" " not in trained.vocab[idx][1:], trained.vocab[idx]


def test_frequent_word_becomes_one_token(trained: BPETokenizer) -> None:
    assert len(trained.encode(" bhai")) == 1
    assert len(trained.encode(" match")) == 1


def test_special_tokens_come_after_merges(trained: BPETokenizer) -> None:
    assert trained.eot_id == 320
    assert len(trained) == 321
    assert trained.decode([trained.eot_id]) == ENDOFTEXT


def test_special_token_text_is_not_parsed(trained: BPETokenizer) -> None:
    assert trained.eot_id not in trained.encode("<|endoftext|>")


def test_compresses_training_text(trained: BPETokenizer) -> None:
    text = CORPUS[0]
    assert len(trained.encode(text)) < 0.5 * len(text.encode("utf-8"))


# --------------------------------------------------------------------------- round trip / io


@pytest.mark.parametrize("text", ROUND_TRIP_TEXTS)
def test_round_trip(trained: BPETokenizer, text: str) -> None:
    assert trained.decode(trained.encode(text)) == text


def test_decode_joins_bytes_before_utf8() -> None:
    """An emoji's 4 bytes split over several tokens must still decode to the emoji."""
    tok = BPETokenizer()
    ids = tok.encode("😂")
    assert len(ids) == 4
    assert tok.decode(ids) == "😂"


def test_encode_matches_after_save_load(trained: BPETokenizer, tmp_path: Path) -> None:
    trained.save(tmp_path / "bpe.json")
    loaded = BPETokenizer.load(tmp_path / "bpe.json")
    assert loaded.merges == trained.merges
    assert loaded.eot_id == trained.eot_id
    for text in ROUND_TRIP_TEXTS:
        assert loaded.encode(text) == trained.encode(text)
