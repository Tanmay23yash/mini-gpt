"""Tests for the Stage 1 tokenizer, RNN classifier and metrics."""

from pathlib import Path

import pytest
import torch

from minigpt.evaluate import compute_metrics
from minigpt.model.lstm import RNNClassifier
from minigpt.tokenizer.simple import PAD, UNK, WordTokenizer

TEXTS = ["bahut badhiya movie!", "bahut bekaar movie 😠", "movie theek thi"]


def test_tokenize_splits_words_and_symbols() -> None:
    tok = WordTokenizer([PAD, UNK])
    assert tok.tokenize("Bahut BADHIYA, don't!") == ["bahut", "badhiya", ",", "don't", "!"]


def test_build_respects_min_freq_and_specials() -> None:
    tok = WordTokenizer.build(TEXTS, min_freq=2, max_size=100)
    assert tok.itos[:2] == [PAD, UNK]
    assert set(tok.itos[2:]) == {"bahut", "movie"}  # the only tokens seen twice or more


def test_encode_unknown_truncate_and_empty() -> None:
    tok = WordTokenizer.build(TEXTS, min_freq=2, max_size=100)
    assert tok.encode("bahut accha movie") == [tok.stoi["bahut"], tok.unk_id, tok.stoi["movie"]]
    assert len(tok.encode("movie " * 10, max_len=4)) == 4
    assert tok.encode("") == [tok.unk_id]  # never an empty sequence (packing needs length >= 1)


def test_save_load_roundtrip(tmp_path: Path) -> None:
    tok = WordTokenizer.build(TEXTS, min_freq=1, max_size=100)
    tok.save(tmp_path / "vocab.json")
    loaded = WordTokenizer.load(tmp_path / "vocab.json")
    assert loaded.itos == tok.itos
    assert loaded.encode(TEXTS[0]) == tok.encode(TEXTS[0])


@pytest.mark.parametrize("rnn_type", ["lstm", "gru"])
def test_rnn_output_shape(rnn_type: str) -> None:
    model = RNNClassifier(vocab_size=50, num_classes=3, embed_dim=8, hidden_dim=16, num_layers=2,
                          rnn_type=rnn_type)
    ids = torch.randint(1, 50, (4, 7))
    lengths = torch.tensor([7, 5, 3, 1])
    assert model(ids, lengths).shape == (4, 3)


@pytest.mark.parametrize("rnn_type", ["lstm", "gru"])
def test_rnn_ignores_padding(rnn_type: str) -> None:
    """Extra padding must not change the logits - packing keeps pads out of the RNN."""
    torch.manual_seed(0)
    model = RNNClassifier(vocab_size=50, num_classes=3, embed_dim=8, hidden_dim=16, rnn_type=rnn_type).eval()
    ids = torch.tensor([[5, 6, 7]])
    padded = torch.tensor([[5, 6, 7, 0, 0, 0]])
    out_short = model(ids, torch.tensor([3]))
    out_padded = model(padded, torch.tensor([3]))
    torch.testing.assert_close(out_short, out_padded)


def test_compute_metrics() -> None:
    labels = ["negative", "neutral", "positive"]
    m = compute_metrics([0, 1, 2, 2], [0, 1, 2, 1], labels)
    assert m["accuracy"] == pytest.approx(0.75)
    assert m["f1_per_class"]["positive"] == pytest.approx(2 / 3)
    assert m["confusion_matrix"] == [[1, 0, 0], [0, 1, 0], [0, 1, 1]]
    # every prediction = majority class -> macro F1 is low even when accuracy is not
    majority = compute_metrics([1, 1, 1, 0, 2], [1] * 5, labels)
    assert majority["accuracy"] == pytest.approx(0.6)
    assert majority["macro_f1"] == pytest.approx(0.25)
