"""Tests for the GPT sentiment classifier (src/minigpt/model/classifier.py)."""

import pytest
import torch

from minigpt.model.classifier import PAD_ID, GPTClassifier, encode_for_classification
from minigpt.model.gpt import GPT, GPTConfig
from minigpt.tokenizer.bpe import BPETokenizer
from minigpt.train import make_amp, train_step

EOT = 40


def _clf(pooling: str = "last") -> GPTClassifier:
    torch.manual_seed(0)
    gpt = GPT(GPTConfig(vocab_size=41, block_size=16, d_model=32, n_layers=2, n_heads=2, dropout=0.0))
    return GPTClassifier(gpt, num_classes=3, dropout=0.0, pooling=pooling).eval()


def _batch() -> torch.Tensor:
    return torch.tensor([[EOT, 5, 6, 7, EOT, PAD_ID, PAD_ID],
                         [EOT, 9, 8, 7, 6, 5, EOT]])


def test_encode_wraps_in_eot_and_truncates() -> None:
    tok = BPETokenizer()  # byte-level only is enough here
    ids = encode_for_classification(tok, "bhai kya scene hai", max_len=8)
    assert ids[0] == ids[-1] == tok.eot_id
    assert len(ids) == 8
    assert PAD_ID not in ids
    assert PAD_ID not in encode_for_classification(tok, "a\x00b", max_len=10)  # NUL can't masquerade as pad


@pytest.mark.parametrize("pooling", ["last", "mean"])
def test_output_shape_and_loss(pooling: str) -> None:
    out = _clf(pooling)(_batch(), targets=torch.tensor([0, 2]))
    assert out.logits.shape == (2, 3)
    assert out.loss.shape == ()


@pytest.mark.parametrize("pooling", ["last", "mean"])
def test_padding_does_not_change_predictions(pooling: str) -> None:
    model = _clf(pooling)
    short = torch.tensor([[EOT, 5, 6, 7, EOT]])
    padded = torch.tensor([[EOT, 5, 6, 7, EOT, PAD_ID, PAD_ID, PAD_ID]])
    torch.testing.assert_close(model(short).logits, model(padded).logits)


def test_last_pooling_reads_the_closing_token() -> None:
    model = _clf("last")
    x = _batch()
    feats = model.gpt(x).hidden[[0, 1], [4, 6]]                           # positions of the closing <eot>
    torch.testing.assert_close(model(x).logits, model.head(feats))


def test_every_real_token_matters() -> None:
    model = _clf("last")
    x, x2 = _batch(), _batch().clone()
    x2[0, 1] = 11                                                          # change the first word of tweet 0
    assert not torch.allclose(model(x).logits[0], model(x2).logits[0])
    torch.testing.assert_close(model(x).logits[1], model(x2).logits[1])   # tweet 1 untouched


def test_train_step_from_pretraining_fine_tunes_the_classifier() -> None:
    model = _clf().train()
    x, y = _batch(), torch.tensor([0, 2])
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    ctx, scaler = make_amp(torch.device("cpu"), False)
    first, _ = train_step(model, x, y, opt, scaler, ctx, 1.0)
    for _ in range(30):
        loss, _ = train_step(model, x, y, opt, scaler, ctx, 1.0)
    assert loss < 0.1 * first


def test_rejects_unknown_pooling() -> None:
    with pytest.raises(ValueError):
        GPTClassifier(_clf().gpt, 3, pooling="max")
