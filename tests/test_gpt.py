"""Tests for the GPT model (src/minigpt/model/gpt.py)."""

import math

import pytest
import torch
import torch.nn.functional as F

from minigpt.model.gpt import GPT, GPTConfig

CFG = GPTConfig(vocab_size=100, block_size=16, d_model=32, n_layers=2, n_heads=4, dropout=0.0)
B, T = 3, 10


@pytest.fixture(autouse=True)
def _seed() -> None:
    torch.manual_seed(0)


def _model(**overrides) -> GPT:
    cfg = GPTConfig(**{**CFG.__dict__, **overrides})
    return GPT(cfg).eval()


def _ids(b: int = B, t: int = T) -> torch.Tensor:
    return torch.randint(0, CFG.vocab_size, (b, t))


# --------------------------------------------------------------------------- provided parts (pass already)


def test_weights_are_tied() -> None:
    model = _model()
    assert model.lm_head.weight is model.tok_emb.weight


def test_parameter_count() -> None:
    c, V, L = CFG.d_model, CFG.vocab_size, CFG.n_layers
    expected = V * c + CFG.block_size * c + L * (12 * c * c + 13 * c) + 2 * c  # lm_head is tied: not counted
    assert _model().num_params() == expected


# --------------------------------------------------------------------------- forward


def test_forward_shapes_without_targets() -> None:
    out = _model()(_ids())
    assert out.logits.shape == (B, T, CFG.vocab_size)
    assert out.hidden.shape == (B, T, CFG.d_model)
    assert out.loss is None
    assert out.attentions is None


def test_loss_at_init_is_about_log_vocab() -> None:
    """An untrained model should be ~uniform over the vocab: loss ~ ln(V). A much higher
    value means the init is off (e.g. huge logits)."""
    ids = _ids(8, 16)
    out = _model()(ids, targets=_ids(8, 16))
    assert out.loss.shape == ()
    assert abs(out.loss.item() - math.log(CFG.vocab_size)) < 0.3


def test_loss_is_cross_entropy_of_logits() -> None:
    ids, targets = _ids(), _ids()
    out = _model()(ids, targets=targets)
    expected = F.cross_entropy(out.logits.reshape(-1, CFG.vocab_size), targets.reshape(-1))
    torch.testing.assert_close(out.loss, expected)


def test_loss_ignores_minus_one_targets() -> None:
    ids, targets = _ids(), _ids()
    targets[:, 6:] = -1  # e.g. padding
    out = _model()(ids, targets=targets)
    expected = F.cross_entropy(out.logits[:, :6].reshape(-1, CFG.vocab_size), targets[:, :6].reshape(-1))
    torch.testing.assert_close(out.loss, expected)


def test_logits_are_lm_head_of_hidden() -> None:
    model = _model()
    out = model(_ids())
    torch.testing.assert_close(out.logits, model.lm_head(out.hidden))


def test_model_is_causal() -> None:
    model, ids = _model(), _ids()
    ids2 = ids.clone()
    ids2[:, 6:] = (ids2[:, 6:] + 1) % CFG.vocab_size
    torch.testing.assert_close(model(ids).logits[:, :6], model(ids2).logits[:, :6])
    assert not torch.allclose(model(ids).logits[:, 6:], model(ids2).logits[:, 6:])


def test_position_embeddings_are_used() -> None:
    """The same token at different positions must get different hidden states."""
    ids = torch.full((1, T), 7)
    hidden = _model()(ids).hidden[0]  # (T, C)
    assert not torch.allclose(hidden[0], hidden[1])
    assert not torch.allclose(hidden[3], hidden[7])


def test_sequence_length_limits() -> None:
    model = _model()
    assert model(_ids(1, 1)).logits.shape == (1, 1, CFG.vocab_size)
    assert model(_ids(1, CFG.block_size)).logits.shape == (1, CFG.block_size, CFG.vocab_size)
    with pytest.raises(ValueError):
        model(_ids(1, CFG.block_size + 1))


def test_returns_attention_of_every_layer() -> None:
    out = _model()(_ids(), return_attn=True)
    assert len(out.attentions) == CFG.n_layers
    for w in out.attentions:
        assert w.shape == (B, CFG.n_heads, T, T)


def test_gradients_reach_embeddings() -> None:
    model = _model()
    model(_ids(), targets=_ids()).loss.backward()
    assert model.tok_emb.weight.grad is not None and model.tok_emb.weight.grad.abs().sum() > 0
    assert model.pos_emb.weight.grad is not None and model.pos_emb.weight.grad.abs().sum() > 0


def test_dropout_train_vs_eval() -> None:
    model, ids = _model(dropout=0.2), _ids()
    torch.testing.assert_close(model(ids).logits, model(ids).logits)
    model.train()
    assert not torch.allclose(model(ids).logits, model(ids).logits)


def test_can_overfit_one_batch() -> None:
    """End-to-end sanity check: the model must memorize a single small batch."""
    model = _model().train()
    ids = _ids(2, 12)
    x, y = ids[:, :-1], ids[:, 1:]  # predict the next token
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    first = model(x, targets=y).loss.item()
    for _ in range(150):
        loss = model(x, targets=y).loss
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert loss.item() < 0.1 * first, (first, loss.item())
