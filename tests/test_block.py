"""Tests for the MLP and the pre-norm transformer block (src/minigpt/model/block.py)."""

import pytest
import torch
import torch.nn.functional as F

from minigpt.model.block import MLP, Block

B, T, C, H = 2, 8, 32, 4


@pytest.fixture(autouse=True)
def _seed() -> None:
    torch.manual_seed(0)


# --------------------------------------------------------------------------- MLP


def test_mlp_shape_and_layers() -> None:
    mlp = MLP(C)
    assert mlp(torch.randn(B, T, C)).shape == (B, T, C)
    assert mlp.fc.weight.shape == (4 * C, C)
    assert mlp.proj.weight.shape == (C, 4 * C)


def test_mlp_matches_reference() -> None:
    mlp, x = MLP(C).eval(), torch.randn(B, T, C)
    expected = mlp.proj(F.gelu(mlp.fc(x)))
    torch.testing.assert_close(mlp(x), expected)


def test_mlp_parameter_count() -> None:
    n_params = sum(p.numel() for p in MLP(C).parameters())
    assert n_params == 8 * C * C + 5 * C  # fc (4C*C + 4C) + proj (4C*C + C)


def test_mlp_works_per_token() -> None:
    """The MLP never mixes tokens: changing token 3 only changes output 3."""
    mlp, x = MLP(C).eval(), torch.randn(B, T, C)
    x2 = x.clone()
    x2[:, 3] = torch.randn(B, C)
    diff = (mlp(x) - mlp(x2)).abs().sum(-1)  # (B, T)
    assert (diff[:, 3] > 0).all()
    assert (diff[:, [0, 1, 2, 4, 5, 6, 7]] == 0).all()


# --------------------------------------------------------------------------- Block


def _block(dropout: float = 0.0) -> Block:
    return Block(d_model=C, n_heads=H, block_size=16, dropout=dropout).eval()


def test_block_output_shape() -> None:
    assert _block()(torch.randn(B, T, C)).shape == (B, T, C)


def test_block_is_pre_norm_with_residuals() -> None:
    """Exactly x + attn(ln1(x)), then + mlp(ln2(.)). Post-norm or a missing residual fails."""
    block, x = _block(), torch.randn(B, T, C)
    with torch.no_grad():  # make the LayerNorms non-trivial so their placement matters
        for ln in (block.ln1, block.ln2):
            ln.weight.uniform_(0.5, 1.5)
            ln.bias.uniform_(-0.5, 0.5)
    h = x + block.attn(block.ln1(x))
    expected = h + block.mlp(block.ln2(h))
    torch.testing.assert_close(block(x), expected)


def test_block_is_identity_when_sublayers_output_zero() -> None:
    """With both output projections zeroed, only the residual path is left."""
    block, x = _block(), torch.randn(B, T, C)
    with torch.no_grad():
        for proj in (block.attn.proj, block.mlp.proj):
            proj.weight.zero_()
            proj.bias.zero_()
    torch.testing.assert_close(block(x), x)


def test_block_parameter_count() -> None:
    n_params = sum(p.numel() for p in _block().parameters())
    # attention 4C^2 + 4C, MLP 8C^2 + 5C, two LayerNorms 2 * 2C
    assert n_params == 12 * C * C + 13 * C


def test_block_is_causal() -> None:
    block, x = _block(), torch.randn(B, T, C)
    x2 = x.clone()
    x2[:, 5:] = torch.randn(B, T - 5, C)
    torch.testing.assert_close(block(x)[:, :5], block(x2)[:, :5])


def test_block_returns_attention_weights() -> None:
    y, weights = _block()(torch.randn(B, T, C), return_attn=True)
    assert y.shape == (B, T, C)
    assert weights.shape == (B, H, T, T)


def test_block_gradients_reach_input() -> None:
    block, x = _block(), torch.randn(B, T, C, requires_grad=True)
    block(x).sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0
    assert block.mlp.fc.weight.grad is not None


def test_block_dropout_train_vs_eval() -> None:
    block, x = _block(dropout=0.3), torch.randn(B, T, C)
    torch.testing.assert_close(block(x), block(x))
    block.train()
    assert not torch.allclose(block(x), block(x))
