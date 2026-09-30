"""Tests for multi-head causal self-attention (src/minigpt/model/attention.py)."""

import pytest
import torch
import torch.nn.functional as F

from minigpt.model.attention import CausalSelfAttention, scaled_dot_product_attention

B, H, T, D = 2, 4, 8, 16
C = H * D


@pytest.fixture(autouse=True)
def _seed() -> None:
    torch.manual_seed(0)


def _qkv() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return torch.randn(B, H, T, D), torch.randn(B, H, T, D), torch.randn(B, H, T, D)


# --------------------------------------------------------------------------- the core function


@pytest.mark.parametrize("causal", [True, False])
def test_sdpa_matches_pytorch(causal: bool) -> None:
    q, k, v = _qkv()
    out, _ = scaled_dot_product_attention(q, k, v, causal=causal)
    expected = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    assert out.shape == (B, H, T, D)
    torch.testing.assert_close(out, expected)


def test_sdpa_weights_are_a_distribution() -> None:
    q, k, v = _qkv()
    _, weights = scaled_dot_product_attention(q, k, v, causal=True)
    assert weights.shape == (B, H, T, T)
    assert (weights >= 0).all()
    torch.testing.assert_close(weights.sum(-1), torch.ones(B, H, T))


def test_sdpa_causal_weights_are_zero_above_diagonal() -> None:
    q, k, v = _qkv()
    _, weights = scaled_dot_product_attention(q, k, v, causal=True)
    upper = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    assert (weights[..., upper] == 0).all()
    torch.testing.assert_close(weights[..., 0, 0], torch.ones(B, H))  # token 0 only sees itself


def test_sdpa_uses_sqrt_d_scaling() -> None:
    """Scaling q by 2 must change the result exactly like PyTorch's (catches a missing /sqrt(D))."""
    q, k, v = _qkv()
    out, _ = scaled_dot_product_attention(q * 2, k, v, causal=False)
    torch.testing.assert_close(out, F.scaled_dot_product_attention(q * 2, k, v))


def test_sdpa_dropout_only_in_training() -> None:
    q, k, v = _qkv()
    eval_out, _ = scaled_dot_product_attention(q, k, v, dropout_p=0.5, training=False)
    torch.testing.assert_close(eval_out, F.scaled_dot_product_attention(q, k, v, is_causal=True))
    train_out, _ = scaled_dot_product_attention(q, k, v, dropout_p=0.5, training=True)
    assert not torch.allclose(train_out, eval_out)


# --------------------------------------------------------------------------- the module


def _layer(dropout: float = 0.0, block_size: int = 16) -> CausalSelfAttention:
    return CausalSelfAttention(d_model=C, n_heads=H, block_size=block_size, dropout=dropout).eval()


def _reference(layer: CausalSelfAttention, x: torch.Tensor) -> torch.Tensor:
    """Same computation written with PyTorch's fused attention, using the layer's own weights."""
    b, t, c = x.shape
    q, k, v = layer.qkv(x).split(c, dim=-1)
    q, k, v = (z.view(b, t, H, c // H).transpose(1, 2) for z in (q, k, v))
    out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
    return layer.proj(out.transpose(1, 2).reshape(b, t, c))


def test_layer_output_shape() -> None:
    x = torch.randn(B, T, C)
    assert _layer()(x).shape == (B, T, C)


def test_layer_matches_reference() -> None:
    layer, x = _layer(), torch.randn(B, T, C)
    torch.testing.assert_close(layer(x), _reference(layer, x))


def test_layer_has_expected_parameters() -> None:
    layer = _layer()
    assert layer.qkv.weight.shape == (3 * C, C)
    assert layer.proj.weight.shape == (C, C)
    n_params = sum(p.numel() for p in layer.parameters())
    assert n_params == 4 * C * C + 4 * C  # qkv (3C*C + 3C) + proj (C*C + C)


def test_layer_is_causal() -> None:
    """Changing future tokens must not change outputs at earlier positions."""
    layer, x = _layer(), torch.randn(B, T, C)
    x2 = x.clone()
    x2[:, 5:] = torch.randn(B, T - 5, C)
    torch.testing.assert_close(layer(x)[:, :5], layer(x2)[:, :5])
    assert not torch.allclose(layer(x)[:, 5:], layer(x2)[:, 5:])


def test_layer_returns_attention_weights() -> None:
    layer, x = _layer(), torch.randn(B, T, C)
    y, weights = layer(x, return_attn=True)
    assert y.shape == (B, T, C)
    assert weights.shape == (B, H, T, T)
    torch.testing.assert_close(weights.sum(-1), torch.ones(B, H, T))


def test_layer_accepts_shorter_and_rejects_longer_sequences() -> None:
    layer = _layer(block_size=16)
    assert layer(torch.randn(1, 3, C)).shape == (1, 3, C)
    assert layer(torch.randn(1, 16, C)).shape == (1, 16, C)
    with pytest.raises(ValueError):
        layer(torch.randn(1, 17, C))


def test_layer_rejects_indivisible_heads() -> None:
    with pytest.raises(ValueError):
        CausalSelfAttention(d_model=30, n_heads=4, block_size=8)


def test_layer_gradients_flow() -> None:
    layer, x = _layer(), torch.randn(B, T, C, requires_grad=True)
    layer(x).sum().backward()
    assert layer.qkv.weight.grad is not None and layer.qkv.weight.grad.abs().sum() > 0
    assert x.grad is not None


def test_layer_dropout_train_vs_eval() -> None:
    layer, x = _layer(dropout=0.5), torch.randn(B, T, C)
    torch.testing.assert_close(layer(x), layer(x))  # eval: deterministic
    layer.train()
    assert not torch.allclose(layer(x), layer(x))  # train: dropout is random


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_layer_runs_on_gpu() -> None:
    layer, x = _layer().cuda(), torch.randn(B, T, C, device="cuda")
    torch.testing.assert_close(layer(x), _reference(layer, x))
