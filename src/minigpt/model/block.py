"""Transformer block (pre-norm, GPT-2 style).   [LEARN FILE]

A block = "communicate, then think":

    x = x + Attention(LayerNorm(x))     # tokens exchange information (causal)
    x = x + MLP(LayerNorm(x))           # each token processes what it gathered, on its own

Two ideas to understand:

- Residual connection (``x = x + f(x)``): each sub-layer only adds a *change* to x.
  Gradients flow straight through the ``+``, so deep stacks of blocks still train.
  If f outputs zeros, the block is the identity - a safe starting point.
- Pre-norm: LayerNorm is applied to the *input of* each sub-layer, not after the add.
  The residual stream x itself is never normalized inside the block, which makes
  training more stable than the original (post-norm) Transformer.

The MLP expands each token vector 4x, applies GELU (a smooth ReLU), and projects back:

    MLP(x) = Dropout( W2 · GELU( W1 · x ) )      W1: C -> 4C,  W2: 4C -> C

Shape legend: B = batch, T = sequence length, C = d_model, H = n_heads.

Tests: ``pytest tests/test_block.py -v``.
"""

from __future__ import annotations

import torch
from torch import nn

from minigpt.model.attention import CausalSelfAttention


class MLP(nn.Module):
    """Position-wise feed-forward network: C -> mlp_ratio*C -> GELU -> C -> dropout."""

    def __init__(self, d_model: int, mlp_ratio: int = 4, dropout: float = 0.0, bias: bool = True) -> None:
        super().__init__()
        self.fc = nn.Linear(d_model, mlp_ratio * d_model, bias=bias)    # expand: C -> 4C
        self.act = nn.GELU()
        self.proj = nn.Linear(mlp_ratio * d_model, d_model, bias=bias)  # shrink back: 4C -> C
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, C) -> (B, T, C): fc -> act -> proj -> dropout."""
        return self.dropout(self.proj(self.act(self.fc(x))))


class Block(nn.Module):
    """One pre-norm transformer block: attention sub-layer + MLP sub-layer, each residual.

    Args:
        d_model: embedding size C.
        n_heads: number of attention heads.
        block_size: maximum sequence length.
        dropout: dropout inside attention and the MLP.
        mlp_ratio: MLP hidden size = mlp_ratio * d_model.
        bias: whether linear layers and LayerNorms have a bias.
    """

    def __init__(self, d_model: int, n_heads: int, block_size: int, dropout: float = 0.0,
                 mlp_ratio: int = 4, bias: bool = True) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model, bias=bias)                                   # before attention
        self.attn = CausalSelfAttention(d_model, n_heads, block_size, dropout, bias)
        self.ln2 = nn.LayerNorm(d_model, bias=bias)                                   # before the MLP
        self.mlp = MLP(d_model, mlp_ratio, dropout, bias)

    def forward(self, x: torch.Tensor, return_attn: bool = False
                ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, C)
            return_attn: also return the attention weights (B, H, T, T) of this block.

        Returns:
            (B, T, C), or (output, weights) if ``return_attn``.

        Steps:
            1. a, weights = self.attn(self.ln1(x), return_attn=True)
            2. x = x + a                       # residual 1
            3. x = x + self.mlp(self.ln2(x))   # residual 2 (note: ln2 of the NEW x)
        """
        a, weights = self.attn(self.ln1(x), return_attn=True)  # communicate
        x = x + a                                               # residual 1
        x = x + self.mlp(self.ln2(x))                           # think, residual 2
        return (x, weights) if return_attn else x