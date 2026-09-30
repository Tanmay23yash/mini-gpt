"""Multi-head causal self-attention.   [LEARN FILE]

The one equation
----------------
For queries Q, keys K, values V (one row per token):

    Attention(Q, K, V) = softmax( Q K^T / sqrt(d_k)  +  mask ) V

- Q K^T          (T, T) scores: how much token i "looks for" what token j "offers".
- / sqrt(d_k)    keeps the scores' scale ~1 so softmax doesn't saturate as d_k grows.
- + mask         causal mask: -inf above the diagonal, so token i can only attend to
                 tokens 0..i (a language model must not see the future).
- softmax        per row -> attention weights, each row sums to 1.
- @ V            each output row is a weighted average of value vectors.

Multi-head: split d_model into n_heads independent heads of size d_k = d_model / n_heads,
run attention in each, concatenate, and mix with an output projection.

Shape legend: B = batch, T = sequence length, C = d_model, H = n_heads, D = C // H.

Tests: ``pytest tests/test_attention.py -v``.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = True,
    dropout_p: float = 0.0,
    training: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Attention for already-split heads.

    Args:
        q, k, v: (B, H, T, D)
        causal: if True, position i may only attend to positions <= i.
        dropout_p: dropout on the attention weights (only when ``training``).
        training: whether dropout is active.

    Returns:
        out: (B, H, T, D)  weighted sum of values
        weights: (B, H, T, T)  softmax attention weights BEFORE dropout (rows sum to 1),
            returned for visualizing attention maps.

    Steps:
        1. scores = q @ k^T / sqrt(D)                       -> (B, H, T, T)
           (``k.transpose(-2, -1)`` swaps the last two dims)
        2. if causal: set scores above the diagonal to -inf
           (build a (T, T) lower-triangular bool mask with ``torch.tril(torch.ones(...))``
           on q's device, then ``scores.masked_fill(~mask, float("-inf"))``)
        3. weights = softmax(scores, dim=-1)                 -> each row sums to 1
        4. apply dropout to the weights: ``F.dropout(weights, dropout_p, training)``
        5. out = dropped_weights @ v                         -> (B, H, T, D)
    """
    t, d = q.size(-2), q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d)                         # (B, H, T, T)
    if causal:
        allowed = torch.tril(torch.ones(t, t, dtype=torch.bool, device=q.device))  # (T, T)
        scores = scores.masked_fill(~allowed, float("-inf"))
    weights = F.softmax(scores, dim=-1)                                     # rows sum to 1
    out = F.dropout(weights, dropout_p, training) @ v                       # (B, H, T, D)
    return out, weights


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention layer (GPT-2 style).

    Args:
        d_model: embedding size C.
        n_heads: number of heads H; must divide d_model.
        block_size: maximum sequence length the layer accepts.
        dropout: dropout on attention weights and on the output.
        bias: whether the linear layers have a bias.
    """

    def __init__(self, d_model: int, n_heads: int, block_size: int, dropout: float = 0.0,
                 bias: bool = True) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.n_heads = n_heads
        self.d_model = d_model
        self.block_size = block_size
        self.dropout = dropout
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=bias)  # Q, K, V for all heads in one matmul
        self.proj = nn.Linear(d_model, d_model, bias=bias)     # mixes the concatenated heads
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, return_attn: bool = False
                ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, T, C) with T <= block_size.
            return_attn: also return the attention weights (B, H, T, T).

        Returns:
            y: (B, T, C), or (y, weights) if ``return_attn``.

        Steps:
            1. check T <= block_size (raise ValueError otherwise)
            2. qkv = self.qkv(x)                              -> (B, T, 3C)
            3. split into q, k, v                             -> 3 x (B, T, C)
               (``qkv.split(self.d_model, dim=-1)``)
            4. reshape each to heads: (B, T, C) -> (B, T, H, D) -> transpose -> (B, H, T, D)
            5. out, weights = scaled_dot_product_attention(q, k, v, causal=True,
                   dropout_p=self.dropout, training=self.training)
            6. merge heads: (B, H, T, D) -> transpose -> (B, T, H, D) -> (B, T, C)
               (after transpose the tensor is not contiguous: use ``.contiguous().view(...)``
               or ``.reshape(...)``)
            7. y = self.resid_dropout(self.proj(out))
        """
        b, t, c = x.shape
        if t > self.block_size:
            raise ValueError(f"sequence length {t} exceeds block_size {self.block_size}")
        head_dim = c // self.n_heads

        q, k, v = self.qkv(x).split(self.d_model, dim=-1)                   # 3 x (B, T, C)
        q = q.view(b, t, self.n_heads, head_dim).transpose(1, 2)            # (B, H, T, D)
        k = k.view(b, t, self.n_heads, head_dim).transpose(1, 2)
        v = v.view(b, t, self.n_heads, head_dim).transpose(1, 2)

        out, weights = scaled_dot_product_attention(
            q, k, v, causal=True, dropout_p=self.dropout, training=self.training)  # (B, H, T, D)

        out = out.transpose(1, 2).reshape(b, t, c)                          # (B, T, C): heads side by side
        y = self.resid_dropout(self.proj(out))
        return (y, weights) if return_attn else y
