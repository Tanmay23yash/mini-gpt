"""Decoder-only GPT language model.   [LEARN FILE: forward pass]

    token ids (B, T)
      -> token embedding (B, T, C)  +  position embedding (T, C)     # what + where
      -> dropout
      -> N x Block                                                   # attention + MLP, residual
      -> final LayerNorm                                             # "hidden" states (B, T, C)
      -> LM head: Linear(C -> vocab), weights tied to the token embedding
      -> logits (B, T, vocab)                                        # next-token scores per position

Training objective: at every position t, predict token t+1. The targets are the
inputs shifted left by one, and the loss is cross-entropy averaged over all positions.

Provided: config, layers, weight tying, initialization, parameter count.
Tests: ``pytest tests/test_gpt.py -v``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn

from minigpt.model.block import Block


@dataclass
class GPTConfig:
    """Model hyperparameters (loaded from a YAML config in training)."""

    vocab_size: int = 8193      # BPE 8192 + <|endoftext|>
    block_size: int = 128       # maximum context length T
    d_model: int = 256          # embedding size C
    n_layers: int = 4           # number of transformer blocks
    n_heads: int = 4            # attention heads per block
    dropout: float = 0.1
    bias: bool = True


class GPTOutput(NamedTuple):
    """What ``GPT.forward`` returns."""

    logits: torch.Tensor                      # (B, T, vocab_size)
    loss: torch.Tensor | None                 # scalar, only when targets are given
    hidden: torch.Tensor                      # (B, T, C) after the final LayerNorm (used by the classifier)
    attentions: list[torch.Tensor] | None     # n_layers x (B, H, T, T), only when return_attn=True


class GPT(nn.Module):
    """GPT-2 style decoder-only transformer."""

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config
        c = config
        self.tok_emb = nn.Embedding(c.vocab_size, c.d_model)   # token id -> vector ("what")
        self.pos_emb = nn.Embedding(c.block_size, c.d_model)   # position 0..T-1 -> vector ("where")
        self.drop = nn.Dropout(c.dropout)
        self.blocks = nn.ModuleList(
            [Block(c.d_model, c.n_heads, c.block_size, c.dropout, bias=c.bias) for _ in range(c.n_layers)]
        )
        self.ln_f = nn.LayerNorm(c.d_model, bias=c.bias)
        self.lm_head = nn.Linear(c.d_model, c.vocab_size, bias=False)

        # Weight tying: the output layer reuses the input embedding matrix (V, C).
        # A token's input vector and its output "score direction" are the same vector:
        # fewer parameters (2.1M saved here) and usually a slightly better model.
        self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)
        # GPT-2 trick: scale the layers that write into the residual stream by 1/sqrt(2 * n_layers),
        # so the sum of 2 * n_layers residual additions keeps a stable variance at init.
        for name, p in self.named_parameters():
            if name.endswith("attn.proj.weight") or name.endswith("mlp.proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * c.n_layers))

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """Small normal init (std 0.02) for linear and embedding weights, zero biases."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def num_params(self, non_embedding: bool = False) -> int:
        """Number of parameters (tied weights counted once). ``non_embedding`` drops position embeddings
        and the token embedding, the convention used when comparing model sizes."""
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.pos_emb.weight.numel() + self.tok_emb.weight.numel()
        return n

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None,
                return_attn: bool = False) -> GPTOutput:
        """
        Args:
            idx: (B, T) token ids, T <= block_size.
            targets: (B, T) next-token ids, or None. Positions with target -1 are ignored
                in the loss (used for padding).
            return_attn: also return every block's attention weights.

        Returns:
            GPTOutput(logits, loss, hidden, attentions)

        Steps:
            1. B, T = idx.shape; raise ValueError if T > block_size
            2. pos = torch.arange(T, device=idx.device)              -> (T,)  0, 1, ..., T-1
            3. x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))  -> (B, T, C)
               ((B, T, C) + (T, C) broadcasts: every sequence gets the same position vectors)
            4. for each block: x = block(x)    (with return_attn=True, collect the weights in a list)
            5. hidden = self.ln_f(x)                                  -> (B, T, C)
            6. logits = self.lm_head(hidden)                          -> (B, T, vocab_size)
            7. if targets is given:
                   loss = F.cross_entropy(logits.view(-1, vocab_size), targets.view(-1), ignore_index=-1)
               (cross_entropy wants (N, classes) and (N,), so flatten B and T together)
               else loss = None
            8. return GPTOutput(logits, loss, hidden, attentions or None)
        """
        # 1. Check the sequence isn't longer than the model can handle
        b, t = idx.shape
        if t > self.config.block_size:
            raise ValueError(f"sequence length {t} exceeds block_size {self.config.block_size}")

        # 2-3. What each token is + where it sits
        pos = torch.arange(t, device=idx.device)                    # (T,)
        x = self.drop(self.tok_emb(idx) + self.pos_emb(pos))        # (B, T, C)

        # 4. Run through every transformer block
        attentions: list[torch.Tensor] = []
        for block in self.blocks:
            if return_attn:
                x, w = block(x, return_attn=True)
                attentions.append(w)
            else:
                x = block(x)

        # 5-6. Final norm, then a score for every token in the vocabulary
        hidden = self.ln_f(x)                                       # (B, T, C)
        logits = self.lm_head(hidden)                               # (B, T, vocab_size)

        # 7. Loss: flatten (B, T) into one long list of predictions
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),                # (B*T, vocab)
                targets.reshape(-1),                                # (B*T,)
                ignore_index=-1,
            )

        # 8. Package everything up
        return GPTOutput(logits, loss, hidden, attentions if return_attn else None)