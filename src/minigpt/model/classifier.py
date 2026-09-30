"""Sentiment classification head on top of the GPT.

Input format (same as pre-training, where tweets were separated by <|endoftext|>):

    [<eot>] tweet tokens ... [<eot>] [pad] [pad] ...      right-padded with id 0

Because attention is causal, the hidden state at the LAST real token (the closing
<eot>) has seen the whole tweet, so it works as a summary vector (like GPT-1's
classification token). Padding sits after it and can never influence it.

Pad id 0 is the byte 0x00 (NUL), which never occurs in real text, so lengths can be
recovered from the ids: length = number of non-zero ids.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn

from minigpt.model.gpt import GPT
from minigpt.tokenizer.bpe import BPETokenizer

PAD_ID = 0


class ClassifierOutput(NamedTuple):
    logits: torch.Tensor           # (B, num_classes)
    loss: torch.Tensor | None


def encode_for_classification(tok: BPETokenizer, text: str, max_len: int) -> list[int]:
    """[<eot>] + tweet tokens (truncated) + [<eot>], never containing the pad id."""
    body = [i for i in tok.encode(text) if i != PAD_ID][: max_len - 2]
    return [tok.eot_id, *body, tok.eot_id]


class GPTClassifier(nn.Module):
    """GPT body + dropout + linear head. ``pooling``: "last" (closing <eot>) or "mean" over real tokens."""

    def __init__(self, gpt: GPT, num_classes: int, dropout: float = 0.1, pooling: str = "last") -> None:
        super().__init__()
        if pooling not in ("last", "mean"):
            raise ValueError(f"unknown pooling {pooling!r}")
        self.gpt = gpt
        self.pooling = pooling
        self.dropout = nn.Dropout(dropout)
        self.head = nn.Linear(gpt.config.d_model, num_classes)
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None) -> ClassifierOutput:
        """idx: (B, T) right-padded with PAD_ID; targets: (B,) class ids or None."""
        hidden = self.gpt(idx).hidden                                        # (B, T, C)
        real = idx != PAD_ID                                                 # (B, T)
        if self.pooling == "last":
            last = real.sum(dim=1) - 1                                       # (B,) index of closing <eot>
            feats = hidden[torch.arange(idx.size(0), device=idx.device), last]
        else:
            mask = real.unsqueeze(-1).to(hidden.dtype)
            feats = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
        logits = self.head(self.dropout(feats))
        loss = F.cross_entropy(logits, targets) if targets is not None else None
        return ClassifierOutput(logits, loss)
