"""Recurrent baseline: embedding -> (Bi)LSTM/GRU -> final hidden states -> linear."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence


class RNNClassifier(nn.Module):
    """Text classifier over padded token ids.

    Uses packed sequences so padding never enters the RNN, then classifies from
    the last hidden state of each direction of the top layer.
    """

    def __init__(self, vocab_size: int, num_classes: int, embed_dim: int, hidden_dim: int,
                 num_layers: int = 1, bidirectional: bool = True, dropout: float = 0.0,
                 rnn_type: str = "lstm", pad_id: int = 0) -> None:
        super().__init__()
        rnn_cls = {"lstm": nn.LSTM, "gru": nn.GRU}[rnn_type]
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.rnn = rnn_cls(embed_dim, hidden_dim, num_layers=num_layers, batch_first=True,
                           bidirectional=bidirectional, dropout=dropout if num_layers > 1 else 0.0)
        self.dropout = nn.Dropout(dropout)
        self.num_directions = 2 if bidirectional else 1
        self.head = nn.Linear(hidden_dim * self.num_directions, num_classes)

    def forward(self, ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ids: (B, T) token ids, right-padded.
            lengths: (B,) true lengths (>= 1), on CPU.
        Returns:
            (B, num_classes) logits.
        """
        x = self.dropout(self.embedding(ids))                        # (B, T, E)
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h = self.rnn(packed)
        h_n = h[0] if isinstance(h, tuple) else h                    # LSTM returns (h, c)
        # h_n: (num_layers * dirs, B, H); take the top layer's state for each direction.
        top = h_n[-self.num_directions:]                             # (dirs, B, H)
        feats = torch.cat(list(top), dim=-1)                         # (B, dirs * H)
        return self.head(self.dropout(feats))
