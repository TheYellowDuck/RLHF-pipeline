"""Scalar value/reward head: hidden states -> per-token scalar."""

from __future__ import annotations

import torch
import torch.nn as nn


class ValueHead(nn.Module):
    """A small MLP-free linear head producing one scalar per position.

    A near-zero initialization keeps initial value/reward estimates close to 0,
    which stabilizes early PPO updates and reward-model training.
    """

    def __init__(self, hidden_size: int, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.proj = nn.Linear(hidden_size, 1)
        nn.init.normal_(self.proj.weight, std=1.0 / (hidden_size + 1))
        nn.init.zeros_(self.proj.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # hidden_states: [B, T, H] -> [B, T]
        return self.proj(self.dropout(hidden_states)).squeeze(-1)
