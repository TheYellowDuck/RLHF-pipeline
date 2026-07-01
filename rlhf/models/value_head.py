"""Scalar value/reward head: hidden states -> per-token scalar."""

from __future__ import annotations

import torch
import torch.nn as nn


class ValueHead(nn.Module):
    """A small MLP-free linear head producing ``num_heads`` scalars per position.

    A near-zero initialization keeps initial value/reward estimates close to 0,
    which stabilizes early PPO updates and reward-model training. With ``num_heads>1``
    it emits one scalar per objective (multi-objective reward model), each an
    independent linear read-out of the same hidden state.
    """

    def __init__(self, hidden_size: int, dropout: float = 0.0, num_heads: int = 1):
        super().__init__()
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.proj = nn.Linear(hidden_size, num_heads)
        nn.init.normal_(self.proj.weight, std=1.0 / (hidden_size + 1))
        nn.init.zeros_(self.proj.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # [B, T, H] -> [B, T] (single head, unchanged) or [B, T, num_heads]
        out = self.proj(self.dropout(hidden_states))
        return out.squeeze(-1) if self.num_heads == 1 else out
