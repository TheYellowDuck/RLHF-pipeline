"""Core tensor / RL math shared across trainers.

All functions are framework-light (plain torch) so the RL logic stays explicit
and auditable. Masks are 1.0 for "real" tokens and 0.0 for padding / prompt.
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F


def logprobs_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """log p(label) per position. logits [B, T, V], labels [B, T] -> [B, T]."""
    logp = F.log_softmax(logits, dim=-1)
    return torch.gather(logp, dim=-1, index=labels.unsqueeze(-1)).squeeze(-1)


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Per-position entropy of the softmax distribution. logits [B, T, V] -> [B, T]."""
    p = F.softmax(logits, dim=-1)
    return torch.logsumexp(logits, dim=-1) - torch.sum(p * logits, dim=-1)


def masked_mean(values: torch.Tensor, mask: torch.Tensor, dim: int | None = None) -> torch.Tensor:
    if dim is None:
        return (values * mask).sum() / mask.sum().clamp_min(1.0)
    return (values * mask).sum(dim) / mask.sum(dim).clamp_min(1.0)


def masked_var(values: torch.Tensor, mask: torch.Tensor, unbiased: bool = True) -> torch.Tensor:
    mean = masked_mean(values, mask)
    centered = (values - mean) * mask
    var = (centered.pow(2)).sum() / mask.sum().clamp_min(1.0)
    if unbiased:
        n = mask.sum()
        bessel = n / (n - 1).clamp_min(1.0)
        var = var * bessel
    return var


def masked_whiten(values: torch.Tensor, mask: torch.Tensor, shift_mean: bool = True) -> torch.Tensor:
    """Zero-mean, unit-variance normalization over masked entries."""
    mean = masked_mean(values, mask)
    var = masked_var(values, mask)
    whitened = (values - mean) * torch.rsqrt(var + 1e-8)
    if not shift_mean:
        whitened = whitened + mean
    return whitened * mask


@torch.no_grad()
def compute_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    mask: torch.Tensor,
    gamma: float = 1.0,
    lam: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Generalized Advantage Estimation over per-token rewards/values.

    All tensors are [B, T] over the *response* positions. `mask` marks valid
    (non-pad) response tokens. Padding is assumed right-aligned and contiguous;
    values/rewards are zeroed at pad positions so the bootstrap terminates at
    the last real token (terminal value 0).

    Returns (advantages, returns), both [B, T], zeroed at pad positions.
    """
    values = values * mask
    rewards = rewards * mask
    T = rewards.shape[1]
    advantages = torch.zeros_like(rewards)
    last_gae = torch.zeros_like(rewards[:, 0])
    for t in reversed(range(T)):
        next_values = values[:, t + 1] if t < T - 1 else torch.zeros_like(values[:, 0])
        delta = rewards[:, t] + gamma * next_values - values[:, t]
        last_gae = delta + gamma * lam * last_gae
        advantages[:, t] = last_gae
    advantages = advantages * mask
    returns = (advantages + values) * mask
    return advantages, returns


def flatten_dict(d: dict, parent: str = "", sep: str = "/") -> dict[str, Any]:
    """Flatten nested dicts for metric logging: {'a': {'b': 1}} -> {'a/b': 1}."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{parent}{sep}{k}" if parent else str(k)
        if isinstance(v, dict):
            out.update(flatten_dict(v, key, sep))
        else:
            out[key] = v
    return out
