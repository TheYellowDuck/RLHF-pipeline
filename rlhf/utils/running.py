"""Online running mean/variance (Welford) for reward normalization in RL."""

from __future__ import annotations

import torch


class RunningMoments:
    """Numerically-stable running mean/std over a stream of (batched) scalars.

    Used to standardize reward-model scores across PPO iterations, which keeps the
    advantage scale stable as the policy drifts.
    """

    def __init__(self):
        self.count = 1e-4
        self.mean = 0.0
        self.var = 1.0

    @property
    def std(self) -> float:
        return float(self.var) ** 0.5

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> tuple[float, float]:
        """Fold a batch of values in; return this batch's (mean, std)."""
        x = x.detach().float().reshape(-1)
        batch_count = x.numel()
        batch_mean = x.mean().item()
        batch_var = x.var(unbiased=False).item() if batch_count > 1 else 0.0

        delta = batch_mean - self.mean
        tot = self.count + batch_count
        new_mean = self.mean + delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta**2 * self.count * batch_count / tot

        self.mean = new_mean
        self.var = m2 / tot
        self.count = tot
        batch_std = batch_var**0.5 if batch_count > 1 else 1.0
        return batch_mean, batch_std
