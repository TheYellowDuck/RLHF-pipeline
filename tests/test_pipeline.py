"""Fast unit tests for the RLHF math (no model downloads).

Run:  ./.venv/bin/python -m pytest tests/ -q
"""

import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rlhf.algorithms.reward_trainer import bradley_terry_loss
from rlhf.algorithms.dpo_trainer import dpo_loss
from rlhf.models.reward_model import last_token_indices
from rlhf.utils import (
    Config,
    apply_overrides,
    compute_gae,
    entropy_from_logits,
    logprobs_from_logits,
    masked_mean,
    masked_whiten,
    response_mask,
)


def test_logprobs_match_manual():
    torch.manual_seed(0)
    logits = torch.randn(2, 4, 7)
    labels = torch.randint(0, 7, (2, 4))
    got = logprobs_from_logits(logits, labels)
    want = F.log_softmax(logits, dim=-1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    assert torch.allclose(got, want, atol=1e-6)


def test_entropy_uniform_equals_log_vocab():
    V = 50
    logits = torch.zeros(3, 2, V)  # uniform distribution
    ent = entropy_from_logits(logits)
    assert torch.allclose(ent, torch.full_like(ent, math.log(V)), atol=1e-5)


def test_masked_mean_ignores_padding():
    vals = torch.tensor([[1.0, 2.0, 999.0], [3.0, 4.0, 5.0]])
    mask = torch.tensor([[1.0, 1.0, 0.0], [1.0, 1.0, 1.0]])
    assert abs(masked_mean(vals, mask).item() - (1 + 2 + 3 + 4 + 5) / 5) < 1e-6


def test_masked_whiten_zero_mean_unit_var():
    torch.manual_seed(1)
    vals = torch.randn(4, 10) * 3 + 5
    mask = torch.ones(4, 10)
    w = masked_whiten(vals, mask)
    assert abs(masked_mean(w, mask).item()) < 1e-5
    var = masked_mean((w - masked_mean(w, mask)) ** 2, mask).item()
    assert abs(var - 1.0) < 0.2  # unbiased estimator, finite sample


def test_gae_matches_hand_computation():
    # gamma=1, lam=0.95, zero values, reward only at t=2 -> discounted credit back
    rewards = torch.tensor([[0.0, 0.0, 2.0]])
    values = torch.zeros(1, 3)
    mask = torch.ones(1, 3)
    adv, ret = compute_gae(rewards, values, mask, gamma=1.0, lam=0.95)
    expected = torch.tensor([[0.95 * 0.95 * 2, 0.95 * 2, 2.0]])
    assert torch.allclose(adv, expected, atol=1e-5)
    assert torch.allclose(ret, adv, atol=1e-5)  # values are zero


def test_gae_masks_padding_and_bootstraps_zero():
    rewards = torch.tensor([[0.0, 1.0, 0.0]])  # pad at t=2
    values = torch.tensor([[0.5, 0.5, 0.5]])
    mask = torch.tensor([[1.0, 1.0, 0.0]])
    adv, ret = compute_gae(rewards, values, mask, gamma=1.0, lam=1.0)
    assert adv[0, 2].item() == 0.0  # padded position contributes nothing
    # last real token (t=1): delta = r + gamma*next(=0) - v = 1 - 0.5 = 0.5
    assert abs(adv[0, 1].item() - 0.5) < 1e-6


def test_bradley_terry_loss_limits():
    big = torch.tensor([10.0]); small = torch.tensor([-10.0])
    assert bradley_terry_loss(big, small).item() < 1e-3          # confident + correct
    eq = torch.tensor([0.0])
    assert abs(bradley_terry_loss(eq, eq).item() - math.log(2)) < 1e-5


def test_dpo_loss_equal_policy_is_log2():
    z = torch.zeros(4)
    loss, c, r = dpo_loss(z, z, z, z, beta=0.1)
    assert abs(loss.item() - math.log(2)) < 1e-5
    assert torch.allclose(c, r)


def test_dpo_loss_prefers_chosen():
    # policy increases chosen logprob relative to ref -> lower loss than the tie
    pol_c = torch.tensor([1.0]); pol_r = torch.tensor([-1.0])
    ref = torch.zeros(1)
    loss, _, _ = dpo_loss(pol_c, pol_r, ref, ref, beta=0.5)
    assert loss.item() < math.log(2)


def test_response_mask_stops_at_first_eos():
    eos = 0
    responses = torch.tensor([[5, 6, 0, 0, 9], [5, 6, 7, 8, 9]])
    m = response_mask(responses, eos)
    # row 0: keep through first eos at idx2 -> [1,1,1,0,0]
    assert m[0].tolist() == [1, 1, 1, 0, 0]
    # row 1: no eos -> all ones
    assert m[1].tolist() == [1, 1, 1, 1, 1]


def test_last_token_indices_left_and_right_padding():
    # right-pad only: last real token is at sum-1
    assert last_token_indices(torch.tensor([[1, 1, 1, 0]]))[0].item() == 2
    # left-pad prompt + right-pad response (the PPO/GRPO case): last real == idx 4
    assert last_token_indices(torch.tensor([[0, 0, 1, 1, 1, 0]]))[0].item() == 4
    # batch with differing real lengths
    idx = last_token_indices(torch.tensor([[0, 1, 1, 0, 0], [0, 0, 0, 1, 1]]))
    assert idx.tolist() == [2, 4]


def test_config_override_coercion():
    cfg = Config({"a": {"b": 1}, "flag": False})
    apply_overrides(cfg, ["a.b=2.5", "flag=true", "a.c=none", "name=gpt2"])
    assert cfg.a.b == 2.5 and cfg.flag is True and cfg.a.c is None and cfg.name == "gpt2"
