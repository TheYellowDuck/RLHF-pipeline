"""Fast unit tests for the RLHF math (no model downloads).

Run:  ./.venv/bin/python -m pytest tests/ -q
"""

import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rlhf.algorithms.reward_trainer import aux_lm_loss, bradley_terry_loss
from rlhf.algorithms.dpo_trainer import dpo_loss
from rlhf.algorithms.ppo_trainer import AdaptiveKLController
from rlhf.models.reward_model import last_token_indices
from rlhf.utils.running import RunningMoments
from rlhf.eval import ClaudeJudge, judge_win_rate, parse_verdict
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


def test_bradley_terry_label_smoothing_and_margin():
    big, small = torch.tensor([10.0]), torch.tensor([-10.0])
    # smoothing keeps confident-correct loss away from 0 (regularizes vs noisy labels)
    assert bradley_terry_loss(big, small, label_smoothing=0.1).item() > bradley_terry_loss(big, small).item()
    eq = torch.tensor([0.0])
    assert abs(bradley_terry_loss(eq, eq, label_smoothing=0.1).item() - math.log(2)) < 1e-5
    # margin shrinks the effective gap -> higher loss for the same rewards
    c, r = torch.tensor([1.0]), torch.tensor([0.0])
    assert bradley_terry_loss(c, r, margin=1.0).item() > bradley_terry_loss(c, r).item()


def test_normalize_pku_safer_response_is_chosen():
    from rlhf.data.preference import _normalize_pku
    ex = {"prompt": "P", "response_0": "unsafe", "response_1": "safe", "safer_response_id": "1"}
    assert _normalize_pku(ex) == {"prompt": "P", "chosen": "safe", "rejected": "unsafe"}
    # no clear safer side (-1) -> emptied so the loader drops it
    assert _normalize_pku({"prompt": "P", "response_0": "a", "response_1": "b",
                           "safer_response_id": -1})["chosen"] == ""


def test_rewardbench_report_category_mean():
    from rlhf.eval import rewardbench_report
    # two Chat subsets (acc 1.0 and 0.5 -> Chat=0.75) and one Safety subset (acc 0.0 -> Safety=0.0)
    subsets = (["alpacaeval-easy"] * 2 + ["mt-bench-easy"] * 2 + ["donotanswer"] * 2)
    correct = ([1, 1] + [1, 0] + [0, 0])
    rep = rewardbench_report(subsets, correct)
    assert abs(rep["per_category"]["Chat"] - 0.75) < 1e-9       # mean(1.0, 0.5)
    assert rep["per_category"]["Safety"] == 0.0
    assert abs(rep["overall"] - 0.375) < 1e-9                   # mean(Chat 0.75, Safety 0.0)
    assert abs(rep["accuracy_micro"] - 3 / 6) < 1e-9           # 3 of 6 correct
    assert rep["n"] == 6


def test_normalize_messages_no_prompt_skywork_style():
    # Skywork-Reward style: chosen/rejected are full message lists, no separate prompt column.
    from rlhf.data.preference import _normalize_messages_no_prompt
    ex = {
        "chosen":   [{"role": "user", "content": "Q?"}, {"role": "assistant", "content": "good"}],
        "rejected": [{"role": "user", "content": "Q?"}, {"role": "assistant", "content": "bad"}],
    }
    assert _normalize_messages_no_prompt(ex) == {"prompt": "Q?", "chosen": "good", "rejected": "bad"}


def test_aux_lm_loss_uniform_logits_masks_prompt_and_padding():
    V = 5
    logits = torch.zeros(1, 4, V)                 # uniform -> per-token CE == log(V)
    input_ids = torch.tensor([[1, 2, 3, 4]])
    # response tokens at positions 1,2 (prompt=pos0, padding=pos3). After the next-token shift,
    # only predicting positions 1 and 2 should count -> mean CE == log(V).
    loss_mask = torch.tensor([[0, 1, 1, 0]])
    assert abs(aux_lm_loss(logits, input_ids, loss_mask).item() - math.log(V)) < 1e-5
    # an all-zero mask (no response tokens) is a safe 0, not a divide-by-zero
    assert aux_lm_loss(logits, input_ids, torch.zeros_like(loss_mask)).item() == 0.0


def test_preference_collator_loss_mask_is_response_only():
    from rlhf.data.preference import PreferenceCollator

    class _WSTok:                                  # whitespace tokenizer: ids don't matter, lengths do
        pad_token_id = 0
        def __call__(self, text, add_special_tokens=False):
            return {"input_ids": [i + 1 for i in range(len(text.split()))]}

    coll = PreferenceCollator(_WSTok(), max_length=512, emit_loss_mask=True)
    batch = [
        {"prompt": "the quick ", "chosen": "brown fox jumps", "rejected": "x"},
        {"prompt": "a ", "chosen": "b", "rejected": "y"},
    ]
    out = coll(batch)
    m = out["chosen_loss_mask"].tolist()
    # row 0: 2 prompt tokens then 3 response tokens -> [0,0,1,1,1]
    assert m[0] == [0, 0, 1, 1, 1]
    # row 1: 1 prompt token, 1 response token, then padding to width 5 -> response-only, pads are 0
    assert m[1] == [0, 1, 0, 0, 0]
    # default (no aux loss) collator stays byte-identical: no loss mask emitted
    assert "chosen_loss_mask" not in PreferenceCollator(_WSTok())(batch)


def test_pair_similarity():
    from rlhf.data.preference import pair_similarity
    assert pair_similarity("a b c", "a b c") == 1.0
    assert pair_similarity("a b c", "x y z") == 0.0
    assert abs(pair_similarity("a b c d", "a b x y") - 2 / 6) < 1e-9   # 2 shared / 6 union


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


def test_adaptive_kl_controller_direction():
    # KL above target -> increase the penalty coefficient; below target -> decrease.
    up = AdaptiveKLController(init_coef=0.2, target=6.0, horizon=100)
    up.update(current_kl=12.0, n_steps=10)
    assert up.value > 0.2
    down = AdaptiveKLController(init_coef=0.2, target=6.0, horizon=100)
    down.update(current_kl=1.0, n_steps=10)
    assert down.value < 0.2


def test_running_moments_converges():
    torch.manual_seed(0)
    rm = RunningMoments()
    data = torch.randn(5000) * 3.0 + 7.0  # mean 7, std 3
    for i in range(0, 5000, 100):
        rm.update(data[i:i + 100])
    assert abs(rm.mean - 7.0) < 0.3
    assert abs(rm.std - 3.0) < 0.3


def test_parse_verdict():
    assert parse_verdict("reasoning...\nVERDICT: A") == "A"
    assert parse_verdict("VERDICT: B\n") == "B"
    assert parse_verdict("I judge **VERDICT: tie**") == "tie"
    assert parse_verdict("no verdict at all") is None
    assert parse_verdict(None) is None


class _StubJudge:
    """Prefers whichever response contains the word 'good'."""
    model = "stub"

    def compare(self, conv, a, b):
        if "good" in a and "good" not in b:
            return "A"
        if "good" in b and "good" not in a:
            return "B"
        return "tie"


def test_judge_win_rate_position_controlled():
    res = judge_win_rate(_StubJudge(), ["c1", "c2"], ["good x", "good"], ["bad", "meh"],
                         swap=True, progress=False)
    assert res["win_rate"] == 1.0 and res["policy_wins"] == 2 and res["n"] == 2


def test_claude_judge_compare_with_fake_client():
    block = type("Block", (), {"type": "text", "text": "because...\nVERDICT: B"})()
    resp = type("Resp", (), {"content": [block], "stop_reason": "end_turn"})()
    fake = type("Client", (), {"messages": type("M", (), {"create": lambda self, **kw: resp})()})()
    judge = ClaudeJudge(client=fake)
    assert judge.compare("conversation", "response a", "response b") == "B"


def test_config_override_coercion():
    cfg = Config({"a": {"b": 1}, "flag": False})
    apply_overrides(cfg, ["a.b=2.5", "flag=true", "a.c=none", "name=gpt2"])
    assert cfg.a.b == 2.5 and cfg.flag is True and cfg.a.c is None and cfg.name == "gpt2"
