"""End-to-end smoke test: every stage on a tiny random GPT-2 with synthetic data.

Runs in ~seconds on CPU. Verifies the code paths, tensor alignment, checkpoint
save/load, and that all losses are finite — NOT that anything actually learns.

    python scripts/smoke_test.py
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rlhf.algorithms import DPOTrainer, GRPOTrainer, PPOTrainer, RewardTrainer, SFTTrainer
from rlhf.data import (
    preference_dataset_from_pairs,
    prompt_dataset_from_list,
    sft_dataset_from_pairs,
)
from rlhf.models import ActorCriticPolicy, RewardModel, load_causal_lm, load_tokenizer
from rlhf.utils import Config, set_seed

MODEL = "sshleifer/tiny-gpt2"
DEVICE = torch.device("cpu")  # tiny model: CPU is fastest + most stable
OUT = "checkpoints/smoke"


def synthetic():
    prompts = [f"\n\nHuman: Tell me fact number {i}.\n\nAssistant:" for i in range(24)]
    chosen = " This is a helpful, polite and correct answer."
    rejected = " bad wrong rude useless nonsense."
    pref = [(p, chosen, rejected) for p in prompts]
    sft = [(p, chosen) for p in prompts]
    return prompts, pref, sft


def banner(msg):
    print(f"\n{'='*60}\n  {msg}\n{'='*60}", flush=True)


def check_finite(model, where):
    for n, p in model.named_parameters():
        if p.requires_grad and not torch.isfinite(p).all():
            raise AssertionError(f"non-finite parameter {n} after {where}")


class _TokenCountReward(torch.nn.Module):
    """Closed-form toy reward: +1 per occurrence of a target token (for the PPO
    learning check). vocab_size=None so the trainer's tokenizer guard is skipped."""

    def __init__(self, target_id):
        super().__init__()
        self.target_id = target_id
        self.config = type("C", (), {"vocab_size": None})()

    def forward(self, input_ids, attention_mask=None):
        return (input_ids == self.target_id).float().sum(dim=1)


def ppo_learning_check(tok):
    """Prove PPO *optimizes* (deterministically): a single PPO update on a fixed
    rollout with positive advantage must INCREASE the log-prob of those tokens —
    the defining property of policy gradient. (A generation-based 'does it learn'
    test is unreliable on a 2-dim random tiny model; this tests the update directly.)"""
    banner("learning check: PPO step raises log-prob of positive-advantage tokens")
    set_seed(0)
    policy = ActorCriticPolicy.from_pretrained_lm(MODEL, dtype=torch.float32)
    ref = load_causal_lm(MODEL, dtype=torch.float32)
    cfg = Config(dict(
        output_dir=f"{OUT}/ppo_dir", data=dict(max_prompt_length=8),
        generation=dict(max_new_tokens=4),
        ppo=dict(total_episodes=4, rollout_batch_size=4, mini_batch_size=4, ppo_epochs=1,
                 gamma=1.0, lam=0.95, cliprange=0.2, cliprange_value=0.2, vf_coef=0.0, ent_coef=0.0,
                 lr=1e-2, max_grad_norm=100.0, whiten_advantages=False,
                 kl=dict(adaptive=False, init_coef=0.0, target=6.0, horizon=10))))
    trainer = PPOTrainer(policy, _TokenCountReward(0), tok, cfg, DEVICE, ref_model=ref)

    P = 8
    ptoks = (tok("the cat sat on the mat today", add_special_tokens=False)["input_ids"] * 2)[:P]
    resp = tok(" yes please", add_special_tokens=False)["input_ids"]
    full = torch.tensor([ptoks + resp] * 4, device=DEVICE)
    attn = torch.ones_like(full)
    G = len(resp)
    resp_mask = torch.ones(4, G, device=DEVICE)
    old_logp, old_vals, _ = trainer._policy_forward(full, attn, P)
    adv = torch.ones(4, G, device=DEVICE)                       # positive advantage everywhere
    batch = dict(full_ids=full, full_attn=attn, resp_mask=resp_mask, P=P,
                 old_logp=old_logp.detach(), old_values=old_vals.detach(),
                 advantages=adv, returns=(adv + old_vals).detach())

    before = trainer._policy_forward(full, attn, P)[0].detach()
    trainer._optimize(batch)
    after = trainer._policy_forward(full, attn, P)[0].detach()
    delta = (after - before).mean().item()
    print(f"  mean Δ log-prob (positive-advantage tokens) = {delta:+.4f}")
    assert delta > 0, f"PPO step did not increase log-prob of positive-advantage tokens (Δ={delta:+.4f})"


def learning_check(tok):
    """Prove the reward model *optimizes*, not just runs: it must learn a
    last-token-separable signal (chosen ends ' good', rejected ends ' bad')."""
    banner("learning check: RM separates a learnable signal")
    prompt = "\n\nHuman: Which answer is better?\n\nAssistant:"
    ds = preference_dataset_from_pairs([(prompt, " good", " bad")] * 16)
    rm = RewardModel.from_backbone(MODEL, dtype=torch.float32)
    cfg = Config(dict(output_dir=f"{OUT}/rm_learn", data=dict(max_length=32),
                      train=dict(epochs=15, batch_size=8, grad_accum=1, lr=5e-3, weight_decay=0.0,
                                 warmup_ratio=0.0, max_grad_norm=1.0, bf16=False,
                                 log_every=1000, eval_every=10000, save_every=10000)))
    trainer = RewardTrainer(rm, tok, cfg, DEVICE)
    trainer.train(ds)
    acc = trainer.evaluate(ds)["eval_accuracy"]
    assert acc > 0.9, f"reward model failed to learn a separable signal (acc={acc:.2f})"
    print(f"  RM learned: chosen>rejected accuracy {acc:.2f}")


def aux_lm_check(tok):
    """GRM aux-LM mode: (1) the reward must be numerically identical with/without the LM head
    (hidden_states[-1] == last_hidden_state), and (2) the auxiliary LM loss path must still
    learn the separable last-token signal — i.e. the regularizer doesn't break BT training."""
    banner("aux-LM (GRM) check: reward unchanged + aux loss trains")
    set_seed(0)
    trunk = RewardModel.from_backbone(MODEL, dtype=torch.float32).eval()
    grm = RewardModel.from_backbone(MODEL, dtype=torch.float32, aux_lm=True).eval()
    grm.value_head.load_state_dict(trunk.value_head.state_dict())  # same head -> compare the trunks
    enc = tok(["the quick brown fox", "hello there"], return_tensors="pt", padding=True)
    with torch.no_grad():
        r_trunk = trunk(enc["input_ids"], enc["attention_mask"])
        r_grm = grm(enc["input_ids"], enc["attention_mask"])
    delta = (r_trunk - r_grm).abs().max().item()
    assert delta < 1e-4, f"GRM reward diverged from trunk-only (max Δ={delta:.2e})"
    print(f"  reward identical with/without LM head (max Δ={delta:.2e})")

    prompt = "\n\nHuman: Which answer is better?\n\nAssistant:"
    ds = preference_dataset_from_pairs([(prompt, " good", " bad")] * 16)
    rm = RewardModel.from_backbone(MODEL, dtype=torch.float32, aux_lm=True)
    cfg = Config(dict(output_dir=f"{OUT}/rm_grm", data=dict(max_length=32),
                      train=dict(epochs=15, batch_size=8, grad_accum=1, lr=5e-3, weight_decay=0.0,
                                 warmup_ratio=0.0, max_grad_norm=1.0, bf16=False, aux_lm_coef=0.05,
                                 log_every=1000, eval_every=10000, save_every=10000)))
    trainer = RewardTrainer(rm, tok, cfg, DEVICE)
    trainer.train(ds)
    check_finite(rm, "GRM RM train")
    acc = trainer.evaluate(ds)["eval_accuracy"]
    assert acc > 0.9, f"aux-LM reward model failed to learn a separable signal (acc={acc:.2f})"
    print(f"  aux-LM RM learned: chosen>rejected accuracy {acc:.2f}")
    # the saved GRM checkpoint must reload as a trunk-only RM (what PPO/eval use downstream)
    reloaded = RewardModel.from_pretrained(f"{OUT}/rm_grm", dtype=torch.float32)
    acc2 = RewardTrainer(reloaded, tok, cfg, DEVICE).evaluate(ds)["eval_accuracy"]
    assert acc2 > 0.9, f"reloaded aux-LM RM lost its signal (acc={acc2:.2f})"
    print(f"  reload as trunk-only RM OK; eval accuracy {acc2:.2f}")


def multi_head_check(tok):
    """Multi-objective RM: two heads, each trained ONLY on its objective's pairs (via the per-pair
    'objective' index). Each head must separate its own signal; save/load must keep num_heads + weights."""
    banner("multi-head (multi-objective) RM check")
    from datasets import Dataset
    prompt = "\n\nHuman: Rate this.\n\nAssistant:"
    n = 16
    ds = Dataset.from_dict({
        "prompt": [prompt] * (2 * n),
        "chosen":   [" good"] * n + [" safe"] * n,      # obj 0: good>bad   |  obj 1: safe>unsafe
        "rejected": [" bad"] * n + [" unsafe"] * n,
        "objective": [0] * n + [1] * n,
    })
    rm = RewardModel.from_backbone(MODEL, dtype=torch.float32, num_heads=2)
    cfg = Config(dict(output_dir=f"{OUT}/rm_mh", data=dict(max_length=32),
                      train=dict(epochs=15, batch_size=8, grad_accum=1, lr=5e-3, weight_decay=0.0,
                                 warmup_ratio=0.0, max_grad_norm=1.0, bf16=False, num_heads=2,
                                 log_every=1000, eval_every=10000, save_every=10000)))
    RewardTrainer(rm, tok, cfg, DEVICE).train(ds)
    check_finite(rm, "multi-head RM")
    rm.eval()

    def heads(chosen, rejected):
        ec = tok([prompt + chosen], return_tensors="pt", padding=True)
        er = tok([prompt + rejected], return_tensors="pt", padding=True)
        with torch.no_grad():
            hc = rm(ec["input_ids"], ec["attention_mask"], return_heads=True)[0]      # [2]
            hr = rm(er["input_ids"], er["attention_mask"], return_heads=True)[0]
        return hc, hr

    c0, r0 = heads(" good", " bad")        # objective 0
    c1, r1 = heads(" safe", " unsafe")     # objective 1
    assert c0[0] > r0[0], "head 0 did not learn its objective (good>bad)"
    assert c1[1] > r1[1], "head 1 did not learn its objective (safe>unsafe)"
    combined = rm(tok([prompt + " good"], return_tensors="pt", padding=True)["input_ids"],
                  tok([prompt + " good"], return_tensors="pt", padding=True)["attention_mask"])
    assert combined.shape == (1,), f"combined reward should be scalar per row, got {tuple(combined.shape)}"
    print(f"  head0 good-bad {(c0[0]-r0[0]).item():+.2f}; head1 safe-unsafe {(c1[1]-r1[1]).item():+.2f}; "
          f"combined scalar OK")

    rm.set_head_weights([1.0, 0.0])
    rm.save_pretrained(f"{OUT}/rm_mh_w", merge=False)
    reloaded = RewardModel.from_pretrained(f"{OUT}/rm_mh_w", dtype=torch.float32)
    assert reloaded.num_heads == 2 and reloaded.head_weights.tolist() == [1.0, 0.0]
    print("  save/load OK; num_heads=2 + head_weights=[1,0] preserved")


def main():
    set_seed(0)
    prompts, pref, sft = synthetic()
    pref_ds = preference_dataset_from_pairs(pref)
    sft_ds = sft_dataset_from_pairs(sft)
    prompt_ds = prompt_dataset_from_list(prompts)
    tok = load_tokenizer(MODEL)

    train_cfg = dict(epochs=1, batch_size=4, grad_accum=1, lr=1e-4, weight_decay=0.0,
                     warmup_ratio=0.0, max_grad_norm=1.0, bf16=False,
                     gradient_checkpointing=True,  # exercise the memory-saving path
                     log_every=2, eval_every=1000, save_every=1000)

    # ---- 1. Reward model -------------------------------------------------
    banner("1/5 Reward model (Bradley-Terry)")
    rm = RewardModel.from_backbone(MODEL, dtype=torch.float32)
    rm_cfg = Config(dict(output_dir=f"{OUT}/rm", data=dict(max_length=48), train=dict(train_cfg)))
    RewardTrainer(rm, tok, rm_cfg, DEVICE).train(pref_ds, eval_ds=pref_ds)
    check_finite(rm, "RM train")
    rm_loaded = RewardModel.from_pretrained(f"{OUT}/rm", dtype=torch.float32)
    print("  RM reload OK; eval:", RewardTrainer(rm_loaded, tok, rm_cfg, DEVICE).evaluate(pref_ds))

    # ---- 2. SFT ----------------------------------------------------------
    banner("2/5 SFT")
    sft_model = load_causal_lm(MODEL, dtype=torch.float32)
    sft_cfg = Config(dict(output_dir=f"{OUT}/sft",
                          data=dict(max_length=48, mask_prompt=True), train=dict(train_cfg)))
    SFTTrainer(sft_model, tok, sft_cfg, DEVICE).train(sft_ds, eval_ds=sft_ds)
    check_finite(sft_model, "SFT train")

    # ---- 3. PPO ----------------------------------------------------------
    banner("3/5 PPO (RM + value head + GAE + adaptive KL)")
    policy = ActorCriticPolicy.from_pretrained_lm(MODEL, dtype=torch.float32)
    ref = load_causal_lm(MODEL, dtype=torch.float32)
    reward_model = RewardModel.from_pretrained(f"{OUT}/rm", dtype=torch.float32)
    ppo_cfg = Config(dict(
        output_dir=f"{OUT}/ppo",
        data=dict(max_prompt_length=24),
        generation=dict(max_new_tokens=8, temperature=1.0, top_k=0, top_p=1.0),
        ppo=dict(total_episodes=16, rollout_batch_size=8, mini_batch_size=4, ppo_epochs=2,
                 gamma=1.0, lam=0.95, cliprange=0.2, cliprange_value=0.2, vf_coef=0.1,
                 ent_coef=0.01, lr=1e-4, max_grad_norm=1.0, whiten_advantages=True,
                 whiten_rewards=False, normalize_rewards=True, length_penalty=0.01,
                 missing_eos_penalty=1.0, log_every=1, save_every=1000,
                 kl=dict(adaptive=True, init_coef=0.2, target=6.0, horizon=10000))))
    PPOTrainer(policy, reward_model, tok, ppo_cfg, DEVICE, ref_model=ref).train(prompt_ds)
    check_finite(policy, "PPO train")

    # resume round-trip: reload policy + restore optimizer/step/KL state
    resumed = ActorCriticPolicy.from_pretrained(f"{OUT}/ppo", dtype=torch.float32)
    rt = PPOTrainer(resumed, RewardModel.from_pretrained(f"{OUT}/rm", dtype=torch.float32),
                    tok, ppo_cfg, DEVICE, ref_model=load_causal_lm(MODEL, dtype=torch.float32))
    rt.load_trainer_state(f"{OUT}/ppo")
    assert rt.global_step == 2, f"PPO resume failed (step={rt.global_step})"
    print(f"  PPO resume OK (global_step restored = {rt.global_step})")

    # ---- 4. DPO ----------------------------------------------------------
    banner("4/5 DPO")
    dpo_model = load_causal_lm(MODEL, dtype=torch.float32)
    dpo_ref = load_causal_lm(MODEL, dtype=torch.float32)
    dpo_cfg = Config(dict(output_dir=f"{OUT}/dpo",
                          data=dict(max_length=48, max_prompt_length=24),
                          dpo=dict(beta=0.1, loss_type="sigmoid", label_smoothing=0.0,
                                   length_normalize=True),
                          train=dict(train_cfg)))
    DPOTrainer(dpo_model, dpo_ref, tok, dpo_cfg, DEVICE).train(pref_ds, eval_ds=pref_ds)
    check_finite(dpo_model, "DPO train")

    # ---- 5. GRPO ---------------------------------------------------------
    banner("5/5 GRPO (group-relative, critic-free)")
    grpo_policy = ActorCriticPolicy.from_pretrained_lm(MODEL, dtype=torch.float32)
    grpo_ref = load_causal_lm(MODEL, dtype=torch.float32)
    grpo_rm = RewardModel.from_pretrained(f"{OUT}/rm", dtype=torch.float32)
    grpo_cfg = Config(dict(
        output_dir=f"{OUT}/grpo",
        data=dict(max_prompt_length=24),
        generation=dict(max_new_tokens=8, temperature=1.0, top_k=0, top_p=1.0),
        grpo=dict(total_episodes=32, prompts_per_step=4, group_size=4, mini_batch_size=8,
                  grpo_epochs=1, cliprange=0.2, lr=1e-4, max_grad_norm=1.0, kl_coef=0.04,
                  scale_rewards=True, log_every=1, save_every=1000)))
    GRPOTrainer(grpo_policy, grpo_rm, tok, grpo_cfg, DEVICE, ref_model=grpo_ref).train(prompt_ds)
    check_finite(grpo_policy, "GRPO train")

    learning_check(tok)
    ppo_learning_check(tok)
    aux_lm_check(tok)
    multi_head_check(tok)

    # ---- bonus: accelerate path (single-process CPU) ---------------------
    banner("bonus: accelerate path (single-process)")
    from accelerate import Accelerator

    acc = Accelerator(cpu=True)
    acc_model = load_causal_lm(MODEL, dtype=torch.float32)
    acc_cfg = Config(dict(output_dir=f"{OUT}/acc_sft",
                          data=dict(max_length=48, mask_prompt=True), train=dict(train_cfg)))
    SFTTrainer(acc_model, tok, acc_cfg, acc.device, accelerator=acc).train(sft_ds)
    print("  accelerate single-process SFT OK")

    banner("ALL STAGES PASSED ✅")


if __name__ == "__main__":
    main()
