"""PPO for language models, implemented from scratch.

Per iteration:
  1. sample prompts -> generate responses with the current policy
  2. score full sequences with the reward model
  3. shape per-token rewards: a KL-to-reference penalty at every response token,
     plus the (scalar) reward-model score added at the final response token
  4. compute per-token advantages/returns with GAE using the value head
  5. several epochs of the clipped PPO surrogate + clipped value loss + entropy
  6. update an adaptive KL coefficient toward a target KL

Token alignment (P = prompt length, T = P + G full length):
  the log-prob / value of response token at full position j comes from the model
  output at position j-1, so the response slice is [P-1 : T-1] of the shifted
  arrays — length G, matching the generated tokens.
"""

from __future__ import annotations

import os

import torch
from torch.utils.data import DataLoader
from transformers import GenerationConfig

from ..data import PromptCollator
from ..utils.common import disable_dropout, get_logger
from ..utils.running import RunningMoments
from ..utils.tensor_ops import (
    compute_gae,
    entropy_from_logits,
    logprobs_from_logits,
    masked_mean,
    masked_whiten,
    response_mask,
)
from .common import save_tokenizer


class FixedKLController:
    def __init__(self, coef: float):
        self.value = coef

    def update(self, current_kl: float, n_steps: int = 1):
        pass


class AdaptiveKLController:
    """Adjusts the KL coefficient toward a target KL (Schulman et al. heuristic)."""

    def __init__(self, init_coef: float, target: float, horizon: float):
        self.value = init_coef
        self.target = target
        self.horizon = horizon

    def update(self, current_kl: float, n_steps: int = 1):
        proportional_error = max(-0.2, min(0.2, current_kl / self.target - 1.0))
        self.value *= 1.0 + proportional_error * n_steps / self.horizon


class PPOTrainer:
    def __init__(self, policy, reward_model, tokenizer, cfg, device,
                 ref_model=None, metric_logger=None):
        self.policy = policy.to(device)
        self.reward_model = reward_model.to(device).eval()
        self.ref_model = ref_model.to(device).eval() if ref_model is not None else None
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.device = device
        self.metrics = metric_logger
        self.log = get_logger("rlhf.ppo")

        # Frozen reference + reward model; disable dropout so old/new log-probs match.
        for m in (self.reward_model, self.ref_model):
            if m is not None:
                for p in m.parameters():
                    p.requires_grad_(False)
        disable_dropout(self.policy)
        if self.ref_model is not None:
            disable_dropout(self.ref_model)

        if self.ref_model is None and not getattr(self.policy, "is_peft", False):
            raise ValueError("PPO needs a reference policy: pass ref_model, or use LoRA.")

        pc = cfg.ppo
        self.kl_ctl = (
            AdaptiveKLController(pc.kl.init_coef, pc.kl.target, pc.kl.horizon)
            if pc.kl.get("adaptive", True)
            else FixedKLController(pc.kl.init_coef)
        )
        self.opt = torch.optim.AdamW(
            [p for p in self.policy.parameters() if p.requires_grad], lr=pc.lr
        )
        self.reward_norm = RunningMoments() if pc.get("normalize_rewards", False) else None
        self.vllm = None
        if pc.get("use_vllm", False):
            from ..utils.vllm_gen import try_build_vllm

            name = getattr(self.policy.config, "_name_or_path", None)
            if name:
                self.vllm = try_build_vllm(
                    name, self.tokenizer, dtype="auto",
                    max_model_len=int(cfg.data.max_prompt_length) + int(cfg.generation.max_new_tokens))
        self.global_step = 0

    # --- generation ---------------------------------------------------------
    def _gen_config(self) -> GenerationConfig:
        g = self.cfg.generation
        return GenerationConfig(
            max_new_tokens=int(g.max_new_tokens),
            do_sample=True,
            temperature=float(g.get("temperature", 1.0)),
            top_k=int(g.get("top_k", 0)),
            top_p=float(g.get("top_p", 1.0)),
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )

    # --- forward helpers (response slice P-1:T-1) ---------------------------
    def _policy_forward(self, full_ids, full_attn, P):
        logits, values = self.policy(full_ids, full_attn)
        T = full_ids.size(1)
        logp = logprobs_from_logits(logits[:, :-1].float(), full_ids[:, 1:])[:, P - 1:]
        ent = entropy_from_logits(logits[:, :-1].float())[:, P - 1:]
        vals = values[:, P - 1: T - 1]
        return logp, vals, ent

    @torch.no_grad()
    def _ref_logprobs(self, full_ids, full_attn, P):
        if self.ref_model is not None:
            logits = self.ref_model(input_ids=full_ids, attention_mask=full_attn).logits
        else:
            with self.policy.disable_adapter():
                logits = self.policy.actor_logits(full_ids, full_attn)
        return logprobs_from_logits(logits[:, :-1].float(), full_ids[:, 1:])[:, P - 1:]

    # --- rollout ------------------------------------------------------------
    @torch.no_grad()
    def _rollout(self, prompt_batch):
        P = prompt_batch["input_ids"].size(1)
        prompt_ids = prompt_batch["input_ids"].to(self.device)
        prompt_attn = prompt_batch["attention_mask"].to(self.device)

        full_ids = resp_mask = None
        if self.vllm is not None:
            try:
                g = self.cfg.generation
                self.vllm.sync_weights(self.policy.lm)
                full_ids, resp_mask = self.vllm.generate_sequences(
                    prompt_ids, prompt_attn, int(g.max_new_tokens),
                    float(g.get("temperature", 1.0)), float(g.get("top_p", 1.0)),
                    int(g.get("top_k", 0)))
            except Exception as e:  # noqa: BLE001
                self.log.warning("vLLM generation failed (%s); HF fallback.", e)
                self.vllm = full_ids = resp_mask = None
        if full_ids is None:
            full_ids = self.policy.generate(prompt_ids, prompt_attn, self._gen_config())
            resp_mask = response_mask(full_ids[:, P:], self.tokenizer.eos_token_id)
        responses = full_ids[:, P:]
        full_attn = torch.cat([prompt_attn, resp_mask], dim=1)

        logp, values, _ = self._policy_forward(full_ids, full_attn, P)
        ref_logp = self._ref_logprobs(full_ids, full_attn, P)
        scores_raw = self.reward_model(full_ids, full_attn).float()
        if self.cfg.ppo.get("score_clip", None):
            c = float(self.cfg.ppo.score_clip)
            scores_raw = scores_raw.clamp(-c, c)

        # EOS handling: penalize responses that never terminated (length/degenerate guard)
        has_eos = (responses == self.tokenizer.eos_token_id).any(dim=1)
        scores = scores_raw.clone()
        miss_pen = float(self.cfg.ppo.get("missing_eos_penalty", 0.0) or 0.0)
        if miss_pen:
            scores = scores - miss_pen * (~has_eos).float()
        # Running standardization keeps the advantage scale stable as the policy drifts.
        if self.reward_norm is not None:
            self.reward_norm.update(scores)
            scores = (scores - self.reward_norm.mean) / (self.reward_norm.std + 1e-8)

        # per-token reward = -beta*KL  (minus a per-token length cost);  + score at last token
        kl = logp - ref_logp
        rewards = -self.kl_ctl.value * kl
        len_pen = float(self.cfg.ppo.get("length_penalty", 0.0) or 0.0)
        if len_pen:
            rewards = rewards - len_pen
        last_idx = (resp_mask.sum(dim=1).long() - 1).clamp_min(0)
        rows = torch.arange(rewards.size(0), device=self.device)
        rewards[rows, last_idx] += scores
        rewards = rewards * resp_mask

        if self.cfg.ppo.get("whiten_rewards", False):
            rewards = masked_whiten(rewards, resp_mask, shift_mean=False)

        advantages, returns = compute_gae(
            rewards, values, resp_mask, gamma=self.cfg.ppo.gamma, lam=self.cfg.ppo.lam
        )
        if self.cfg.ppo.get("whiten_advantages", True):
            advantages = masked_whiten(advantages, resp_mask)

        return {
            "full_ids": full_ids, "full_attn": full_attn, "resp_mask": resp_mask, "P": P,
            "old_logp": logp, "old_values": values, "advantages": advantages, "returns": returns,
            "scores": scores_raw, "kl": kl, "resp_len": resp_mask.sum(dim=1),
            "frac_eos": has_eos.float().mean(),
        }

    # --- PPO optimization over one rollout ----------------------------------
    def _optimize(self, batch):
        pc = self.cfg.ppo
        B = batch["full_ids"].size(0)
        P = batch["P"]
        idxs = torch.arange(B)
        stats = {k: [] for k in ("pg_loss", "vf_loss", "entropy", "approx_kl", "clipfrac")}

        for _ in range(pc.ppo_epochs):
            perm = idxs[torch.randperm(B)]
            for s in range(0, B, pc.mini_batch_size):
                mb = perm[s: s + pc.mini_batch_size]
                full_ids, full_attn = batch["full_ids"][mb], batch["full_attn"][mb]
                mask = batch["resp_mask"][mb]
                old_logp = batch["old_logp"][mb]
                old_values = batch["old_values"][mb]   # already the [B,G] response slice
                adv, ret = batch["advantages"][mb], batch["returns"][mb]

                logp, values, ent = self._policy_forward(full_ids, full_attn, P)

                ratio = torch.exp(logp - old_logp)
                pg1 = -adv * ratio
                pg2 = -adv * torch.clamp(ratio, 1.0 - pc.cliprange, 1.0 + pc.cliprange)
                pg_loss = masked_mean(torch.max(pg1, pg2), mask)

                v_clipped = old_values + torch.clamp(
                    values - old_values, -pc.cliprange_value, pc.cliprange_value
                )
                vf_loss = 0.5 * masked_mean(
                    torch.max((values - ret) ** 2, (v_clipped - ret) ** 2), mask
                )
                entropy = masked_mean(ent, mask)
                loss = pg_loss + pc.vf_coef * vf_loss - pc.ent_coef * entropy

                self.opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), pc.max_grad_norm)
                self.opt.step()

                with torch.no_grad():
                    stats["pg_loss"].append(pg_loss.item())
                    stats["vf_loss"].append(vf_loss.item())
                    stats["entropy"].append(entropy.item())
                    stats["approx_kl"].append(masked_mean((old_logp - logp), mask).item())
                    stats["clipfrac"].append(
                        masked_mean((torch.abs(ratio - 1.0) > pc.cliprange).float(), mask).item()
                    )
        return {k: (sum(v) / len(v) if v else 0.0) for k, v in stats.items()}

    # --- training loop ------------------------------------------------------
    def train(self, prompt_ds):
        pc = self.cfg.ppo
        coll = PromptCollator(self.tokenizer, self.cfg.data.max_prompt_length)
        loader = DataLoader(prompt_ds, batch_size=pc.rollout_batch_size, shuffle=True,
                            collate_fn=coll, drop_last=True)
        n_iters = max(1, pc.total_episodes // pc.rollout_batch_size)
        self.log.info("PPO: %d iterations x %d prompts/iter (from step %d)",
                      n_iters, pc.rollout_batch_size, self.global_step)

        it = self.global_step  # resume-aware (1 iteration == 1 global step)
        while it < n_iters:
            for prompt_batch in loader:
                if it >= n_iters:
                    break
                rollout = self._rollout(prompt_batch)
                opt_stats = self._optimize(rollout)

                seq_kl = (rollout["kl"] * rollout["resp_mask"]).sum(dim=1)
                mean_kl = seq_kl.mean().item()
                self.kl_ctl.update(mean_kl, pc.rollout_batch_size)
                self.global_step += 1
                it += 1

                if self.global_step % pc.get("log_every", 1) == 0:
                    m = {
                        "reward/score": rollout["scores"].mean().item(),
                        "reward/kl_seq": mean_kl,
                        "reward/kl_coef": self.kl_ctl.value,
                        "reward/resp_len": rollout["resp_len"].float().mean().item(),
                        "reward/frac_eos": rollout["frac_eos"].item(),
                        "adv/mean": masked_mean(rollout["advantages"], rollout["resp_mask"]).item(),
                        "loss/policy": opt_stats["pg_loss"],
                        "loss/value": opt_stats["vf_loss"],
                        "loss/entropy": opt_stats["entropy"],
                        "ppo/approx_kl": opt_stats["approx_kl"],
                        "ppo/clipfrac": opt_stats["clipfrac"],
                    }
                    if self.metrics:
                        self.metrics.log_metrics(m, self.global_step, prefix="ppo")
                    else:
                        self.log.info("iter %d %s", self.global_step, m)
                if self.global_step % pc.get("save_every", 50) == 0:
                    self.save(self.cfg.output_dir)
        self.save(self.cfg.output_dir)
        return self.policy

    def save(self, path):
        self.policy.save_pretrained(path)
        save_tokenizer(self.tokenizer, path)
        state = {"optimizer": self.opt.state_dict(), "global_step": self.global_step,
                 "kl_coef": self.kl_ctl.value}
        if self.reward_norm is not None:
            state["reward_norm"] = {"mean": self.reward_norm.mean, "var": self.reward_norm.var,
                                    "count": self.reward_norm.count}
        torch.save(state, os.path.join(path, "trainer_state.pt"))
        self.log.info("saved PPO policy -> %s", path)

    def load_trainer_state(self, path):
        """Restore optimizer + step + KL-coef (+ reward-norm) to resume a run."""
        sp = os.path.join(path, "trainer_state.pt")
        if not os.path.exists(sp):
            self.log.warning("no trainer_state.pt in %s; fresh optimizer", path)
            return
        st = torch.load(sp, map_location=self.device)
        self.opt.load_state_dict(st["optimizer"])
        self.global_step = st.get("global_step", 0)
        self.kl_ctl.value = st.get("kl_coef", self.kl_ctl.value)
        if self.reward_norm is not None and "reward_norm" in st:
            rn = st["reward_norm"]
            self.reward_norm.mean, self.reward_norm.var, self.reward_norm.count = (
                rn["mean"], rn["var"], rn["count"])
        self.log.info("resumed PPO from step %d (kl_coef=%.4f)", self.global_step, self.kl_ctl.value)
