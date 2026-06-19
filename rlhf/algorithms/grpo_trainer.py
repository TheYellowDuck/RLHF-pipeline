"""GRPO — Group Relative Policy Optimization (DeepSeek-style, critic-free).

Instead of a learned value baseline, GRPO samples a *group* of G responses per
prompt and uses the group's mean (optionally /std) reward as the baseline, so the
advantage of a response is how much better than its peers it is. The policy loss
is the clipped PPO surrogate with an explicit per-token KL-to-reference penalty
(the unbiased k3 estimator), so no value head is needed.
"""

from __future__ import annotations

import os

import torch
from torch.utils.data import DataLoader
from transformers import GenerationConfig

from ..data import PromptCollator
from ..utils.common import disable_dropout, get_logger
from ..utils.tensor_ops import logprobs_from_logits, masked_mean, response_mask
from .common import save_tokenizer


class GRPOTrainer:
    def __init__(self, policy, reward_model, tokenizer, cfg, device, ref_model=None, metric_logger=None):
        self.policy = policy.to(device)
        self.reward_model = reward_model.to(device).eval()
        self.ref_model = ref_model.to(device).eval() if ref_model is not None else None
        for m in (self.reward_model, self.ref_model):
            if m is not None:
                for p in m.parameters():
                    p.requires_grad_(False)
        disable_dropout(self.policy)
        if self.ref_model is None and not getattr(self.policy, "is_peft", False):
            raise ValueError("GRPO needs a reference policy: pass ref_model, or use LoRA.")
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.device = device
        self.metrics = metric_logger
        self.log = get_logger("rlhf.grpo")
        self.opt = torch.optim.AdamW([p for p in self.policy.parameters() if p.requires_grad], lr=cfg.grpo.lr)
        self.vllm = None
        if cfg.grpo.get("use_vllm", False):
            from ..utils.vllm_gen import try_build_vllm

            name = getattr(self.policy.config, "_name_or_path", None)
            if name:
                self.vllm = try_build_vllm(
                    name, self.tokenizer, dtype="auto",
                    max_model_len=int(cfg.data.max_prompt_length) + int(cfg.generation.max_new_tokens))
        self.global_step = 0

    def _gen_config(self):
        g = self.cfg.generation
        return GenerationConfig(max_new_tokens=int(g.max_new_tokens), do_sample=True,
                                temperature=float(g.get("temperature", 1.0)),
                                top_k=int(g.get("top_k", 0)), top_p=float(g.get("top_p", 1.0)),
                                pad_token_id=self.tokenizer.pad_token_id,
                                eos_token_id=self.tokenizer.eos_token_id)

    def _logp(self, full_ids, full_attn, P):
        logits = self.policy.actor_logits(full_ids, full_attn)
        return logprobs_from_logits(logits[:, :-1].float(), full_ids[:, 1:])[:, P - 1:]

    @torch.no_grad()
    def _ref_logp(self, full_ids, full_attn, P):
        if self.ref_model is not None:
            logits = self.ref_model(input_ids=full_ids, attention_mask=full_attn).logits
        else:
            with self.policy.disable_adapter():
                logits = self.policy.actor_logits(full_ids, full_attn)
        return logprobs_from_logits(logits[:, :-1].float(), full_ids[:, 1:])[:, P - 1:]

    @torch.no_grad()
    def _rollout(self, prompt_batch):
        gc = self.cfg.grpo
        P = prompt_batch["input_ids"].size(1)
        ids = prompt_batch["input_ids"].to(self.device).repeat_interleave(gc.group_size, dim=0)
        attn = prompt_batch["attention_mask"].to(self.device).repeat_interleave(gc.group_size, dim=0)

        seqs = resp_mask = None
        if self.vllm is not None:
            try:
                g = self.cfg.generation
                self.vllm.sync_weights(self.policy.lm)
                seqs, resp_mask = self.vllm.generate_sequences(
                    ids, attn, int(g.max_new_tokens), float(g.get("temperature", 1.0)),
                    float(g.get("top_p", 1.0)), int(g.get("top_k", 0)))
            except Exception as e:  # noqa: BLE001
                self.log.warning("vLLM generation failed (%s); HF fallback.", e)
                self.vllm = seqs = resp_mask = None
        if seqs is None:
            seqs = self.policy.generate(ids, attn, self._gen_config())
            resp_mask = response_mask(seqs[:, P:], self.tokenizer.eos_token_id)
        full_attn = torch.cat([attn, resp_mask], dim=1)

        scores = self.reward_model(seqs, full_attn).float()                 # [N]
        n_prompts = prompt_batch["input_ids"].size(0)
        grouped = scores.view(n_prompts, gc.group_size)
        mean = grouped.mean(dim=1, keepdim=True)
        adv = grouped - mean
        if gc.get("scale_rewards", True):
            adv = adv / (grouped.std(dim=1, keepdim=True) + 1e-6)
        adv = adv.reshape(-1)                                               # [N]

        old_logp = self._logp(seqs, full_attn, P)
        ref_logp = self._ref_logp(seqs, full_attn, P)
        return {"full_ids": seqs, "full_attn": full_attn, "resp_mask": resp_mask, "P": P,
                "old_logp": old_logp, "ref_logp": ref_logp, "adv": adv, "scores": scores,
                "group_std": grouped.std(dim=1).mean(), "resp_len": resp_mask.sum(dim=1)}

    def _optimize(self, batch):
        gc = self.cfg.grpo
        N = batch["full_ids"].size(0)
        P = batch["P"]
        idxs = torch.arange(N)
        pg_acc, kl_acc, n = 0.0, 0.0, 0
        for _ in range(gc.get("grpo_epochs", 1)):
            perm = idxs[torch.randperm(N)]
            for s in range(0, N, gc.mini_batch_size):
                mb = perm[s: s + gc.mini_batch_size]
                full_ids, full_attn, mask = batch["full_ids"][mb], batch["full_attn"][mb], batch["resp_mask"][mb]
                old_logp, ref_logp = batch["old_logp"][mb], batch["ref_logp"][mb]
                adv = batch["adv"][mb].unsqueeze(1)                          # [mb,1] broadcast over tokens

                logp = self._logp(full_ids, full_attn, P)
                ratio = torch.exp(logp - old_logp)
                pg1 = -adv * ratio
                pg2 = -adv * torch.clamp(ratio, 1.0 - gc.cliprange, 1.0 + gc.cliprange)
                pg_loss = torch.max(pg1, pg2)
                # unbiased per-token KL (k3): E[exp(r)-r-1] >= 0, r = ref_logp - logp
                logr = ref_logp - logp
                kl = torch.exp(logr) - logr - 1.0
                per_token = pg_loss + gc.kl_coef * kl
                loss = masked_mean(per_token, mask)

                self.opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), gc.max_grad_norm)
                self.opt.step()
                pg_acc += masked_mean(pg_loss, mask).item(); kl_acc += masked_mean(kl, mask).item(); n += 1
        return {"pg_loss": pg_acc / max(1, n), "kl": kl_acc / max(1, n)}

    def train(self, prompt_ds):
        gc = self.cfg.grpo
        coll = PromptCollator(self.tokenizer, self.cfg.data.max_prompt_length)
        loader = DataLoader(prompt_ds, batch_size=gc.prompts_per_step, shuffle=True,
                            collate_fn=coll, drop_last=True)
        n_iters = max(1, gc.total_episodes // (gc.prompts_per_step * gc.group_size))
        self.log.info("GRPO: %d iters x %d prompts x %d samples (from step %d)",
                      n_iters, gc.prompts_per_step, gc.group_size, self.global_step)

        it = self.global_step  # resume-aware
        while it < n_iters:
            for prompt_batch in loader:
                if it >= n_iters:
                    break
                rollout = self._rollout(prompt_batch)
                opt_stats = self._optimize(rollout)
                self.global_step += 1; it += 1
                if self.global_step % gc.get("log_every", 1) == 0:
                    m = {"reward/score": rollout["scores"].mean().item(),
                         "reward/group_std": rollout["group_std"].item(),
                         "reward/resp_len": rollout["resp_len"].float().mean().item(),
                         "loss/policy": opt_stats["pg_loss"], "loss/kl": opt_stats["kl"]}
                    if self.metrics: self.metrics.log_metrics(m, self.global_step, prefix="grpo")
                    else: self.log.info("iter %d %s", self.global_step, m)
                if self.global_step % gc.get("save_every", 50) == 0:
                    self.save(self.cfg.output_dir)
        self.save(self.cfg.output_dir)
        return self.policy

    def save(self, path):
        self.policy.save_pretrained(path); save_tokenizer(self.tokenizer, path)
        torch.save({"optimizer": self.opt.state_dict(), "global_step": self.global_step},
                   os.path.join(path, "trainer_state.pt"))
        self.log.info("saved GRPO policy -> %s", path)

    def load_trainer_state(self, path):
        sp = os.path.join(path, "trainer_state.pt")
        if not os.path.exists(sp):
            self.log.warning("no trainer_state.pt in %s; fresh optimizer", path)
            return
        st = torch.load(sp, map_location=self.device)
        self.opt.load_state_dict(st["optimizer"])
        self.global_step = st.get("global_step", 0)
        self.log.info("resumed GRPO from step %d", self.global_step)
