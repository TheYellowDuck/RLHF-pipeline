"""Direct Preference Optimization — fit preferences without a separate RM/RL.

The implicit reward is r(x,y) = beta * log[ pi(y|x) / pi_ref(y|x) ]; DPO maximizes
the Bradley-Terry likelihood of (chosen ≻ rejected) under that implicit reward.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..data import DPOCollator
from ..utils.common import get_logger
from ..utils.tensor_ops import logprobs_from_logits
from .common import (
    acc_backward,
    acc_clip_grad_norm,
    acc_is_main,
    acc_prepare,
    acc_unwrap,
    autocast_ctx,
    build_optimizer,
    build_scheduler,
    move_to_device,
    save_tokenizer,
    setup_gradient_checkpointing,
)


def dpo_loss(policy_chosen_logps, policy_rejected_logps, ref_chosen_logps, ref_rejected_logps,
             beta: float = 0.1, loss_type: str = "sigmoid", label_smoothing: float = 0.0):
    """Return (loss, chosen_reward, rejected_reward) over the batch."""
    pi_logratios = policy_chosen_logps - policy_rejected_logps
    ref_logratios = ref_chosen_logps - ref_rejected_logps
    logits = pi_logratios - ref_logratios  # the DPO "preference logit"

    if loss_type == "sigmoid":
        losses = (
            -F.logsigmoid(beta * logits) * (1 - label_smoothing)
            - F.logsigmoid(-beta * logits) * label_smoothing
        )
    elif loss_type == "ipo":
        losses = (logits - 1 / (2 * beta)) ** 2
    elif loss_type == "hinge":
        losses = torch.relu(1 - beta * logits)
    else:
        raise ValueError(f"unknown loss_type {loss_type}")

    chosen_reward = beta * (policy_chosen_logps - ref_chosen_logps).detach()
    rejected_reward = beta * (policy_rejected_logps - ref_rejected_logps).detach()
    return losses.mean(), chosen_reward, rejected_reward


class DPOTrainer:
    def __init__(self, model, ref_model, tokenizer, cfg, device, metric_logger=None, accelerator=None):
        self.acc = accelerator
        self.device = accelerator.device if accelerator is not None else device
        self.model = model.to(self.device)
        self.is_peft = hasattr(model, "disable_adapter")
        self.ref_model = ref_model.to(self.device).eval() if ref_model is not None else None
        if self.ref_model is not None:
            for p in self.ref_model.parameters():
                p.requires_grad_(False)
        elif not self.is_peft:
            raise ValueError("DPO needs a reference: pass ref_model, or use LoRA.")
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.metrics = metric_logger
        self.log = get_logger("rlhf.dpo")
        self.bf16 = bool(cfg.train.get("bf16", False))
        setup_gradient_checkpointing(self.model, cfg.train.get("gradient_checkpointing", False))
        self.global_step = 0

    def _seq_logps(self, model, input_ids, attention_mask, loss_mask):
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        per_token = logprobs_from_logits(logits[:, :-1].float(), input_ids[:, 1:])
        mask = loss_mask[:, 1:].float()
        return (per_token * mask).sum(dim=-1)

    def _both_sides_logps(self, model, batch):
        chosen = self._seq_logps(model, batch["chosen_input_ids"], batch["chosen_attention_mask"],
                                 batch["chosen_loss_mask"])
        rejected = self._seq_logps(model, batch["rejected_input_ids"], batch["rejected_attention_mask"],
                                   batch["rejected_loss_mask"])
        return chosen, rejected

    def _ref_logps(self, batch):
        if self.ref_model is not None:
            with torch.no_grad():
                return self._both_sides_logps(self.ref_model, batch)
        # LoRA reference: disable the adapter on the (possibly DDP-wrapped) base.
        base = acc_unwrap(self.acc, self.model)
        with torch.no_grad(), base.disable_adapter():
            return self._both_sides_logps(self.model, batch)

    def _loader(self, ds, shuffle):
        coll = DPOCollator(self.tokenizer, self.cfg.data.max_length,
                           self.cfg.data.get("max_prompt_length", self.cfg.data.max_length // 2))
        return DataLoader(ds, batch_size=self.cfg.train.batch_size, shuffle=shuffle,
                          collate_fn=coll, drop_last=shuffle)

    def train(self, train_ds, eval_ds=None):
        loader = self._loader(train_ds, shuffle=True)
        grad_accum = max(1, int(self.cfg.train.get("grad_accum", 1)))
        total_steps = max(1, len(loader) // grad_accum) * self.cfg.train.epochs
        opt = build_optimizer(self.model, self.cfg.train.lr, self.cfg.train.get("weight_decay", 0.0))
        sched = build_scheduler(opt, total_steps, self.cfg.train.get("warmup_ratio", 0.0))
        self.model, opt, sched, loader = acc_prepare(self.acc, self.model, opt, sched, loader)
        dc = self.cfg.dpo
        self.log.info("DPO training: %d optimizer steps (beta=%.3f, %s)%s", total_steps, dc.beta,
                      dc.loss_type, " (accelerate)" if self.acc is not None else "")

        self.model.train(); opt.zero_grad(); micro = 0
        for epoch in range(self.cfg.train.epochs):
            for batch in loader:
                batch = move_to_device(batch, self.device)
                ref_c, ref_r = self._ref_logps(batch)
                with autocast_ctx(self.device, self.bf16):
                    pol_c, pol_r = self._both_sides_logps(self.model, batch)
                    loss, c_rew, r_rew = dpo_loss(
                        pol_c, pol_r, ref_c, ref_r, beta=dc.beta,
                        loss_type=dc.get("loss_type", "sigmoid"),
                        label_smoothing=dc.get("label_smoothing", 0.0))
                acc_backward(self.acc, loss / grad_accum); micro += 1
                if micro % grad_accum == 0:
                    acc_clip_grad_norm(self.acc, self.model, self.cfg.train.max_grad_norm)
                    opt.step(); sched.step(); opt.zero_grad(); self.global_step += 1
                    main = acc_is_main(self.acc)
                    if main and self.global_step % self.cfg.train.get("log_every", 10) == 0:
                        m = {"loss": loss.item(),
                             "reward_chosen": c_rew.mean().item(), "reward_rejected": r_rew.mean().item(),
                             "reward_margin": (c_rew - r_rew).mean().item(),
                             "accuracy": (c_rew > r_rew).float().mean().item(),
                             "lr": sched.get_last_lr()[0]}
                        if self.metrics: self.metrics.log_metrics(m, self.global_step, prefix="dpo")
                        else: self.log.info("step %d %s", self.global_step, m)
                    if main and eval_ds is not None and self.global_step % self.cfg.train.get("eval_every", 200) == 0:
                        self._run_eval(eval_ds)
        if acc_is_main(self.acc):
            if eval_ds is not None:
                self._run_eval(eval_ds)
            self.save(self.cfg.output_dir)
        return self.model

    @torch.no_grad()
    def evaluate(self, eval_ds):
        self.model.eval(); loader = self._loader(eval_ds, shuffle=False)
        n, correct, margin = 0, 0, 0.0
        for batch in loader:
            batch = move_to_device(batch, self.device)
            ref_c, ref_r = self._both_sides_logps(self.ref_model or self.model, batch) \
                if self.ref_model is not None else self._ref_logps(batch)
            pol_c, pol_r = self._both_sides_logps(self.model, batch)
            dc = self.cfg.dpo
            c_rew = dc.beta * (pol_c - ref_c); r_rew = dc.beta * (pol_r - ref_r)
            bsz = pol_c.size(0); n += bsz
            correct += (c_rew > r_rew).sum().item(); margin += (c_rew - r_rew).sum().item()
        self.model.train()
        return {"eval_accuracy": correct / max(1, n), "eval_margin": margin / max(1, n), "eval_n": n}

    def _run_eval(self, eval_ds):
        m = self.evaluate(eval_ds)
        if self.metrics: self.metrics.log_metrics(m, self.global_step, prefix="dpo")
        else: self.log.info("eval %s", m)

    def save(self, path):
        acc_unwrap(self.acc, self.model).save_pretrained(path); save_tokenizer(self.tokenizer, path)
        self.log.info("saved DPO model -> %s", path)
