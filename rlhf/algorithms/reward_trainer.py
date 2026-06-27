"""Reward-model training with the Bradley-Terry pairwise preference loss.

P(chosen ≻ rejected) = sigmoid(r_chosen - r_rejected), so the negative
log-likelihood is  -log sigmoid(r_chosen - r_rejected).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..data import PreferenceCollator
from ..utils.common import get_logger
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
)


def bradley_terry_loss(chosen_rewards: torch.Tensor, rejected_rewards: torch.Tensor,
                       margin: float = 0.0, label_smoothing: float = 0.0):
    """Bradley-Terry preference loss with optional margin + label smoothing.

    ``margin`` pushes chosen above rejected by a fixed gap (Llama-2 style; helps on
    clearly-separable pairs, arXiv:2307.09288). ``label_smoothing`` puts a small mass
    on the *wrong* ordering, regularizing against noisy/mislabeled pairs — HH-RLHF
    has ~25% effectively-mislabeled pairs (arXiv:2401.06080), so this helps held-out
    accuracy on noisy preference data (cDPO-style soft labels).
    """
    diff = chosen_rewards - rejected_rewards - margin
    if label_smoothing > 0.0:
        return -((1.0 - label_smoothing) * F.logsigmoid(diff)
                 + label_smoothing * F.logsigmoid(-diff)).mean()
    return -F.logsigmoid(diff).mean()


class RewardTrainer:
    def __init__(self, model, tokenizer, cfg, device, metric_logger=None, accelerator=None):
        self.acc = accelerator
        self.device = accelerator.device if accelerator is not None else device
        self.model = model.to(self.device)
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.metrics = metric_logger
        self.log = get_logger("rlhf.reward")
        self.bf16 = bool(cfg.train.get("bf16", False))
        self.margin = float(cfg.train.get("margin", 0.0))
        self.label_smoothing = float(cfg.train.get("label_smoothing", 0.0))
        if cfg.train.get("gradient_checkpointing", False):
            self.model.enable_gradient_checkpointing()
        self.global_step = 0

    def _loader(self, ds, shuffle: bool):
        coll = PreferenceCollator(self.tokenizer, max_length=self.cfg.data.max_length)
        return DataLoader(
            ds, batch_size=self.cfg.train.batch_size, shuffle=shuffle, collate_fn=coll, drop_last=shuffle
        )

    def _scores(self, batch):
        c = self.model(batch["chosen_input_ids"], batch["chosen_attention_mask"])
        r = self.model(batch["rejected_input_ids"], batch["rejected_attention_mask"])
        return c, r

    def train(self, train_ds, eval_ds=None):
        loader = self._loader(train_ds, shuffle=True)
        grad_accum = max(1, int(self.cfg.train.get("grad_accum", 1)))
        steps_per_epoch = max(1, len(loader) // grad_accum)
        total_steps = steps_per_epoch * self.cfg.train.epochs
        opt = build_optimizer(self.model, self.cfg.train.lr, self.cfg.train.get("weight_decay", 0.0))
        sched = build_scheduler(opt, total_steps, self.cfg.train.get("warmup_ratio", 0.0))
        self.model, opt, sched, loader = acc_prepare(self.acc, self.model, opt, sched, loader)
        self.log.info("RM training: %d optimizer steps (%d/epoch x %d epochs)%s",
                      total_steps, steps_per_epoch, self.cfg.train.epochs,
                      " (accelerate)" if self.acc is not None else "")

        self.model.train()
        opt.zero_grad()
        micro = 0
        for epoch in range(self.cfg.train.epochs):
            for batch in loader:
                batch = move_to_device(batch, self.device)
                with autocast_ctx(self.device, self.bf16):
                    c, r = self._scores(batch)
                    loss = bradley_terry_loss(c, r, self.margin, self.label_smoothing) / grad_accum
                acc_backward(self.acc, loss)
                micro += 1
                if micro % grad_accum == 0:
                    acc_clip_grad_norm(self.acc, self.model, self.cfg.train.max_grad_norm)
                    opt.step(); sched.step(); opt.zero_grad()
                    self.global_step += 1
                    main = acc_is_main(self.acc)
                    if main and self.global_step % self.cfg.train.get("log_every", 10) == 0:
                        acc = (c > r).float().mean().item()
                        m = {"loss": loss.item() * grad_accum, "accuracy": acc,
                             "reward_chosen": c.mean().item(), "reward_rejected": r.mean().item(),
                             "reward_margin": (c - r).mean().item(), "lr": sched.get_last_lr()[0]}
                        if self.metrics: self.metrics.log_metrics(m, self.global_step, prefix="rm")
                        else: self.log.info("step %d %s", self.global_step, m)
                    if main and eval_ds is not None and self.global_step % self.cfg.train.get("eval_every", 200) == 0:
                        self._run_eval(eval_ds)
                    if main and self.global_step % self.cfg.train.get("save_every", 500) == 0:
                        self.save(self.cfg.output_dir)
        if acc_is_main(self.acc):
            if eval_ds is not None:
                self._run_eval(eval_ds)
            self.save(self.cfg.output_dir, merge=True)
        return self.model

    @torch.no_grad()
    def evaluate(self, eval_ds):
        self.model.eval()
        loader = self._loader(eval_ds, shuffle=False)
        n, correct, loss_sum, margin_sum = 0, 0, 0.0, 0.0
        for batch in loader:
            batch = move_to_device(batch, self.device)
            with autocast_ctx(self.device, self.bf16):
                c, r = self._scores(batch)
                loss = bradley_terry_loss(c, r, self.margin, self.label_smoothing)
            bsz = c.size(0)
            n += bsz
            correct += (c > r).sum().item()
            loss_sum += loss.item() * bsz
            margin_sum += (c - r).sum().item()
        self.model.train()
        return {"eval_accuracy": correct / max(1, n), "eval_loss": loss_sum / max(1, n),
                "eval_margin": margin_sum / max(1, n), "eval_n": n}

    def _run_eval(self, eval_ds):
        m = self.evaluate(eval_ds)
        if self.metrics: self.metrics.log_metrics(m, self.global_step, prefix="rm")
        else: self.log.info("eval %s", m)

    def save(self, path: str, merge: bool = False):
        acc_unwrap(self.acc, self.model).save_pretrained(path, merge=merge)
        save_tokenizer(self.tokenizer, path)
        self.log.info("saved reward model -> %s%s", path, " (LoRA merged)" if merge else "")
