"""Supervised fine-tuning of a causal LM on (prompt, response) demonstrations."""

from __future__ import annotations

import math

import torch
from torch.utils.data import DataLoader

from ..data import SFTCollator
from ..utils.common import get_logger
from .common import autocast_ctx, build_optimizer, build_scheduler, move_to_device, save_tokenizer


class SFTTrainer:
    """Trains a HuggingFace causal LM via teacher forcing (labels with -100 mask)."""

    def __init__(self, model, tokenizer, cfg, device, metric_logger=None):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.device = device
        self.metrics = metric_logger
        self.log = get_logger("rlhf.sft")
        self.bf16 = bool(cfg.train.get("bf16", False))
        self.global_step = 0

    def _loader(self, ds, shuffle):
        coll = SFTCollator(self.tokenizer, self.cfg.data.max_length, self.cfg.data.get("mask_prompt", True))
        return DataLoader(ds, batch_size=self.cfg.train.batch_size, shuffle=shuffle,
                          collate_fn=coll, drop_last=shuffle)

    def train(self, train_ds, eval_ds=None):
        loader = self._loader(train_ds, shuffle=True)
        grad_accum = max(1, int(self.cfg.train.get("grad_accum", 1)))
        steps_per_epoch = max(1, len(loader) // grad_accum)
        total_steps = steps_per_epoch * self.cfg.train.epochs
        opt = build_optimizer(self.model, self.cfg.train.lr, self.cfg.train.get("weight_decay", 0.0))
        sched = build_scheduler(opt, total_steps, self.cfg.train.get("warmup_ratio", 0.0))
        self.log.info("SFT training: %d optimizer steps", total_steps)

        self.model.train(); opt.zero_grad(); micro = 0
        for epoch in range(self.cfg.train.epochs):
            for batch in loader:
                batch = move_to_device(batch, self.device)
                with autocast_ctx(self.device, self.bf16):
                    out = self.model(input_ids=batch["input_ids"],
                                     attention_mask=batch["attention_mask"], labels=batch["labels"])
                    loss = out.loss / grad_accum
                loss.backward(); micro += 1
                if micro % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.train.max_grad_norm)
                    opt.step(); sched.step(); opt.zero_grad(); self.global_step += 1
                    if self.global_step % self.cfg.train.get("log_every", 10) == 0:
                        l = loss.item() * grad_accum
                        m = {"loss": l, "ppl": math.exp(min(20, l)), "lr": sched.get_last_lr()[0]}
                        if self.metrics: self.metrics.log_metrics(m, self.global_step, prefix="sft")
                        else: self.log.info("step %d %s", self.global_step, m)
                    if eval_ds is not None and self.global_step % self.cfg.train.get("eval_every", 200) == 0:
                        self._run_eval(eval_ds)
                    if self.global_step % self.cfg.train.get("save_every", 500) == 0:
                        self.save(self.cfg.output_dir)
        if eval_ds is not None:
            self._run_eval(eval_ds)
        self.save(self.cfg.output_dir)
        return self.model

    @torch.no_grad()
    def evaluate(self, eval_ds):
        self.model.eval(); loader = self._loader(eval_ds, shuffle=False)
        loss_sum, n = 0.0, 0
        for batch in loader:
            batch = move_to_device(batch, self.device)
            with autocast_ctx(self.device, self.bf16):
                out = self.model(input_ids=batch["input_ids"],
                                 attention_mask=batch["attention_mask"], labels=batch["labels"])
            loss_sum += out.loss.item() * batch["input_ids"].size(0); n += batch["input_ids"].size(0)
        self.model.train()
        avg = loss_sum / max(1, n)
        return {"eval_loss": avg, "eval_ppl": math.exp(min(20, avg)), "eval_n": n}

    def _run_eval(self, eval_ds):
        m = self.evaluate(eval_ds)
        if self.metrics: self.metrics.log_metrics(m, self.global_step, prefix="sft")
        else: self.log.info("eval %s", m)

    def save(self, path):
        self.model.save_pretrained(path); save_tokenizer(self.tokenizer, path)
        self.log.info("saved SFT model -> %s", path)
