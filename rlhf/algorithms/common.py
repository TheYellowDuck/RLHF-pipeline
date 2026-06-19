"""Shared training utilities for the supervised-style trainers (RM/SFT/DPO)."""

from __future__ import annotations

from contextlib import contextmanager, nullcontext

import torch


@contextmanager
def autocast_ctx(device: torch.device, enabled: bool, dtype: torch.dtype = torch.bfloat16):
    """Autocast on CUDA when enabled; otherwise a no-op (CPU/MPS stay fp32)."""
    if enabled and device.type == "cuda":
        with torch.autocast(device_type="cuda", dtype=dtype):
            yield
    else:
        with nullcontext():
            yield


def build_optimizer(model, lr: float, weight_decay: float = 0.0):
    """AdamW with no weight decay on biases / norm parameters."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or name.endswith(".bias") or "norm" in name.lower() or "ln" in name.lower():
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=lr)


def build_scheduler(optimizer, num_training_steps: int, warmup_ratio: float = 0.0, kind: str = "cosine"):
    from transformers import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup

    warmup_steps = int(num_training_steps * warmup_ratio)
    if kind == "linear":
        return get_linear_schedule_with_warmup(optimizer, warmup_steps, num_training_steps)
    return get_cosine_schedule_with_warmup(optimizer, warmup_steps, num_training_steps)


def move_to_device(batch: dict, device: torch.device) -> dict:
    return {
        k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v)
        for k, v in batch.items()
    }


def save_tokenizer(tokenizer, path: str):
    try:
        tokenizer.save_pretrained(path)
    except Exception:  # noqa: BLE001
        pass
