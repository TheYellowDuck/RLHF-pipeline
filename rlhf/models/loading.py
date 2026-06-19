"""Backbone / tokenizer loaders with transformers-v5-safe kwargs and LoRA."""

from __future__ import annotations

import inspect

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer

from ..utils.common import get_logger

_log = get_logger("rlhf.models")


def load_tokenizer(name_or_path: str):
    tok = AutoTokenizer.from_pretrained(name_or_path)
    if tok.pad_token_id is None:
        # GPT-2 et al. have no pad token; reuse EOS (attention masks handle it).
        tok.pad_token = tok.eos_token
    return tok


def _from_pretrained(cls, name_or_path: str, dtype: torch.dtype, **kwargs):
    """from_pretrained that works on both transformers v4 (torch_dtype) and v5 (dtype)."""
    params = inspect.signature(cls.from_pretrained).parameters
    if "dtype" in params:
        kwargs["dtype"] = dtype
    else:  # older transformers
        kwargs["torch_dtype"] = dtype
    return cls.from_pretrained(name_or_path, **kwargs)


def load_causal_lm(name_or_path: str, dtype: torch.dtype = torch.float32, **kwargs):
    return _from_pretrained(AutoModelForCausalLM, name_or_path, dtype, **kwargs)


def load_base_model(name_or_path: str, dtype: torch.dtype = torch.float32, **kwargs):
    """Load the encoder/decoder *trunk* (no LM head) for a reward backbone."""
    return _from_pretrained(AutoModel, name_or_path, dtype, **kwargs)


def apply_lora(model, lora_cfg, task_type: str = "CAUSAL_LM"):
    """Wrap a model with a LoRA adapter. ``target_modules=None`` -> all linear layers."""
    from peft import LoraConfig, get_peft_model

    target_modules = lora_cfg.get("target_modules", None)
    config = LoraConfig(
        r=int(lora_cfg.get("r", 16)),
        lora_alpha=int(lora_cfg.get("alpha", 32)),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        target_modules=target_modules if target_modules else "all-linear",
        bias="none",
        task_type=task_type,
    )
    model = get_peft_model(model, config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    _log.info("LoRA enabled: %.2f%% params trainable (%d / %d)", 100 * trainable / total, trainable, total)
    return model
