"""Actor-critic policy: a causal LM (actor) sharing a trunk with a value head.

The value head reads the final hidden states, giving per-token state values for
GAE. With LoRA, only the adapter + value head train and the frozen reference
policy is recovered by disabling the adapter (no second copy of weights).
"""

from __future__ import annotations

import json
import os
from contextlib import contextmanager

import torch
import torch.nn as nn

from .loading import apply_lora, load_causal_lm
from .value_head import ValueHead

_HEAD_NAME = "value_head.pt"
_CONFIG_NAME = "policy_config.json"


class ActorCriticPolicy(nn.Module):
    def __init__(self, lm: nn.Module, hidden_size: int, is_peft: bool = False):
        super().__init__()
        self.lm = lm
        self.value_head = ValueHead(hidden_size)
        self.is_peft = is_peft
        self.config = lm.config

    # --- forward passes -----------------------------------------------------
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """Return (logits [B,T,V], values [B,T])."""
        out = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = out.hidden_states[-1]
        values = self.value_head(hidden.to(self.value_head.proj.weight.dtype))
        return out.logits, values

    def actor_logits(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Logits only (used for the reference policy)."""
        out = self.lm(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        return out.logits

    @contextmanager
    def disable_adapter(self):
        """Context manager yielding the reference (pre-RL) policy for LoRA runs."""
        if self.is_peft:
            with self.lm.disable_adapter():
                yield
        else:
            yield

    @torch.no_grad()
    def generate(self, input_ids, attention_mask, generation_config):
        return self.lm.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
        )

    def enable_gradient_checkpointing(self):
        if hasattr(self.lm, "gradient_checkpointing_enable"):
            self.lm.gradient_checkpointing_enable()

    # --- construction / (de)serialization ----------------------------------
    @classmethod
    def from_pretrained_lm(
        cls,
        name_or_path: str,
        dtype: torch.dtype = torch.float32,
        use_lora: bool = False,
        lora_cfg=None,
    ) -> "ActorCriticPolicy":
        lm = load_causal_lm(name_or_path, dtype=dtype)
        hidden_size = lm.config.hidden_size
        if use_lora:
            lm = apply_lora(lm, lora_cfg or {}, task_type="CAUSAL_LM")
        return cls(lm, hidden_size, is_peft=use_lora)

    def save_pretrained(self, path: str):
        os.makedirs(path, exist_ok=True)
        self.lm.save_pretrained(path)  # full weights, or adapter if peft
        torch.save(self.value_head.state_dict(), os.path.join(path, _HEAD_NAME))
        with open(os.path.join(path, _CONFIG_NAME), "w") as f:
            json.dump({"hidden_size": self.value_head.proj.in_features, "is_peft": self.is_peft}, f)

    @classmethod
    def from_pretrained(cls, path: str, dtype: torch.dtype = torch.float32) -> "ActorCriticPolicy":
        is_peft = os.path.exists(os.path.join(path, "adapter_config.json"))
        if is_peft:
            from peft import PeftModel

            with open(os.path.join(path, "adapter_config.json")) as f:
                base = json.load(f)["base_model_name_or_path"]
            lm = load_causal_lm(base, dtype=dtype)
            lm = PeftModel.from_pretrained(lm, path, is_trainable=True)
        else:
            lm = load_causal_lm(path, dtype=dtype)
        hidden_size = lm.config.hidden_size
        model = cls(lm, hidden_size, is_peft=is_peft)
        head_path = os.path.join(path, _HEAD_NAME)
        if os.path.exists(head_path):
            model.value_head.load_state_dict(torch.load(head_path, map_location="cpu"))
        return model
