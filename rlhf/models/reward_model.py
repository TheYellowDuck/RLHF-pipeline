"""Reward model: a pretrained trunk + scalar head, scored at the last real token."""

from __future__ import annotations

import json
import os

import torch
import torch.nn as nn

from .loading import apply_lora, load_base_model
from .value_head import ValueHead

_CONFIG_NAME = "reward_config.json"
_HEAD_NAME = "value_head.pt"


def last_token_indices(attention_mask: torch.Tensor) -> torch.Tensor:
    """Index of the last non-pad token per row (right-padded). [B] long."""
    return attention_mask.sum(dim=1).clamp_min(1).long() - 1


class RewardModel(nn.Module):
    def __init__(self, backbone: nn.Module, hidden_size: int):
        super().__init__()
        self.backbone = backbone
        self.value_head = ValueHead(hidden_size)
        self.config = backbone.config

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_per_token: bool = False,
    ):
        out = self.backbone(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )
        hidden = out.last_hidden_state                       # [B, T, H]
        per_token = self.value_head(hidden.to(self.value_head.proj.weight.dtype))  # [B, T]
        idx = last_token_indices(attention_mask)             # [B]
        rewards = per_token[torch.arange(per_token.size(0), device=per_token.device), idx]
        if return_per_token:
            return rewards, per_token
        return rewards

    # --- construction / (de)serialization ----------------------------------
    @classmethod
    def from_backbone(
        cls,
        name_or_path: str,
        dtype: torch.dtype = torch.float32,
        use_lora: bool = False,
        lora_cfg=None,
    ) -> "RewardModel":
        backbone = load_base_model(name_or_path, dtype=dtype)
        hidden_size = backbone.config.hidden_size
        if use_lora:
            backbone = apply_lora(backbone, lora_cfg or {}, task_type="FEATURE_EXTRACTION")
        return cls(backbone, hidden_size)

    def enable_gradient_checkpointing(self):
        if hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable()

    def save_pretrained(self, path: str):
        os.makedirs(path, exist_ok=True)
        self.backbone.save_pretrained(path)
        torch.save(self.value_head.state_dict(), os.path.join(path, _HEAD_NAME))
        with open(os.path.join(path, _CONFIG_NAME), "w") as f:
            json.dump({"hidden_size": self.value_head.proj.in_features}, f)

    @classmethod
    def from_pretrained(
        cls, path: str, dtype: torch.dtype = torch.float32
    ) -> "RewardModel":
        backbone = load_base_model(path, dtype=dtype)
        with open(os.path.join(path, _CONFIG_NAME)) as f:
            cfg = json.load(f)
        model = cls(backbone, cfg["hidden_size"])
        head_path = os.path.join(path, _HEAD_NAME)
        if os.path.exists(head_path):
            model.value_head.load_state_dict(torch.load(head_path, map_location="cpu"))
        return model
