"""Reward model: a pretrained trunk + scalar head, scored at the last real token."""

from __future__ import annotations

import json
import os

import torch
import torch.nn as nn

from .loading import apply_lora, load_base_model, load_causal_lm, merge_if_peft
from .value_head import ValueHead

_CONFIG_NAME = "reward_config.json"
_HEAD_NAME = "value_head.pt"


def last_token_indices(attention_mask: torch.Tensor) -> torch.Tensor:
    """Index of the last non-pad token per row, robust to LEFT or RIGHT padding.

    PPO/GRPO score sequences shaped ``[left-padded prompt][response right-pad]``,
    so the real tokens are not a prefix and ``sum-1`` would land inside the
    prompt. Find the last position where attention_mask == 1 instead.
    """
    T = attention_mask.shape[1]
    last_from_right = torch.flip(attention_mask, dims=[1]).float().argmax(dim=1)
    return (T - 1 - last_from_right).long()


class RewardModel(nn.Module):
    def __init__(self, backbone: nn.Module, hidden_size: int, aux_lm: bool = False):
        super().__init__()
        self.backbone = backbone
        self.value_head = ValueHead(hidden_size)
        self.config = backbone.config
        # GRM mode (arXiv:2406.10216): backbone is an AutoModelForCausalLM whose LM head is
        # kept, so an auxiliary language-modeling loss can regularize the hidden states. The
        # reward is still the value head on the last token's hidden state — identical to the
        # trunk-only path, since hidden_states[-1] is the same post-final-norm representation.
        self.aux_lm = aux_lm

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_per_token: bool = False,
        return_lm_logits: bool = False,
    ):
        if self.aux_lm:
            out = self.backbone(
                input_ids=input_ids, attention_mask=attention_mask,
                output_hidden_states=True, return_dict=True,
            )
            hidden = out.hidden_states[-1]                   # == AutoModel.last_hidden_state
        else:
            out = self.backbone(
                input_ids=input_ids, attention_mask=attention_mask, return_dict=True
            )
            hidden = out.last_hidden_state                   # [B, T, H]
        per_token = self.value_head(hidden.to(self.value_head.proj.weight.dtype))  # [B, T]
        idx = last_token_indices(attention_mask)             # [B]
        rewards = per_token[torch.arange(per_token.size(0), device=per_token.device), idx]
        if return_lm_logits:
            if not self.aux_lm:
                raise ValueError("return_lm_logits requires a model built with aux_lm=True")
            if return_per_token:
                return rewards, per_token, out.logits
            return rewards, out.logits
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
        aux_lm: bool = False,
    ) -> "RewardModel":
        # GRM keeps the LM head -> load the full causal LM; otherwise just the trunk.
        backbone = (load_causal_lm if aux_lm else load_base_model)(name_or_path, dtype=dtype)
        hidden_size = backbone.config.hidden_size
        if use_lora:
            task = "CAUSAL_LM" if aux_lm else "FEATURE_EXTRACTION"
            backbone = apply_lora(backbone, lora_cfg or {}, task_type=task)
        return cls(backbone, hidden_size, aux_lm=aux_lm)

    def enable_gradient_checkpointing(self):
        if getattr(self.backbone, "config", None) is not None:
            self.backbone.config.use_cache = False
        if hasattr(self.backbone, "gradient_checkpointing_enable"):
            try:
                self.backbone.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False})
            except TypeError:
                self.backbone.gradient_checkpointing_enable()
        if hasattr(self.backbone, "enable_input_require_grads"):
            self.backbone.enable_input_require_grads()

    def save_pretrained(self, path: str, merge: bool = False):
        os.makedirs(path, exist_ok=True)
        backbone = merge_if_peft(self.backbone) if merge else self.backbone
        backbone.save_pretrained(path)
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
