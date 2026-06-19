"""Prompt-only data for on-policy RL (PPO / GRPO).

Decoder-only generation requires **left padding** so that every prompt's last
token sits at the same final position; the collator enforces this regardless of
the tokenizer's global setting.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .preference import load_preference_dataset


def load_prompt_dataset(
    name: str,
    split: str = "train",
    max_samples: int | None = None,
    num_proc: int | None = None,
):
    """Return a dataset with a single ``prompt`` column (context to complete)."""
    ds = load_preference_dataset(name, split=split, max_samples=max_samples, num_proc=num_proc)
    ds = ds.map(
        lambda ex: {"prompt": ex["prompt"]},
        remove_columns=[c for c in ds.column_names if c != "prompt"],
    )
    ds = ds.filter(lambda ex: len(ex["prompt"]) > 0)
    return ds


def prompt_dataset_from_list(prompts: Sequence[str]):
    from datasets import Dataset

    return Dataset.from_dict({"prompt": list(prompts)})


@dataclass
class PromptCollator:
    tokenizer: object
    max_prompt_length: int = 256

    def __call__(self, batch: list[dict]) -> dict:
        pad_id = self.tokenizer.pad_token_id
        token_lists = [
            self.tokenizer(ex["prompt"], add_special_tokens=False)["input_ids"][
                -self.max_prompt_length:
            ]
            for ex in batch
        ]
        maxlen = max(len(t) for t in token_lists)
        input_ids, attn = [], []
        for t in token_lists:
            pad = maxlen - len(t)
            input_ids.append([pad_id] * pad + t)   # LEFT pad
            attn.append([0] * pad + [1] * len(t))
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "prompt": [ex["prompt"] for ex in batch],
        }
