"""SFT data: (prompt, response) demonstrations + a teacher-forcing collator.

For HH-RLHF the ``chosen`` transcript is treated as the demonstration. With
``mask_prompt=True`` the loss is computed on response tokens only.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from .preference import load_preference_dataset


def load_sft_dataset(
    name: str,
    split: str = "train",
    max_samples: int | None = None,
    num_proc: int | None = None,
):
    """Return a dataset with columns ``prompt`` and ``response`` (the chosen one)."""
    ds = load_preference_dataset(name, split=split, max_samples=max_samples, num_proc=num_proc)
    ds = ds.map(
        lambda ex: {"prompt": ex["prompt"], "response": ex["chosen"]},
        remove_columns=[c for c in ds.column_names if c not in ("prompt",)],
    )
    return ds


def sft_dataset_from_pairs(pairs: Sequence[tuple[str, str]]):
    from datasets import Dataset

    return Dataset.from_dict(
        {"prompt": [p for p, _ in pairs], "response": [r for _, r in pairs]}
    )


@dataclass
class SFTCollator:
    tokenizer: object
    max_length: int = 512
    mask_prompt: bool = True

    def __call__(self, batch: list[dict]) -> dict:
        pad_id = self.tokenizer.pad_token_id
        eos_id = self.tokenizer.eos_token_id
        seqs, label_seqs = [], []
        for ex in batch:
            p_ids = self.tokenizer(ex["prompt"], add_special_tokens=False)["input_ids"]
            r_ids = self.tokenizer(ex["response"], add_special_tokens=False)["input_ids"]
            if eos_id is not None:
                r_ids = r_ids + [eos_id]
            ids = (p_ids + r_ids)[: self.max_length]
            if self.mask_prompt:
                labels = ([-100] * len(p_ids) + r_ids)[: self.max_length]
            else:
                labels = list(ids)
            seqs.append(ids)
            label_seqs.append(labels)

        maxlen = max(len(s) for s in seqs)
        input_ids, attn, labels_out = [], [], []
        for ids, labels in zip(seqs, label_seqs):
            pad = maxlen - len(ids)
            input_ids.append(ids + [pad_id] * pad)
            attn.append([1] * len(ids) + [0] * pad)
            labels_out.append(labels + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels_out, dtype=torch.long),
        }
