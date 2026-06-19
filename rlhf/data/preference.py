"""Preference data: load + normalize (prompt, chosen, rejected) and collate.

Two input shapes are supported automatically:
  * Anthropic HH-RLHF style: columns ``chosen`` / ``rejected`` are *full
    transcripts*; the shared prefix up to the final ``\\n\\nAssistant:`` is the
    prompt and the divergent suffix is the response.
  * Explicit style: columns ``prompt`` + ``chosen`` + ``rejected`` where the
    latter two are response strings (e.g. Dahoas/rm-static, ultrafeedback).

Normalized examples are dicts: ``{"prompt": str, "chosen": str, "rejected": str}``
where chosen/rejected are *response-only* strings.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

ASSISTANT_TAG = "\n\nAssistant:"


def extract_anthropic_prompt(transcript: str) -> str:
    """Return the prompt prefix (through the final ``\\n\\nAssistant:``)."""
    idx = transcript.rfind(ASSISTANT_TAG)
    if idx == -1:
        return ""
    return transcript[: idx + len(ASSISTANT_TAG)]


def _normalize_hh(example: dict) -> dict:
    chosen, rejected = example["chosen"], example["rejected"]
    prompt = extract_anthropic_prompt(chosen)
    return {
        "prompt": prompt,
        "chosen": chosen[len(prompt):],
        "rejected": rejected[len(prompt):],
    }


def _normalize_explicit(example: dict) -> dict:
    return {
        "prompt": example["prompt"],
        "chosen": example["chosen"],
        "rejected": example["rejected"],
    }


def load_preference_dataset(
    name: str,
    split: str = "train",
    max_samples: int | None = None,
    num_proc: int | None = None,
):
    """Load + normalize a preference dataset to columns prompt/chosen/rejected."""
    from datasets import load_dataset

    ds = load_dataset(name, split=split)
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))

    cols = set(ds.column_names)
    if {"chosen", "rejected"} <= cols and "prompt" not in cols:
        fn = _normalize_hh
    elif {"prompt", "chosen", "rejected"} <= cols:
        fn = _normalize_explicit
    else:
        raise ValueError(
            f"Dataset '{name}' has columns {sorted(cols)}; expected HH-style "
            "(chosen/rejected) or explicit (prompt/chosen/rejected)."
        )
    ds = ds.map(fn, remove_columns=ds.column_names, num_proc=num_proc)
    ds = ds.filter(lambda ex: len(ex["chosen"]) > 0 and len(ex["rejected"]) > 0)
    return ds


def preference_dataset_from_pairs(pairs: Sequence[tuple[str, str, str]]):
    """Build a tiny in-memory preference dataset from (prompt, chosen, rejected)."""
    from datasets import Dataset

    data = {
        "prompt": [p for p, _, _ in pairs],
        "chosen": [c for _, c, _ in pairs],
        "rejected": [r for _, _, r in pairs],
    }
    return Dataset.from_dict(data)


def _pad_to(seqs: list[list[int]], pad_id: int, side: str = "right"):
    """Pad a list of token-id lists; return (input_ids, attention_mask) tensors."""
    maxlen = max(len(s) for s in seqs)
    input_ids, attn = [], []
    for s in seqs:
        pad = [pad_id] * (maxlen - len(s))
        if side == "right":
            input_ids.append(s + pad)
            attn.append([1] * len(s) + [0] * (maxlen - len(s)))
        else:
            input_ids.append(pad + s)
            attn.append([0] * (maxlen - len(s)) + [1] * len(s))
    return torch.tensor(input_ids, dtype=torch.long), torch.tensor(attn, dtype=torch.long)


@dataclass
class PreferenceCollator:
    """Collate (prompt, chosen, rejected) into padded chosen/rejected tensors.

    Used by the reward-model trainer: each side is the full prompt+response so
    the scalar head can score the complete sequence.
    """

    tokenizer: object
    max_length: int = 512

    def _encode(self, prompt: str, response: str) -> list[int]:
        ids = self.tokenizer(prompt + response, add_special_tokens=False)["input_ids"]
        return ids[: self.max_length]

    def __call__(self, batch: list[dict]) -> dict:
        pad_id = self.tokenizer.pad_token_id
        chosen = [self._encode(ex["prompt"], ex["chosen"]) for ex in batch]
        rejected = [self._encode(ex["prompt"], ex["rejected"]) for ex in batch]
        c_ids, c_attn = _pad_to(chosen, pad_id, side="right")
        r_ids, r_attn = _pad_to(rejected, pad_id, side="right")
        return {
            "chosen_input_ids": c_ids,
            "chosen_attention_mask": c_attn,
            "rejected_input_ids": r_ids,
            "rejected_attention_mask": r_attn,
        }


@dataclass
class DPOCollator:
    """Collate for DPO: per-side input_ids/attention_mask + a response loss_mask.

    The loss_mask is 1 over response tokens only (prompt + padding are 0) so the
    DPO objective sums log-probs over generated tokens.
    """

    tokenizer: object
    max_length: int = 512
    max_prompt_length: int = 256

    def _encode(self, prompt: str, response: str):
        p_ids = self.tokenizer(prompt, add_special_tokens=False)["input_ids"]
        r_ids = self.tokenizer(response, add_special_tokens=False)["input_ids"]
        p_ids = p_ids[-self.max_prompt_length:]
        r_ids = r_ids[: self.max_length - len(p_ids)]
        ids = p_ids + r_ids
        loss_mask = [0] * len(p_ids) + [1] * len(r_ids)
        return ids, loss_mask

    def _pad_side(self, seqs, masks, pad_id):
        maxlen = max(len(s) for s in seqs)
        ids_out, attn_out, lm_out = [], [], []
        for s, m in zip(seqs, masks):
            pad = maxlen - len(s)
            ids_out.append(s + [pad_id] * pad)
            attn_out.append([1] * len(s) + [0] * pad)
            lm_out.append(m + [0] * pad)
        return (
            torch.tensor(ids_out, dtype=torch.long),
            torch.tensor(attn_out, dtype=torch.long),
            torch.tensor(lm_out, dtype=torch.long),
        )

    def __call__(self, batch: list[dict]) -> dict:
        pad_id = self.tokenizer.pad_token_id
        c_seqs, c_masks, r_seqs, r_masks = [], [], [], []
        for ex in batch:
            ci, cm = self._encode(ex["prompt"], ex["chosen"])
            ri, rm = self._encode(ex["prompt"], ex["rejected"])
            c_seqs.append(ci); c_masks.append(cm)
            r_seqs.append(ri); r_masks.append(rm)
        c_ids, c_attn, c_lm = self._pad_side(c_seqs, c_masks, pad_id)
        r_ids, r_attn, r_lm = self._pad_side(r_seqs, r_masks, pad_id)
        return {
            "chosen_input_ids": c_ids, "chosen_attention_mask": c_attn, "chosen_loss_mask": c_lm,
            "rejected_input_ids": r_ids, "rejected_attention_mask": r_attn, "rejected_loss_mask": r_lm,
        }
