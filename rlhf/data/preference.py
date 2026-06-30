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

from ..utils.common import get_logger

_log = get_logger("rlhf.data")
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


def _last_assistant(messages) -> str:
    """Final assistant turn from a chat-message list (e.g. UltraFeedback)."""
    for m in reversed(messages):
        if isinstance(m, dict) and m.get("role") == "assistant":
            return m.get("content", "")
    return messages[-1].get("content", "") if messages else ""


def _normalize_messages(example: dict) -> dict:
    return {
        "prompt": example.get("prompt", ""),
        "chosen": _last_assistant(example["chosen"]),
        "rejected": _last_assistant(example["rejected"]),
    }


def _first_user(messages) -> str:
    """First user turn from a chat-message list."""
    for m in messages:
        if isinstance(m, dict) and m.get("role") == "user":
            return m.get("content", "")
    return messages[0].get("content", "") if messages else ""


def _normalize_messages_no_prompt(example: dict) -> dict:
    """chosen/rejected are full chat-message lists with NO separate prompt column
    (e.g. Skywork-Reward-Preference-80K). The shared user turn is the prompt; each
    side's final assistant turn is its response."""
    return {
        "prompt": _first_user(example["chosen"]),
        "chosen": _last_assistant(example["chosen"]),
        "rejected": _last_assistant(example["rejected"]),
    }


def _is_true(v) -> bool:
    return v in (True, 1, "True", "true", "1")


def _normalize_pku(example: dict) -> dict:
    """PKU-SafeRLHF: prompt + response_0/response_1 + safety/helpfulness annotations.

    Safety-AWARE but helpfulness-preserving labeling (avoids teaching over-refusal):
      * exactly one response unsafe -> chosen = the SAFE one  (learn to refuse real harm)
      * both safe                   -> chosen = the more HELPFUL one (``better_response_id``)
                                       (don't prefer the refusal when nothing is wrong)
      * both unsafe                 -> dropped (no clean signal)
    """
    s0, s1 = _is_true(example.get("is_response_0_safe")), _is_true(example.get("is_response_1_safe"))
    if s0 != s1:
        cid = 0 if s0 else 1
    elif s0 and s1:                                   # both safe -> prefer helpfulness
        try:
            cid = int(example["better_response_id"])
        except (TypeError, ValueError, KeyError):
            cid = -1
    else:                                             # both unsafe -> no clean preference
        cid = -1
    if cid not in (0, 1):
        return {"prompt": "", "chosen": "", "rejected": ""}   # filtered out downstream
    return {
        "prompt": example["prompt"],
        "chosen": example[f"response_{cid}"],
        "rejected": example[f"response_{1 - cid}"],
    }


def pair_similarity(a: str, b: str) -> float:
    """Token-set Jaccard similarity of two responses (0 = disjoint, 1 = identical)."""
    sa, sb = set(a.split()), set(b.split())
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / (len(sa | sb) or 1)


def _embedding_keep_mask(chosen, rejected, max_cosine, model_name, batch_size=128, max_length=256):
    """Keep-mask for pairs whose chosen/rejected EMBEDDING cosine <= max_cosine.

    Semantic contrast (mean-pooled MiniLM via transformers — no extra dependency).
    Dropping the most-similar (low-contrast) pairs can raise small-model RM accuracy
    (arXiv:2409.09603); on HH the effect is modest (pairs are already contrastive).
    """
    import torch
    from transformers import AutoModel, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    enc = AutoModel.from_pretrained(model_name).to(device).eval()

    @torch.no_grad()
    def embed(texts):
        chunks = []
        for i in range(0, len(texts), batch_size):
            b = tok(list(texts[i:i + batch_size]), padding=True, truncation=True,
                    max_length=max_length, return_tensors="pt").to(device)
            h = enc(**b).last_hidden_state
            m = b["attention_mask"].unsqueeze(-1).float()
            e = (h * m).sum(1) / m.sum(1).clamp_min(1e-9)        # mean pool
            chunks.append(torch.nn.functional.normalize(e, dim=-1).cpu())
        return torch.cat(chunks)

    cos = (embed(chosen) * embed(rejected)).sum(-1)
    return (cos <= max_cosine).tolist()


def load_preference_dataset(
    name: str,
    split: str = "train",
    max_samples: int | None = None,
    num_proc: int | None = None,
    max_pair_similarity: float = 1.0,
    contrast_metric: str = "jaccard",
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
):
    """Load + normalize a preference dataset to columns prompt/chosen/rejected.

    ``max_pair_similarity`` < 1 drops low-contrast chosen/rejected pairs. With
    ``contrast_metric="jaccard"`` (cheap, lexical — weak on HH) it's token overlap;
    with ``"embedding"`` it's mean-pooled MiniLM cosine (semantic; on HH use ~0.6).
    Dropping low-contrast pairs can raise small-model RM accuracy (arXiv:2409.09603).
    Filtering runs before ``max_samples`` so the cap is taken from the cleaned pool.
    """
    from datasets import concatenate_datasets, load_dataset

    def _load_normalized(nm, sp):
        d = load_dataset(nm, split=sp)
        cols = set(d.column_names)
        if {"chosen", "rejected"} <= cols and "prompt" not in cols:
            # string transcripts (HH) vs chat-message lists with no prompt column (Skywork)
            fn = _normalize_messages_no_prompt if isinstance(d[0]["chosen"], list) else _normalize_hh
        elif {"prompt", "chosen", "rejected"} <= cols:
            # message-list format (e.g. UltraFeedback) vs plain strings (e.g. rm-static)
            fn = _normalize_messages if isinstance(d[0]["chosen"], list) else _normalize_explicit
        elif {"response_0", "response_1", "safer_response_id"} <= cols:
            fn = _normalize_pku                              # PKU-SafeRLHF (safety preferences)
        else:
            raise ValueError(
                f"Dataset '{nm}' has columns {sorted(cols)}; expected HH-style "
                "(chosen/rejected), explicit (prompt/chosen/rejected), chat-message lists, "
                "or PKU-SafeRLHF (response_0/response_1/safer_response_id)."
            )
        d = d.map(fn, remove_columns=d.column_names, num_proc=num_proc)
        return d.filter(lambda ex: len(ex["chosen"]) > 0 and len(ex["rejected"]) > 0)

    # `name` may be a comma-separated MIX of sources, each optionally "name:split"
    # (data curation/mixing — e.g. cleaned UltraFeedback + Skywork-Reward-80K).
    parts = []
    for spec in str(name).split(","):
        spec = spec.strip()
        if not spec:
            continue
        nm, _, sp = spec.partition(":")
        parts.append(_load_normalized(nm.strip(), sp.strip() or split))
    ds = parts[0] if len(parts) == 1 else concatenate_datasets(parts)
    if len(parts) > 1:
        _log.info("mixed %d preference sources -> %d pairs", len(parts), len(ds))

    # Bound embedding cost by pre-capping to a pool ~3x the target before encoding.
    if max_samples is not None and max_pair_similarity < 1.0 and contrast_metric == "embedding":
        ds = ds.select(range(min(len(ds), max_samples * 3)))

    if max_pair_similarity < 1.0:
        if contrast_metric == "embedding":
            try:
                keep = _embedding_keep_mask(ds["chosen"], ds["rejected"], max_pair_similarity, embedding_model)
                ds = ds.select([i for i, k in enumerate(keep) if k])
                _log.info("embedding contrast filter kept %d/%d pairs (cosine<=%.2f)",
                          len(ds), len(keep), max_pair_similarity)
            except Exception as e:  # noqa: BLE001 — never let filtering break a run
                _log.warning("embedding contrast filter failed (%s); skipping", e)
        else:
            ds = ds.filter(
                lambda ex: pair_similarity(ex["chosen"], ex["rejected"]) <= max_pair_similarity,
                num_proc=num_proc,
            )
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))
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
    emit_loss_mask: bool = False   # also emit chosen_loss_mask (response-only) for the GRM aux LM loss

    def _encode(self, prompt: str, response: str) -> list[int]:
        ids = self.tokenizer(prompt + response, add_special_tokens=False)["input_ids"]
        return ids[: self.max_length]

    def _prompt_len(self, prompt: str) -> int:
        return len(self.tokenizer(prompt, add_special_tokens=False)["input_ids"])

    def __call__(self, batch: list[dict]) -> dict:
        pad_id = self.tokenizer.pad_token_id
        chosen = [self._encode(ex["prompt"], ex["chosen"]) for ex in batch]
        rejected = [self._encode(ex["prompt"], ex["rejected"]) for ex in batch]
        c_ids, c_attn = _pad_to(chosen, pad_id, side="right")
        r_ids, r_attn = _pad_to(rejected, pad_id, side="right")
        out = {
            "chosen_input_ids": c_ids,
            "chosen_attention_mask": c_attn,
            "rejected_input_ids": r_ids,
            "rejected_attention_mask": r_attn,
        }
        if self.emit_loss_mask:
            # 1 over chosen-response tokens only (prompt + padding -> 0). The boundary is the
            # prompt's token count, capped to the (possibly truncated) real sequence length.
            T = c_ids.size(1)
            masks = []
            for ex, ids in zip(batch, chosen):
                plen = min(self._prompt_len(ex["prompt"]), len(ids))
                masks.append([0] * plen + [1] * (len(ids) - plen) + [0] * (T - len(ids)))
            out["chosen_loss_mask"] = torch.tensor(masks, dtype=torch.long)
        return out


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
