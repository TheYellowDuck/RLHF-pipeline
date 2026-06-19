"""Inference helpers: batched generation and reward scoring for evaluation."""

from __future__ import annotations

import torch
from transformers import GenerationConfig

from ..data import PromptCollator


@torch.no_grad()
def generate_responses(policy, tokenizer, prompts, device, max_new_tokens=64,
                       temperature=1.0, top_p=1.0, top_k=0, do_sample=True,
                       max_prompt_length=256, batch_size=8):
    """Generate a completion per prompt. `policy` is an ActorCriticPolicy.

    Returns a list of decoded response strings (prompt stripped)."""
    coll = PromptCollator(tokenizer, max_prompt_length=max_prompt_length)
    gc = GenerationConfig(
        max_new_tokens=max_new_tokens, do_sample=do_sample, temperature=temperature,
        top_p=top_p, top_k=top_k, pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id)
    out = []
    for i in range(0, len(prompts), batch_size):
        chunk = prompts[i: i + batch_size]
        batch = coll([{"prompt": p} for p in chunk])
        ids = batch["input_ids"].to(device)
        attn = batch["attention_mask"].to(device)
        seqs = policy.generate(ids, attn, gc)
        for row in seqs[:, ids.size(1):]:
            out.append(tokenizer.decode(row, skip_special_tokens=True))
    return out


@torch.no_grad()
def score_texts(reward_model, tokenizer, prompts, responses, device, max_length=512, batch_size=8):
    """Reward-model score for each (prompt+response). Returns a 1-D tensor."""
    prev_side = tokenizer.padding_side
    tokenizer.padding_side = "right"  # RewardModel scores the last non-pad token
    scores = []
    for i in range(0, len(prompts), batch_size):
        texts = [p + r for p, r in zip(prompts[i: i + batch_size], responses[i: i + batch_size])]
        enc = tokenizer(texts, return_tensors="pt", padding=True, truncation=True,
                        max_length=max_length, add_special_tokens=False)
        enc = {k: v.to(device) for k, v in enc.items()}
        scores.append(reward_model(enc["input_ids"], enc["attention_mask"]).float().cpu())
    tokenizer.padding_side = prev_side
    return torch.cat(scores) if scores else torch.zeros(0)
