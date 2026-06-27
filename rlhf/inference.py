"""Chat inference: load a trained policy, generate, optionally Best-of-N rerank by a reward model.

Shared by the CLI (`scripts/chat.py`) and the web UI (`app.py`)."""

from __future__ import annotations

import torch
from transformers import GenerationConfig

from .models import RewardModel, load_causal_lm, load_tokenizer
from .utils import resolve_device, resolve_dtype
from .utils.generation import score_texts


class ChatEngine:
    """Wraps a policy (any local checkpoint or HF id) for multi-turn chat.

    If ``best_of_n > 1`` and a ``reward_model`` path is given, each turn samples N candidate
    replies and returns the one the reward model scores highest (inference-time alignment)."""

    def __init__(self, model_id, reward_model=None, best_of_n=1, device="auto", dtype="auto"):
        self.device = resolve_device(device)
        dt = resolve_dtype(dtype, self.device)
        self.model = load_causal_lm(model_id, dtype=dt).to(self.device).eval()
        self.tok = load_tokenizer(model_id)
        self.best_of_n = max(1, int(best_of_n))
        self.rm = None
        if self.best_of_n > 1 and reward_model:
            self.rm = RewardModel.from_pretrained(reward_model, dtype=dt).to(self.device).eval()

    def _input_ids(self, messages):
        if getattr(self.tok, "chat_template", None):
            return self.tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        text = "".join(f"{m['role']}: {m['content']}\n" for m in messages) + "assistant:"
        return self.tok(text, return_tensors="pt").input_ids

    @torch.no_grad()
    def reply(self, messages, max_new_tokens=256, temperature=0.7, top_p=0.9,
              greedy=False, max_length=1024):
        """Return (reply_text, info). info is None unless Best-of-N reranking ran."""
        input_ids = self._input_ids(messages).to(self.device)
        attn = torch.ones_like(input_ids)
        n = self.best_of_n
        pad = self.tok.pad_token_id if self.tok.pad_token_id is not None else self.tok.eos_token_id
        gc = GenerationConfig(
            max_new_tokens=max_new_tokens, do_sample=(n > 1 or not greedy),
            temperature=temperature, top_p=top_p, num_return_sequences=n,
            pad_token_id=pad, eos_token_id=self.tok.eos_token_id)
        out = self.model.generate(input_ids, attention_mask=attn, generation_config=gc)
        cands = [self.tok.decode(r[input_ids.size(1):], skip_special_tokens=True).strip() for r in out]
        if n == 1 or self.rm is None:
            return cands[0], None
        ctx = self.tok.decode(input_ids[0], skip_special_tokens=True)
        scores = score_texts(self.rm, self.tok, [ctx] * n, cands, self.device, max_length=max_length)
        best = int(scores.argmax())
        return cands[best], {"reward": scores[best].item(), "mean": scores.mean().item(), "n": n}
