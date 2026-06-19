"""Optional vLLM-backed generation for PPO/GRPO rollouts.

Rollout generation (autoregressive sampling) dominates RL wall-clock; vLLM's
paged-attention engine is far faster than HF ``.generate``. This wrapper handles
the two awkward parts: (1) **weight sync** — the policy changes every PPO step,
so vLLM's copy must be refreshed; (2) **reconstruction** — turning vLLM's
per-prompt outputs back into a left-padded ``[prompt][response]`` batch with an
exact response mask, aligned with the HF training-forward path.

EXPERIMENTAL: requires a CUDA GPU + ``pip install vllm`` and was not runnable in
the dev environment (Apple Silicon, no CUDA). The caller (PPOTrainer/GRPOTrainer)
falls back to HF generation if construction, weight-sync, or generation fails, so
enabling ``use_vllm`` can never break a run — at worst it logs a warning and uses
the HF path.
"""

from __future__ import annotations

import torch

from .common import get_logger

_log = get_logger("rlhf.vllm")


class VLLMGenerator:
    def __init__(self, model_name_or_path: str, tokenizer, dtype: str = "auto",
                 max_model_len: int | None = None, gpu_memory_utilization: float = 0.4):
        from vllm import LLM, SamplingParams  # noqa: F401 (import-time availability check)

        self.tokenizer = tokenizer
        self.SamplingParams = SamplingParams
        self.llm = LLM(
            model=model_name_or_path,
            dtype=dtype if dtype != "auto" else "auto",
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            enforce_eager=True,        # avoids CUDA-graph capture stalls during frequent weight sync
            disable_log_stats=True,
        )

    def _model_runner(self):
        # The internal handle differs across vLLM versions; try the common paths.
        engine = self.llm.llm_engine
        for attr in ("model_executor", "engine"):
            ex = getattr(engine, attr, None)
            if ex is not None:
                worker = getattr(ex, "driver_worker", None) or getattr(ex, "worker", None)
                runner = getattr(worker, "model_runner", None) if worker else None
                model = getattr(runner, "model", None) if runner else None
                if model is not None:
                    return model
        raise RuntimeError("could not locate vLLM model runner for weight sync")

    @torch.no_grad()
    def sync_weights(self, hf_lm) -> None:
        """Load current HF policy weights into the vLLM engine (best-effort)."""
        model = getattr(hf_lm, "merge_and_unload", None)
        src = hf_lm.merge_and_unload() if (model and getattr(hf_lm, "peft_config", None)) else hf_lm
        state = ((n, p.detach()) for n, p in src.named_parameters())
        self._model_runner().load_weights(state)

    @torch.no_grad()
    def generate_sequences(self, prompt_ids: torch.Tensor, prompt_attn: torch.Tensor,
                           max_new_tokens: int, temperature: float, top_p: float, top_k: int):
        """Return (full_ids [B, P+G], resp_mask [B, G]) aligned with the HF path.

        ``prompt_ids`` is the left-padded prompt block of width P; responses are
        appended right-padded to a common width G. The response mask comes from
        the exact generated lengths (EOS appended for naturally-finished rows).
        """
        device = prompt_ids.device
        eos_id, pad_id = self.tokenizer.eos_token_id, self.tokenizer.pad_token_id
        token_id_lists = [
            prompt_ids[i][prompt_attn[i].bool()].tolist() for i in range(prompt_ids.size(0))
        ]
        sp = self.SamplingParams(
            n=1, temperature=temperature, top_p=top_p,
            top_k=top_k if top_k and top_k > 0 else -1, max_tokens=max_new_tokens,
        )
        outs = self.llm.generate(prompt_token_ids=token_id_lists, sampling_params=sp, use_tqdm=False)

        responses = []
        for o in outs:
            comp = o.outputs[0]
            ids = list(comp.token_ids)
            if comp.finish_reason == "stop" and (not ids or ids[-1] != eos_id):
                ids.append(eos_id)              # make the stop token explicit for KL/credit
            responses.append(ids[:max_new_tokens])

        G = max((len(r) for r in responses), default=1)
        resp_tensor, mask = [], []
        for r in responses:
            pad = G - len(r)
            resp_tensor.append(r + [pad_id] * pad)
            mask.append([1.0] * len(r) + [0.0] * pad)
        resp_tensor = torch.tensor(resp_tensor, dtype=torch.long, device=device)
        full_ids = torch.cat([prompt_ids, resp_tensor], dim=1)
        return full_ids, torch.tensor(mask, dtype=torch.float, device=device)


def try_build_vllm(model_name_or_path, tokenizer, **kwargs):
    """Construct a VLLMGenerator, returning None (with a warning) on any failure."""
    try:
        gen = VLLMGenerator(model_name_or_path, tokenizer, **kwargs)
        _log.info("vLLM rollout backend enabled (%s)", model_name_or_path)
        return gen
    except Exception as e:  # noqa: BLE001
        _log.warning("vLLM unavailable (%s); falling back to HF generation.", e)
        return None
