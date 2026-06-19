# PROGRESS — RLHF pipeline (resume state)

This file is the **source of truth** for autonomous continuation. A heartbeat cron
(and any fresh session) reads this to know what is done and what to do next.

## Resume protocol
1. `cd /Users/georgezhang/RLHF-pipeline`
2. Read this file + `git log --oneline -15`.
3. Do the **first unchecked** milestone below. Keep changes small + verified.
4. Update this file, then `git add -A && git commit -m "checkpoint: <step>"` (local only).
5. If everything is checked: stop, print a one-line status. Don't invent new scope.

## Environment
- Local: Apple Silicon Mac, **no NVIDIA GPU**, Python 3.11 venv at `.venv` (torch+MPS/CPU).
  Local is for **smoke tests only** (tiny models on CPU/MPS).
- Real training: **Kaggle** free GPU (T4 x2 / P100) via `notebooks/kaggle_rlhf.ipynb`.
- Activate venv: `source .venv/bin/activate` (or call `./.venv/bin/python`).

## Design (locked)
Reward Model (Bradley-Terry preference loss) + **PPO from scratch** (GAE, clipped
surrogate, per-token KL-to-reference penalty, value head, adaptive KL controller).
Plus **DPO** and **GRPO** as alternative/modern post-training methods. SFT precursor.
Backbones via HF `transformers`; all RL/RM logic hand-written. LoRA via `peft`.
Default smoke model: tiny GPT-2. Kaggle default: `Qwen/Qwen2.5-0.5B` or `EleutherAI/pythia-410m`.
Default preference data: `Anthropic/hh-rlhf` (+ `Dahoas/rm-static` fallback).

## Milestones
- [x] M0  Heartbeat cron + PROGRESS.md + git repo + venv/deps install
- [x] M1  Scaffold: configs, requirements, README(skel), .gitignore
- [x] M2  Utils: config loader, logging/metrics, tensor/RL ops
- [x] M3  Data layer: preference pairs, prompt-only, SFT datasets + collators
- [x] M4  Models: reward model (scalar head), actor-critic policy (value head), LoRA
        + integration test PASSES on tiny-gpt2 (transformers v5, MPS box)
- [x] M5  Reward-model trainer (BT loss + eval accuracy)  [written; smoke-tested in M9]
- [x] M6  SFT trainer  [written; smoke-tested in M9]
- [x] M7  PPO trainer from scratch (GAE, KL shaping, clip, value, entropy, adaptive KL)
- [x] M8  DPO trainer + GRPO trainer
- [x] M9  Eval + generation utilities + end-to-end CPU smoke test (PASSES all 5 stages)
- [x] M10 CLI scripts (train_*/evaluate) + Kaggle notebook + README + pyproject
- [x] M11 Unit tests green (11 passed); reward-model CLI verified on real HH-RLHF

## Core pipeline: COMPLETE ✅ (M0-M11). Now in ENHANCEMENTS phase (user: "all recommend or more").

## Enhancements (live work)
- [x] E0  Fix: reward model scored wrong token under left-padded prompts (PPO/GRPO)
- [x] E1  PPO stability pack: running reward normalization + length/EOS penalties + richer KL/reward logging
- [x] E2  LLM-judge eval (Claude API via `anthropic` SDK, model claude-opus-4-8): independent win-rate,
        position-bias control, graceful no-key fallback, mock-tested parsing
- [x] E3  Gradient-checkpointing flag wired through trainers (memory on Kaggle)
- [x] E4  accelerate multi-GPU (DDP) for supervised trainers (SFT/RM/DPO); verify single-process smoke
- [x] E5  vLLM rollout backend behind a flag with HF fallback (untestable locally; documented)
- [x] E6  Re-run smoke + unit tests, update README/PROGRESS, commit

After each enhancement: smoke_test.py + pytest must stay green; commit a checkpoint.

## ENHANCEMENTS COMPLETE ✅ (E0-E6). 17 unit tests + 5-stage smoke + accelerate + learning checks green.

## Polish (F) — live work (Kaggle run is the user's, done last & separately)
- [x] F1  Checkpoint resume for PPO/GRPO (optimizer + global_step + KL-coef + reward-norm state)
- [ ] F2  Tokenizer-mismatch guard in PPO/GRPO (RM vs policy vocab size)   <-- NEXT
- [ ] F3  DPO length-normalization option (per-token-averaged logps; curbs length bias)
- [ ] F4  Toy-reward PPO learning test (closed-form reward must increase / target-token prob rises)
- [ ] F5  Re-run smoke + tests, update README/PROGRESS, commit

## Notes / decisions log
- 2026-06-19: durable cron flag did not persist to disk in this harness build;
  heartbeat is in-memory (survives usage resets while Claude Code stays open).
  Checkpointing (this file + commits) is the real cross-session insurance.
