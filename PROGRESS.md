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
- [ ] M1  Scaffold: configs, requirements, README, .gitignore
- [ ] M2  Utils: config loader, logging/metrics, tensor/RL ops, generation helpers
- [ ] M3  Data layer: preference pairs, prompt-only, SFT datasets + collators
- [ ] M4  Models: reward model (scalar head), actor-critic policy (value head), LoRA
- [ ] M5  Reward-model trainer (BT loss + eval accuracy) + script
- [ ] M6  SFT trainer + script
- [ ] M7  PPO trainer from scratch + script
- [ ] M8  DPO trainer + GRPO trainer + scripts
- [ ] M9  Eval + generation utilities + end-to-end CPU smoke test (must pass)
- [ ] M10 Kaggle notebook + README finalization
- [ ] M11 Unit tests (logprobs, GAE, BT loss, RM forward, masking) green

## Notes / decisions log
- 2026-06-19: durable cron flag did not persist to disk in this harness build;
  heartbeat is in-memory (survives usage resets while Claude Code stays open).
  Checkpointing (this file + commits) is the real cross-session insurance.
