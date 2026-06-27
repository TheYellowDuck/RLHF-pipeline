# RLHF Pipeline — Reward Modeling + PPO from Scratch

An end-to-end **Reinforcement Learning from Human Feedback (RLHF)** pipeline that reproduces the
post-training recipe behind aligned large language models — **supervised fine-tuning → reward
modeling → PPO** — with the reinforcement-learning core written **from scratch** in **PyTorch**:
generalized advantage estimation (GAE), the clipped policy-gradient surrogate, a per-token
KL-to-reference penalty, and an adaptive KL controller. It also implements **DPO** and **GRPO**
as modern alternatives, plus LoRA parameter-efficient fine-tuning, multi-GPU training, and an
independent **LLM-as-judge** evaluation. Built on **HuggingFace Transformers** for the backbone
models and tokenizers; every reward-modeling and RL component is hand-implemented, not wrapped
from a higher-level library.

```text
        pretrained LM
              │
              ▼
   ┌──────────────────────┐   demonstrations
   │  1. SFT              │ ◄──────────────────  (chosen transcripts)
   └──────────┬───────────┘
              │  policy π_sft  (also the reference π_ref)
              ▼
   ┌──────────────────────┐   preference pairs (chosen ≻ rejected)
   │  2. Reward Model     │ ◄──────────────────  Bradley-Terry loss
   └──────────┬───────────┘
              │  reward r_φ(x, y)
              ▼
   ┌──────────────────────┐   on-policy rollouts, scored by r_φ
   │  3a. PPO  (RL)       │   advantage = GAE, KL(π‖π_ref) penalty
   │  3b. GRPO (RL)       │   group-relative advantage, critic-free
   └──────────────────────┘
                ▲
                │   …or skip the RM+RL entirely with
   ┌──────────────────────┐
   │  DPO                 │   implicit reward, preference pairs only
   └──────────────────────┘
```

## Features

- **Three alignment methods, one codebase** — PPO (reward-model-based RL), DPO (reward-model-free), and GRPO (critic-free, DeepSeek-style).
- **PPO written from scratch** — on-policy rollouts, GAE, the clipped surrogate objective, clipped value loss, entropy bonus, per-token KL-to-reference reward shaping, and an adaptive KL controller.
- **Reward model** — a scalar reward head on a pretrained trunk, trained with the Bradley-Terry pairwise preference loss, scored at the last real token (robust to left/right padding).
- **Supervised fine-tuning** — teacher forcing with prompt-masked labels as the policy initializer and PPO reference.
- **Parameter-efficient fine-tuning** — LoRA adapters everywhere via PEFT; the frozen RL reference is recovered by disabling the adapter (no second copy of weights).
- **Scales out** — multi-GPU data parallelism via HuggingFace Accelerate (DDP-ready) and gradient checkpointing for the supervised trainers.
- **RL stability knobs** — running reward normalization, length and missing-EOS penalties to curb reward hacking, and KL-vs-reward logging.
- **Checkpoint resume** — optimizer, global step, and KL-controller state save and restore for long runs.
- **Independent evaluation** — an LLM-as-judge win-rate computed via the Anthropic Claude API with position-bias control, alongside reward-model accuracy.
- **Optional fast rollouts** — a flag-gated vLLM generation backend that auto-falls back to HuggingFace generation if unavailable.
- **Validated to actually optimize** — unit tests for the RL/RM math plus an end-to-end harness that proves the reward model *learns* a separable signal and a PPO step *increases* the log-prob of positive-advantage tokens.
- **Laptop-to-GPU** — runs a tiny end-to-end smoke test on a CPU in seconds; trains real models on a free Kaggle GPU.

## How It Works

The repository mirrors the standard post-training stack. The RL and reward-modeling logic
(GAE, the PPO surrogate, KL shaping, the Bradley-Terry / DPO / GRPO objectives) is implemented
by hand; HuggingFace Transformers supplies only the pretrained backbones and tokenizers, and
PEFT supplies optional LoRA.

- **Reward model.** A scalar head reads the trunk's last non-pad hidden state and is trained with `−log σ(r_chosen − r_rejected)`, the maximum-likelihood objective of the Bradley-Terry preference model.
- **PPO reward shaping.** Each response token's reward is a KL-to-reference penalty `−β·(log π − log π_ref)`; the scalar reward-model score is added at the final token. β is adapted toward a target KL, which keeps the policy from drifting off-distribution while chasing reward (reward hacking).
- **Advantages.** GAE(γ, λ) over the value head's per-token estimates, then whitened. Token/value alignment is handled explicitly: the log-prob and value of response token *j* come from the model output at position *j−1*.
- **DPO.** Optimizes the same Bradley-Terry preference likelihood but with the *implicit* reward `β·log(π/π_ref)`, removing the separate reward model and RL loop. Supports sigmoid / IPO / hinge losses and optional length normalization.
- **GRPO.** Replaces the value critic with a group baseline — sample *G* responses per prompt, advantage = (reward − group mean) / group std — with an unbiased k3 KL penalty. Cheaper and stable; the DeepSeek-R1 recipe.

**Engineering notes.** Numerically careful throughout: log-probs and entropy computed from
logits, masked whitening of advantages, Welford running moments for reward normalization, and
correct masking for variable-length, padded sequences. Development surfaced a genuine
correctness bug — the reward model was scoring a *prompt* token instead of the response under
left-padded prompts (which PPO/GRPO always produce) — caught by reasoning about token alignment
and locked down with a regression test.

## Architecture

```text
rlhf/
  data/         preference / prompt / SFT datasets + collators (correct padding sides)
  models/       reward model, actor-critic policy, value head, loaders (LoRA, transformers-v5-safe)
  algorithms/   reward / sft / ppo / dpo / grpo trainers
  utils/        config, metrics, generation, RL/tensor math (GAE, log-probs), running moments, vLLM
  eval/         LLM-as-judge (Claude API)
  cli.py        shared argparse + logger plumbing
scripts/        train_*.py, evaluate.py, smoke_test.py
configs/        one YAML per stage
tests/          fast unit tests for the math
notebooks/      Kaggle GPU runner
```

## Status & Validation

The pipeline is validated at three levels: it **runs** (a 5-stage CPU smoke test exercises
RM → SFT → PPO → DPO → GRPO with checkpoint save/load + resume and the Accelerate path), it
**optimizes** (the reward model learns a separable preference signal to 100% accuracy, and a
single PPO update provably raises the log-prob of positive-advantage tokens — a deterministic
policy-gradient check), and it is **static-clean** (17 unit tests, `ruff`/`pyflakes` with no
undefined names across every branch). Reward-model accuracy parsing was verified on the real
`Anthropic/hh-rlhf` dataset.

This is an educational, faithful single-GPU reproduction of the algorithms, not a
throughput-optimized training stack: it does not include FSDP/ZeRO sharding or a fully verified
distributed-rollout path.

**Headline GPU result (Qwen2.5-0.5B, free Kaggle T4).** The reward model reaches **0.726**
held-out preference accuracy on cleaned UltraFeedback — and getting there was a diagnosis, not just
a bigger run. Accuracy was stuck at ~0.63 regardless of epochs/label-smoothing/filtering; I traced
the ceiling to *label noise*, not capacity: `HuggingFaceH4/ultrafeedback_binarized` mislabels ~50% of
pairs via a known-buggy `overall_score` binarization. Training + evaluating on the re-binarized
`argilla/ultrafeedback-binarized-preferences-cleaned` set and initializing the RM from an instruct
backbone lifted accuracy to **0.726 (+9.6 pts)** — while the *same* model drops to 0.59 on the noisy H4
labels, which confirms the diagnosis (a more-correct model disagrees more with bad labels). PPO then
optimizes the Instruct policy against this reward model and **wins 56% of head-to-head comparisons vs
the un-tuned policy** (mean reward −0.654 → −0.587), i.e. RL measurably moved the policy in the
reward-increasing direction. Full run: `notebooks/kaggle_rlhf_full.ipynb`.

## Skills Demonstrated

- Reinforcement Learning from Human Feedback (RLHF) — full supervised-fine-tuning → reward-model → PPO post-training loop
- Proximal Policy Optimization (PPO) — from-scratch actor-critic with GAE, clipped surrogate, clipped value loss, and entropy bonus
- Policy-gradient methods — on-policy rollouts, advantage estimation, importance-ratio clipping, KL-to-reference regularization
- Reward modeling — Bradley-Terry pairwise preference loss with a scalar reward head
- Generalized Advantage Estimation (GAE) — per-token advantage/return computation with masked, padded sequences
- Adaptive KL control — Schulman-style proportional controller targeting a fixed KL budget
- Direct Preference Optimization (DPO) — implicit-reward Bradley-Terry objective with sigmoid/IPO/hinge losses and length normalization
- Group Relative Policy Optimization (GRPO) — critic-free, group-relative advantages with a k3 KL estimator (DeepSeek-style)
- Supervised fine-tuning (SFT) — teacher forcing with prompt-masked labels
- Deep learning with PyTorch — hand-written training loops, autograd, mixed-precision autocast
- HuggingFace Transformers — pretrained backbones and tokenizers (transformers v5 compatible)
- Parameter-efficient fine-tuning — LoRA adapters via PEFT and an adapter-disable reference trick
- Distributed / multi-GPU training — data parallelism via HuggingFace Accelerate (DDP) with gradient checkpointing
- Numerical stability — log-prob/entropy from logits, masked whitening, Welford running moments
- LLM-as-judge evaluation — independent win-rate via the Anthropic Claude API with position-bias control
- Reward-hacking mitigation — running reward normalization plus length and missing-EOS penalties
- Checkpointing and resumability — optimizer / step / KL-controller state persistence
- Unit testing and validation harnesses — RL/RM math tests plus optimization-proof learning checks
- Software design — modular Python package (data / models / algorithms / utils), YAML configs, CLI scripts
- Reproducibility and static analysis — seeded runs, deterministic checks, `ruff` / `pyflakes` clean

## Tech Stack

- Python 3.11
- PyTorch — autograd, custom training loops, mixed precision
- HuggingFace Transformers — backbone models and tokenizers
- HuggingFace Datasets — preference / SFT data loading (`Anthropic/hh-rlhf`)
- HuggingFace Accelerate — multi-GPU data parallelism (DDP)
- PEFT — LoRA parameter-efficient fine-tuning
- NumPy, tqdm, PyYAML, TensorBoard
- Anthropic Claude API — LLM-as-judge evaluation
- vLLM — optional fast generation backend for rollouts
- pytest, ruff — testing and static analysis
- Kaggle — free-GPU training (T4 / P100)

## Getting Started

### Install

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

(Apple-Silicon / CPU works for smoke tests; CUDA recommended for real runs.)

### 30-second sanity check

Runs the whole pipeline — RM → SFT → PPO → DPO → GRPO — on a tiny random GPT-2 with synthetic
data, on CPU, in a few seconds. Verifies tensor alignment, masking, checkpoint save/load +
resume, the Accelerate path, and that things actually **optimize** (the RM learns a separable
signal to 1.0 accuracy; a PPO step provably raises the log-prob of positive-advantage tokens):

```bash
python scripts/smoke_test.py
python -m pytest tests/ -q          # 17 fast unit tests for the RL/RM math
```

### Train the full recipe

Defaults use the `Anthropic/hh-rlhf` preferences. Pick a small base model that fits your GPU
(e.g. `gpt2`, `EleutherAI/pythia-410m`, `Qwen/Qwen2.5-0.5B`).

```bash
# 1. Supervised fine-tuning (policy init + PPO reference)
python scripts/train_sft.py -o model.name_or_path=Qwen/Qwen2.5-0.5B -o output_dir=checkpoints/sft

# 2. Reward model
python scripts/train_reward_model.py -o model.name_or_path=Qwen/Qwen2.5-0.5B -o output_dir=checkpoints/reward_model

# 3a. PPO against the reward model
python scripts/train_ppo.py -o policy.name_or_path=checkpoints/sft \
    -o reward_model.name_or_path=checkpoints/reward_model

# 3b. …or GRPO (critic-free) against the same reward model
python scripts/train_grpo.py -o policy.name_or_path=checkpoints/sft \
    -o reward_model.name_or_path=checkpoints/reward_model

# DPO alternative — no reward model, no RL loop
python scripts/train_dpo.py -o model.name_or_path=checkpoints/sft
```

Every script takes `--config <yaml>`, repeatable `-o key.sub=value` overrides, and
`--report-to {none,tensorboard,wandb}`. Metrics also stream to `<output_dir>/metrics.jsonl`.
Configs live in [configs/](configs/).

### Scaling, resume, and speed

```bash
# LoRA (fits bigger models on small GPUs)
python scripts/train_ppo.py -o policy.use_lora=true -o policy.name_or_path=checkpoints/sft \
    -o reward_model.name_or_path=checkpoints/reward_model

# Activation checkpointing (less memory) — SFT/RM/DPO
python scripts/train_sft.py -o train.gradient_checkpointing=true ...

# Multi-GPU data parallelism (DDP) for the supervised trainers
accelerate launch scripts/train_sft.py --accelerate ...

# PPO stability knobs (curb reward hacking / length exploitation)
python scripts/train_ppo.py -o ppo.normalize_rewards=true \
    -o ppo.length_penalty=0.001 -o ppo.missing_eos_penalty=1.0 ...

# Resume an interrupted PPO/GRPO run (restores optimizer + step + KL state)
python scripts/train_ppo.py --resume ...

# Length-normalized DPO (curbs length bias)
python scripts/train_dpo.py -o dpo.length_normalize=true ...

# Experimental: vLLM-backed rollouts (GPU only; auto-falls back to HF if unavailable)
pip install vllm && python scripts/train_ppo.py -o ppo.use_vllm=true ...
```

### Evaluate

```bash
# Reward-model accuracy on held-out preferences
python scripts/evaluate.py rm-accuracy --reward-model checkpoints/reward_model \
    --data Anthropic/hh-rlhf --split test --max-samples 1000

# Did RLHF help? Win-rate of the PPO policy vs the SFT baseline, judged by the reward model
python scripts/evaluate.py score-policy --policy checkpoints/ppo \
    --reward-model checkpoints/reward_model --compare checkpoints/sft --num 200

# Independent LLM-as-judge win-rate (Claude), with position-bias control
pip install anthropic && export ANTHROPIC_API_KEY=...
python scripts/evaluate.py judge --policy checkpoints/ppo --base checkpoints/sft --num 100

# Best-of-N: sample N per prompt, keep the reward model's top pick (inference-time alignment)
python scripts/evaluate.py score-policy --policy checkpoints/ppo \
    --reward-model checkpoints/reward_model --compare checkpoints/ppo --best-of-n 8 --num 200
```

### Chat with your model

Talk to the trained policy — a real chat interface (CLI or browser) over the model this pipeline
produced. Works with any local checkpoint or HF id. **Best-of-N** samples several replies per turn
and returns the one the reward model scores highest — better answers at zero extra training cost.

```bash
# Terminal chat
python scripts/chat.py --model checkpoints/ppo
python scripts/chat.py --model checkpoints/ppo --best-of-n 8 --reward-model checkpoints/reward_model

# Browser UI (Gradio chat window at http://localhost:7860)
pip install gradio
python app.py --model checkpoints/ppo --best-of-n 8 --reward-model checkpoints/reward_model
```

### Train on Kaggle (free GPU)

Open [notebooks/kaggle_rlhf.ipynb](notebooks/kaggle_rlhf.ipynb) on Kaggle (T4×2 / P100), enable
the GPU + internet, and **Save Version ▸ Save & Run All (Commit)**. It clones this repo, runs
SFT → reward model (initialized from the SFT checkpoint) → PPO, and writes `RESULTS.md`
(reward-model accuracy, PPO-vs-SFT win-rate, sample completions) to the output. The default
`Qwen/Qwen2.5-0.5B` `full` run is ~3–5 h; `PRESET='fast'` is ~2 h. Free GPUs can be preempted
(the run restarts), so prefer the short preset or split the stages across separate commits.

**Headless via the Kaggle API (no browser):** with `pip install kaggle` and a token in
`~/.kaggle/kaggle.json`, set your username in [kernel-metadata.json](kernel-metadata.json) and run
[scripts/run_on_kaggle.sh](scripts/run_on_kaggle.sh) — it pushes the notebook, polls until done,
and downloads `RESULTS.md` + checkpoints to `./kaggle_out`.

### License

Licensed under the [PolyForm Noncommercial License 1.0.0](https://polyformproject.org/licenses/noncommercial/1.0.0/) — see [LICENSE](LICENSE). You may use, modify, and share this work for any non-commercial purpose with attribution, but not for commercial purposes (including selling it).
