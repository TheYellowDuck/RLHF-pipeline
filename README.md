# RLHF Pipeline (from scratch)

A complete, readable implementation of the post-training recipe used to align
frontier language models:

```
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

The reinforcement-learning and reward-modelling logic is **written by hand**
(GAE, the clipped PPO surrogate, the per-token KL-to-reference shaping, the
Bradley-Terry / DPO / GRPO objectives). HuggingFace `transformers` supplies only
the pretrained backbones and tokenizers; `peft` supplies optional LoRA.

> Built and validated against **transformers 5.x** and **torch 2.12**. Designed
> to smoke-test on a laptop CPU and train for real on a free Kaggle GPU.

---

## What's included

| Stage | File | Method |
|-------|------|--------|
| SFT | [rlhf/algorithms/sft_trainer.py](rlhf/algorithms/sft_trainer.py) | teacher forcing, prompt-masked labels |
| Reward model | [rlhf/algorithms/reward_trainer.py](rlhf/algorithms/reward_trainer.py) | Bradley-Terry pairwise loss |
| **PPO** | [rlhf/algorithms/ppo_trainer.py](rlhf/algorithms/ppo_trainer.py) | rollouts → RM score → KL shaping → GAE → clipped surrogate + clipped value + entropy → adaptive KL |
| DPO | [rlhf/algorithms/dpo_trainer.py](rlhf/algorithms/dpo_trainer.py) | implicit-reward preference optimization (sigmoid / IPO / hinge) |
| GRPO | [rlhf/algorithms/grpo_trainer.py](rlhf/algorithms/grpo_trainer.py) | group-relative advantages, critic-free, k3 KL penalty |

Models: scalar-head [reward model](rlhf/models/reward_model.py) and an
[actor-critic policy](rlhf/models/policy.py) (shared trunk + value head). LoRA is
supported everywhere; with LoRA the frozen reference policy is recovered by
disabling the adapter (no second copy of weights in memory).

---

## Install

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

(Apple-Silicon / CPU works for smoke tests; CUDA recommended for real runs.)

## 30-second sanity check

Runs the whole pipeline — RM → SFT → PPO → DPO → GRPO — on a tiny random GPT-2
with synthetic data, on CPU, in a few seconds. Verifies tensor alignment,
masking, checkpoint save/load and that all losses are finite:

```bash
python scripts/smoke_test.py
python -m pytest tests/ -q          # 11 fast unit tests for the RL/RM math
```

---

## Real training (full recipe)

Defaults use [`Anthropic/hh-rlhf`](https://huggingface.co/datasets/Anthropic/hh-rlhf)
preferences. Pick a small base model that fits your GPU (e.g. `gpt2`,
`EleutherAI/pythia-410m`, `Qwen/Qwen2.5-0.5B`).

```bash
# 1. Supervised fine-tuning (the policy init + the PPO reference)
python scripts/train_sft.py -o model.name_or_path=Qwen/Qwen2.5-0.5B \
    -o data.max_samples=20000 -o output_dir=checkpoints/sft

# 2. Reward model
python scripts/train_reward_model.py -o model.name_or_path=Qwen/Qwen2.5-0.5B \
    -o data.max_samples=40000 -o output_dir=checkpoints/reward_model

# 3a. PPO against the reward model
python scripts/train_ppo.py \
    -o policy.name_or_path=checkpoints/sft \
    -o reward_model.name_or_path=checkpoints/reward_model

# 3b. …or GRPO (critic-free) against the same reward model
python scripts/train_grpo.py \
    -o policy.name_or_path=checkpoints/sft \
    -o reward_model.name_or_path=checkpoints/reward_model

# DPO alternative — no reward model, no RL loop
python scripts/train_dpo.py -o model.name_or_path=checkpoints/sft
```

Every script takes `--config <yaml>`, repeatable `-o key.sub=value` overrides, and
`--report-to {none,tensorboard,wandb}`. Metrics also stream to
`<output_dir>/metrics.jsonl`. Configs live in [configs/](configs/).

### Use LoRA (fits bigger models on small GPUs)

```bash
python scripts/train_ppo.py -o policy.use_lora=true \
    -o policy.name_or_path=checkpoints/sft \
    -o reward_model.name_or_path=checkpoints/reward_model
```

---

## Evaluation

```bash
# Reward-model accuracy on held-out preferences
python scripts/evaluate.py rm-accuracy --reward-model checkpoints/reward_model \
    --data Anthropic/hh-rlhf --split test --max-samples 1000

# Did RLHF help? Win-rate of the PPO policy vs the SFT baseline, judged by the RM
python scripts/evaluate.py score-policy --policy checkpoints/ppo \
    --reward-model checkpoints/reward_model --compare checkpoints/sft --num 200
```

---

## Train on Kaggle (free GPU)

Open [notebooks/kaggle_rlhf.ipynb](notebooks/kaggle_rlhf.ipynb) on Kaggle
(T4×2 / P100), enable the GPU + internet, and run top to bottom. It clones/copies
this repo, runs SFT → RM → PPO, and scores the result.

---

## Method notes

- **Reward model.** A scalar head reads the trunk's last non-pad hidden state.
  Trained with `−log σ(r_chosen − r_rejected)`, the MLE of the Bradley-Terry model.
- **PPO reward shaping.** Per response token the reward is a KL-to-reference
  penalty `−β·(log π − log π_ref)`; the scalar RM score is added at the final
  token. β is adapted toward a target KL. This keeps the policy from drifting
  off-distribution while chasing reward (reward hacking).
- **Advantages.** GAE(γ, λ) over the value head's per-token estimates, then
  whitened. Token/value alignment: the log-prob and value of response token at
  position `j` come from the model output at `j−1`.
- **DPO.** Optimizes the same Bradley-Terry preference likelihood but with the
  *implicit* reward `β·log(π/π_ref)`, removing the separate RM and RL loop.
- **GRPO.** Replaces the value critic with a group baseline: sample G responses
  per prompt, advantage = (reward − group mean) / group std. Cheaper and stable;
  the DeepSeek-R1 recipe.

## Repo layout

```
rlhf/
  data/         preference / prompt / SFT datasets + collators (correct padding sides)
  models/       reward model, actor-critic policy, value head, loaders (LoRA, v5-safe)
  algorithms/   reward / sft / ppo / dpo / grpo trainers
  utils/        config, metrics, generation, and the RL/tensor math (GAE, logprobs…)
  cli.py        shared argparse + logger plumbing
scripts/        train_*.py, evaluate.py, smoke_test.py
configs/        one YAML per stage
tests/          fast unit tests for the math
notebooks/      Kaggle GPU runner
```

## Limitations / honest scope

- Reference is a single-machine, single-GPU implementation (plus gradient
  accumulation + LoRA). No multi-GPU sharding, vLLM rollout, or PagedAttention.
- Reward models on small backbones overfit fast — use eval accuracy as the guide.
- This is an educational, faithful reproduction of the algorithms, not a
  throughput-optimized training stack.
