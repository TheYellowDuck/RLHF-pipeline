# Session handoff — RLHF pipeline (continue here)

**Read this first in the next session.** State, the plan for "all recommended for the cleanest result,"
exact commands, and the Kaggle gotchas so nothing gets re-discovered.

## Where we are (results — all committed to `main`)

| Reward model | Cleaned held-out acc | Notes |
|---|---|---|
| 0.5B, original on buggy H4 data | ~0.63 | the wall — turned out to be **label noise**, not capacity |
| 0.5B-Instruct + **cleaned** UltraFeedback | **0.726** | fixing the data (margin 0.79) |
| **1.5B-Instruct + cleaned, LoRA** | **0.8025** | bigger backbone (margin 1.39) — **current best RM** |

- **Full arc: 0.63 → 0.726 → 0.8025** (~+17 pts), every step a diagnosis.
- **PPO (0.5B)**: 56% win-rate vs the un-tuned Instruct policy, RM-judged (mean reward −0.654 → −0.587).
- **Chat UI/CLI** works: `./chat` (terminal) and `./ui` (zero-dep browser UI), Best-of-N reranking built in.

**ACTIVE RUN (2026-06-28):** kernel `georgezhang06/rlhf-pipeline-run` **v15** = step #2, the fresh full
0.5B pipeline (cleaned-data RM → PPO → eval, forced T4), launched and RUNNING. A ~30-min heartbeat is
polling it; on COMPLETE it pulls RESULTS.md, reports RM acc + win-rate, then runs #1 (judge-validate).
If you're resuming and this run is already COMPLETE, just `kaggle kernels output` it — don't relaunch.
Remote: `TheYellowDuck/RLHF-pipeline`.

## Reality check on time (READ THIS)
A 1.5B RM on a free T4 with the OOM-safe config (batch 4 + gradient checkpointing) takes **~9 h** for
4000 pairs. So a **full 1.5B RM+PPO in one 12 h session is NOT feasible** — it must be staged, or use
the 0.5B for the policy. Plan accordingly below.

## The plan: "all recommended for the cleanest result"

### 1. Judge-validate the win-rate  ← do first (free, fast, makes the 56% credible)
The RM-judged 56% is circular (Goodhart). An independent Claude judge gives the honest number.
Your `ANTHROPIC_API_KEY` is in `.env` (gitignored, auto-loaded by `scripts/evaluate.py`). Needs a
**policy checkpoint local** — easiest is to do step 2 first (it produces one), then:
```
./.venv/bin/python scripts/evaluate.py judge \
  --policy <ppo_checkpoint> --base Qwen/Qwen2.5-0.5B-Instruct --num 100 --device cpu
```

### 2. Clean, complete, DOWNLOADABLE 0.5B pipeline  ← the cleanest *feasible* full result (~5-6 h)
Re-run the full 0.5B pipeline on the validated recipe (Instruct policy + cleaned-data RM + PPO) so we
have a fresh, fully-downloadable set of checkpoints to chat with + judge. Notebook **already exists**:
`notebooks/kaggle_rlhf_full.ipynb` (it uses Qwen2.5-0.5B-Instruct + cleaned data).
```
python3 -c "import json;m=json.load(open('kernel-metadata.json'));m['code_file']='notebooks/kaggle_rlhf_full.ipynb';json.dump(m,open('kernel-metadata.json','w'),indent=2)"
./.venv/bin/kaggle kernels push -p . --accelerator NvidiaTeslaT4
```
Arm a heartbeat (~30-min). On COMPLETE → report RM acc + win-rate; download checkpoints → run #1 (judge).

### 3. STRETCH — full 1.5B policy (staged; the absolute cleanest, but more work)
**Notebook now exists: `notebooks/kaggle_ppo_1.5b.ipynb`** (auto-discovers the RM under `/kaggle/input`
via its `reward_config.json` marker, asserts non-empty weights, runs the PPO recipe below + RM-judged
win-rate → RESULTS.md). The ONLY manual step left is creating the RM Dataset (next bullet); then push
with that dataset attached (`+ Add Input`, or add its slug to `kernel-metadata.json` `dataset_sources`)
on a forced T4.
The 0.8025 RM is done (Kaggle v14 output). To get a **1.5B policy**, run PPO reusing it:
- Get the 0.8025 RM into a Kaggle **Dataset** (the kernel output's `checkpoints/reward_model/` —
  local downloads are partial, so add it via the Kaggle UI "New Dataset → from kernel output", or
  re-download with retries until `model.safetensors` is non-zero, then `kaggle datasets create`).
- New notebook: attach that Dataset, `cp` it to `checkpoints/reward_model`, run **PPO only** with
  `policy.name_or_path=Qwen/Qwen2.5-1.5B-Instruct policy.use_lora=true`, `ppo.total_episodes=1024
  rollout_batch_size=8 mini_batch_size=1 generation.max_new_tokens=40` (~5 h), then eval + judge.
- PPO at 1.5B holds policy(LoRA)+RM on the T4 — should fit with LoRA + mini-batch 1; if OOM drop rollout to 4.

### 4. GRM auxiliary-LM regularization  ✅ BUILT + TESTED (2026-06-28) — ready to launch, no run yet
Implemented as a strictly **opt-in** lever (default off → existing behavior byte-identical; the 0.5B run
above is unaffected). `L = L_BT + α·L_LM`, α from `train.aux_lm_coef` (SFT-reg on the chosen response,
no ref model). When on, the backbone loads as `AutoModelForCausalLM` (LM head kept); the reward is still
the value head on `hidden_states[-1]`, which is **numerically identical** to the old trunk path
(smoke test: max Δ=0.00). Touched: `reward_model.py` (aux_lm flag in from_backbone/forward),
`reward_trainer.py` (`aux_lm_loss` + wiring + per-step bt/aux logging), `preference.py` (opt-in
`emit_loss_mask` = response-only mask), `train_reward_model.py` (auto-enable LM head when coef>0),
`configs/reward_model.yaml` (`model.aux_lm`, `train.aux_lm_coef`). Tests: 2 new unit tests (21 pass)
+ `aux_lm_check` in smoke_test (reward-equivalence, learns to 1.00, reloads as trunk-only RM). Expected
+3–8 OOD (GRM, arXiv:2406.10216). **To run** (one knob turns it all on):
```
!python scripts/train_reward_model.py -o model.name_or_path=Qwen/Qwen2.5-0.5B-Instruct \
  -o data.name=argilla/ultrafeedback-binarized-preferences-cleaned \
  -o data.train_split='train[2000:]' -o data.eval_split='train[:2000]' \
  -o data.max_samples=10000 -o train.epochs=1 -o train.lr=1.0e-5 -o train.bf16=true \
  -o train.aux_lm_coef=0.05 -o train.batch_size=2 -o train.grad_accum=8   # batch 2: logits are [B,T,vocab]
```
Then eval on the CLEANED test (and an OOD set, e.g. Skywork, to see the GRM lift) vs the 0.726 baseline.
**Memory caveat:** the LM head makes `[B,T,vocab]` logits — at 0.5B on a T4 use batch 2 / grad_accum 8.

## Kaggle gotchas (CRITICAL — caused ~10 failed runs)
- **Always force T4:** `kaggle kernels push -p . --accelerator NvidiaTeslaT4`. Default is **P100**,
  whose sm_60 Kaggle's base torch dropped → every CUDA op dies. (`run_on_kaggle.sh` already defaults to T4.)
- **Pin `transformers<5`** — 5.x breaks the Qwen2.5 tokenizer on Kaggle (notebooks already do).
- **1.5B OOMs** without gradient checkpointing (LoRA freezes params but activations are full-size):
  `train.batch_size=4 train.grad_accum=4 train.gradient_checkpointing=true`.
- **Output downloads are often partial** (0-byte big files) — but eval runs *on Kaggle* and writes
  RESULTS.md, so trust RESULTS.md. If empty, the heartbeat re-evals the (LoRA-merged) checkpoint locally.
- ~12 h commit limit; timed-out runs are `CANCEL_ACKNOWLEDGED` but still persist output.
- Username `georgezhang06`; kernel id `georgezhang06/rlhf-pipeline-run`. Auth `~/.kaggle/access_token`
  (kaggle CLI 2.x in `.venv`). **Never paste the API token in chat.**

## Validated recipe
RM: init from an **Instruct** model + **cleaned** data `argilla/ultrafeedback-binarized-preferences-cleaned`,
`train_split='train[2000:]' eval_split='train[:2000]'`, **1 epoch**, `label_smoothing=0`, no contrast filter,
lr 1e-5 (full FT) / 1e-4 (LoRA), last-token pooling. **Eval on the CLEANED test**; report old H4 only for contrast.
LoRA-merge: trainers `save(merge=True)` on final save → full model next stage loads (`merge_if_peft`, `rlhf/models/loading.py`).

## Key files
- Notebooks: `kaggle_rlhf_full.ipynb` (0.5B full — use for #2), `kaggle_rm_1.5b.ipynb` (1.5B RM → 0.8025),
  `kaggle_rm_experiment.ipynb` (0.5B RM).
- Chat: `scripts/chat.py`, `app.py`, `rlhf/inference.py`; launchers `./chat`, `./ui`.
- Eval: `scripts/evaluate.py` (rm-accuracy / score-policy `--best-of-n` / judge). Data-mixing: `data.name="a,b"`.
- Memory: `~/.claude/.../memory/` — `reward-model-0.63-was-label-noise.md`, `kaggle-p100-torch-incompat.md`,
  `autonomous-heartbeat-working-style.md`.

## To resume next session
1. `cd /Users/georgezhang/RLHF-pipeline && git pull`. 2. Read this + the memory notes.
3. Do #2 (fresh 0.5B full pipeline on forced T4 + heartbeat) → #1 (judge it) for a clean, validated,
   complete result. 4. Then #3 (1.5B policy, staged) and/or #4 (GRM) as quota/time allow.
