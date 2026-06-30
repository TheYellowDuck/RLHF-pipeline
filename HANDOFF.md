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
- **PPO (0.5B), fresh v15 run**: RM-judged **63.0%** win-rate (reward **−0.835 → −0.623**) BUT the
  independent **Claude judge says 49%** (39 win / 43 lose / 18 tie, n=100, Opus 4.8, position-swapped) —
  a statistical tie, slightly favoring the *un-tuned* base. **The 63% was Goodhart**: PPO inflated the
  RM score without real quality gain (the 14-pt RM-vs-judge gap = the reward-hacking tax). This is the
  headline lesson of step #1 — never trust the RM-judged win-rate alone. (An earlier run scored 56% RM-judged.)
- **PPO (1.5B, hack-resistant) — THE WIN (#3, judge-validated):** RM-judged 69% AND independent **Claude
  judge 59.25%** (52 win / 33 lose / 15 tie, n=100, Opus 4.8, swapped). Unlike v15, the directions
  **agree** and the judge clears 50% by ~9 pts — a **genuine** quality gain, not Goodhart. Recipe: 1.5B
  LoRA policy + the **0.8025** RM + tight KL (target 3) + length/EOS penalties + score_clip, lr 1e-6,
  1024 episodes. Note: KL/seq stayed tiny (~0.03 ≪ 3) and training-time RM score was flat, yet the move
  was small-but-real — *low KL was healthy (a targeted non-hacking nudge), not under-training* (I
  predicted ~50% and was wrong; the judge proved a real win). Checkpoint: `kaggle_ppo_ckpt/` (3.09 GB, intact).
- **PPO (1.5B) v2 — the RM-ceiling result:** pushed harder (lr 5e-6, 2048 ep) — this time KL/seq DID rise
  into the budget (0.03 → ~2.0, max 3.2) and the RM score climbed (−1.89 → −1.43), but the **judge = 57.25%**
  (49 W / 33 L / 18 T) — statistically identical to v1's 59.25% (same 33 base-wins). So **harder PPO bought
  no extra real win**; the RM-vs-judge gap stayed ~10 pts (Goodhart). Lesson: **~59% is THIS RM's quality
  ceiling — the reward model, not the PPO, is now the bottleneck.** To go higher, improve the RM (→ GRM).
  Checkpoint: `kaggle_ppo_v2/` (3.09 GB, intact).
- **GRM aux-LM A/B (#4) — honest NEGATIVE:** matched 0.5B RMs, only diff = `aux_lm_coef` (0 vs 0.05),
  same cleaned UF. in-dist: base **0.7255** vs GRM **0.7175**; OOD (HH-RLHF): base **0.483** vs GRM **0.488**.
  The aux loss gave **+0.5 pt OOD / −0.8 pt in-dist — both within noise** (n=2000, ±~1.1 pt SE), nowhere
  near the claimed +3–8. Caveat: both RMs are **~chance (0.48) on HH-RLHF** (a UF-trained RM doesn't
  transfer to HH dialogue) — the OOD set is too far-shifted to discriminate (floor effect). **Verdict: no
  evidence GRM helps here; NOT worth a 1.5B GRM re-run on this evidence.** To retest GRM fairly, use a
  closer OOD set where the base RM is above chance (e.g. RewardBench / another instruction-following pref set).
- **Chat UI/CLI** works: `./chat` (terminal) and `./ui` (zero-dep browser UI), Best-of-N reranking built in.

**COMPLETED RUN (2026-06-28):** kernel `georgezhang06/rlhf-pipeline-run` **v15** = step #2, the fresh full
0.5B pipeline (cleaned-data RM → PPO → eval, forced T4) — **DONE + judge-validated (#1)**. RESULTS.md:
RM cleaned **0.726** (margin 0.79), RM old-H4 0.591, PPO RM-judged win-rate **63.0%** — but the
**independent Claude judge gives 49%** (Goodhart; see the PPO bullet). Output downloaded to
`kaggle_output/`; BOTH checkpoints intact + fully downloadable (`checkpoints/ppo/` 988 MB +
`checkpoints/reward_model/` 988 MB). Steps **#1 + #2 COMPLETE**.

**COMPLETED RUN (2026-06-28):** kernel **`georgezhang06/rlhf-ppo-1p5b` v1** = step #3, the hack-resistant
1.5B PPO — **DONE + judge-validated**. Ran ~1.5 h (faster than the ~5 h estimate), rollout 8 fit the T4
(no OOM). Result: RM-judged 69%, **Claude judge 59.25%** — the project's first genuine PPO win (see the
PPO-1.5B bullet). Checkpoint pulled via a *targeted* `--file-pattern 'checkpoints/ppo/' --page-size 200 -o`
download (the broad pull left the 3 GB safetensors 0-byte — the partial-download bug). **Steps #1–#4
status: #1 ✅, #2 ✅, #3 ✅ (real win), #4 GRM built+tested but not yet run.**
Remote: `TheYellowDuck/RLHF-pipeline` — **origin is in sync (all pushed).**

- **Better-DATA A/B (#5) — broader but no OOD lift:** matched 0.5B RMs, fixed 6000-pair budget. The
  in-notebook uf-only cell crashed (eval_split defaulted to `test`, which cleaned-UF lacks — now fixed),
  so the **uf-only control = the GRM-base RM** (identical 6000-UF recipe): UF **0.7255**, HH-OOD **0.483**.
  The **mix** (3000 UF + 3000 Skywork): UF **0.7205**, Skywork-held **0.995**, HH-OOD **0.4855**. So mixing
  Skywork makes a **broadly-competent** RM (UF held *and* Skywork nailed at 0.995) with no UF loss — but
  **no HH-OOD lift** (0.4855 ≈ 0.483 ≈ chance; same floor effect as GRM). **Meta-lesson: HH-RLHF accuracy
  is a poor yardstick for "better RM"** (everything UF-trained is at the floor); to really test "better RM
  → better policy" you need the 1.5B-RM → PPO → judge chain or a non-saturated bench (RewardBench).

**EARLIER ARC COMPLETE (2026-06-29) — #1–#4 done.** (A) PPO v2 `rlhf-ppo-1p5b` v2 — judge
**57.25%** (= v1's 59.25% = the RM ceiling). (B) GRM A/B `rlhf-rm-grm` — **negative** (GRM ≈ base, no OOD
lift; see the GRM bullet). No Kaggle runs active; no heartbeats. Local checkpoints: `kaggle_ppo_ckpt/`
(v1 1.5B PPO), `kaggle_ppo_v2/` (v2), `kaggle_output/` (v15 0.5B RM+PPO). origin in sync.

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
via its `reward_config.json` marker, skips bundled smoke RMs, asserts the 1.5B weights). It runs a
**hack-resistant PPO recipe** — KL target 6→3 + firmer init, `length_penalty 0.01`, `missing_eos_penalty
1.0`, `score_clip 8.0` — added *because v15's 0.5B PPO reward-hacked (RM 63% vs judge 49%)*. On-Kaggle
win-rate is RM-judged (circular); **judge-validate locally after download** (the notebook prints the
exact command). Est. **~5–6 h** on a T4 (PPO is ~90%; ≤12 h even if OOM forces rollout 4). RM dataset is
ready, so launch = push → set kernel id `rlhf-ppo-1p5b` + `dataset_sources` → T4 push.

**Getting the 0.8025 RM into a Dataset — ✅ DONE (2026-06-28).** The dataset
**`georgezhang06/rlhf-rm-1p5b-08025`** is created + verified (RESULTS.md shows 0.8025; `model.safetensors`
= 3.09 GB, complete). It bundles the whole `/kaggle/working` tree (incl. tiny smoke RMs), but the PPO
notebook now skips `/smoke/` and picks the largest-weights checkpoint, so it lands on the real 1.5B RM.
**So #3 is ready to launch** (push → set kernel id `rlhf-ppo-1p5b` + `dataset_sources` → T4 push). How it
was made, for reference: there is **no API/CLI way to pull a specific *old* kernel version's output**
(`kaggle kernels output` only serves the LATEST — issue #442); v14 was UI-only, so it was created via the
Kaggle UI "New Dataset → from kernel output" on version 14:
1. Kaggle UI → the `rlhf-pipeline-run` kernel → pick **version 14** → *Output* → **Download** the
   `checkpoints/reward_model/` folder (browser download is reliable; the CLI partial-download bug doesn't apply).
2. `scripts/make_rm_dataset.sh <that-folder>` — drills to the `reward_config.json` dir, sanity-checks the
   weights are non-empty, writes `dataset-metadata.json`, and runs `kaggle datasets create` →
   `georgezhang06/rlhf-rm-1p5b-08025`. The script then prints the exact PPO-launch commands.
3. The PPO notebook needs no change (it globs `/kaggle/input/**/reward_config.json`, so a `dataset_sources`
   OR a `kernel_sources` mount both work). Launch on a **new kernel id** (`rlhf-ppo-1p5b`) so it never
   buries another output again — the root cause of this whole detour was reusing one kernel id.

**Partial-download workaround** (for whenever you DO pull a *latest* output and big files come back 0-byte):
`kaggle kernels output <kernel> -p out/ --page-size 200 --file-pattern 'reward_model' -o` and retry; the
`-o` re-fetches, the pattern skips unrelated files, and page-size 200 avoids pagination drops.

The 0.8025 RM is done (Kaggle v14 output). To get a **1.5B policy**, run PPO reusing it:
- Get the 0.8025 RM into a Kaggle **Dataset** (the kernel output's `checkpoints/reward_model/` —
  local downloads are partial, so add it via the Kaggle UI "New Dataset → from kernel output", or
  re-download with retries until `model.safetensors` is non-zero, then `kaggle datasets create`).
- New notebook: attach that Dataset, `cp` it to `checkpoints/reward_model`, run **PPO only** with
  `policy.name_or_path=Qwen/Qwen2.5-1.5B-Instruct policy.use_lora=true`, `ppo.total_episodes=1024
  rollout_batch_size=8 mini_batch_size=1 generation.max_new_tokens=40` (~5 h), then eval + judge.
- PPO at 1.5B holds policy(LoRA)+RM on the T4 — should fit with LoRA + mini-batch 1; if OOM drop rollout to 4.

### 4. GRM auxiliary-LM regularization  ✅ BUILT + RUN (2026-06-29) — negative result (see the GRM bullet up top)
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
