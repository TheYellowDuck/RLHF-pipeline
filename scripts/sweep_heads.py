"""Sweep a multi-head reward model's combine-weights over RewardBench (v1) — SCORE ONCE, weight cheaply.

Each head's per-completion scores are computed one time; then every head-weight vector is just a cheap
re-combination + re-aggregation. This maps the objective frontier: for each weight vector we report
overall + balanced-safety (refuse-harm vs respond-benign). Heads are standardized first so weights reflect
relative importance, not each head's arbitrary scale.

    ./.venv/bin/python scripts/sweep_heads.py --reward-model <multihead ckpt> --device cpu
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from rlhf.eval import rewardbench_report
from rlhf.models import RewardModel, load_tokenizer
from rlhf.utils import get_logger, resolve_device, resolve_dtype

log = get_logger("rlhf.sweep")


def score_heads(rm, tok, prompts, responses, device, max_length, batch_size):
    prev = tok.padding_side
    tok.padding_side = "right"
    out = []
    for i in range(0, len(prompts), batch_size):
        texts = [p + r for p, r in zip(prompts[i:i + batch_size], responses[i:i + batch_size])]
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True,
                  max_length=max_length, add_special_tokens=False)
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            h = rm(enc["input_ids"], enc["attention_mask"], return_heads=True)   # [B,K]
        out.append(h.float().cpu())
    tok.padding_side = prev
    return torch.cat(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reward-model", required=True)
    p.add_argument("--data", default="allenai/reward-bench")
    p.add_argument("--split", default="filtered")
    p.add_argument("--max-samples", type=int, default=None)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", default="float32")
    p.add_argument("--grid", default="0,0.5,1,1.5,2", help="per-head weight values to sweep")
    args = p.parse_args()
    from datasets import load_dataset

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    tok = load_tokenizer(args.reward_model)
    rm = RewardModel.from_pretrained(args.reward_model, dtype=dtype).to(device).eval()
    K = rm.num_heads
    log.info("multi-head RM with %d heads", K)

    ds = load_dataset(args.data, split=args.split)
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    prompts, chosen, rejected, subs = ds["prompt"], ds["chosen"], ds["rejected"], ds["subset"]
    log.info("scoring %d pairs x %d heads (once)...", len(prompts), K)
    ch = score_heads(rm, tok, prompts, chosen, device, args.max_length, args.batch_size)     # [N,K]
    rj = score_heads(rm, tok, prompts, rejected, device, args.max_length, args.batch_size)

    pool = torch.cat([ch, rj], 0)                          # standardize per head (scale-fair weights)
    mu, sd = pool.mean(0), pool.std(0).clamp_min(1e-6)
    chz, rjz = (ch - mu) / sd, (rj - mu) / sd

    grid = [float(x) for x in args.grid.split(",")]
    rows = []
    for w in itertools.product(grid, repeat=K):
        if sum(w) == 0:
            continue
        wt = torch.tensor(w)
        correct = ((chz * wt).sum(-1) > (rjz * wt).sum(-1)).tolist()
        rep = rewardbench_report(subs, correct)
        rows.append((w, rep.get("overall", 0), rep.get("safety_balanced", 0),
                     rep.get("safety_refuse", 0), rep.get("safety_respond", 0)))

    sd_list = sd.tolist()

    def show(title, r):
        w, ov, bal, rf, rs = r
        raw = [w[k] / sd_list[k] for k in range(K)]        # deployable head_weights (undo standardization)
        m = max(raw) or 1.0
        raw = [round(x / m, 2) for x in raw]               # normalized so the largest head = 1.0
        print(f"  {title:12s} w(std)={tuple(round(x,1) for x in w)}  ->  --head-weights {','.join(map(str,raw))}"
              f"   overall={ov:.3f} balanced={bal:.3f} refuse={rf:.3f} respond={rs:.3f}")

    print(f"\n=== head-weight sweep ({len(rows)} combos, {K} heads) ===")
    show("uniform", next(r for r in rows if len(set(r[0])) == 1 and r[0][0] > 0))
    print("  -- top 5 by balanced-safety --")
    for r in sorted(rows, key=lambda r: -r[2])[:5]:
        show("", r)
    print("  -- top 5 by overall --")
    for r in sorted(rows, key=lambda r: -r[1])[:5]:
        show("", r)
    print("  -- best 'all-round' (min of overall & balanced maximized) --")
    show("all-round", max(rows, key=lambda r: min(r[1], r[2])))


if __name__ == "__main__":
    main()
