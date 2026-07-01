"""Evaluate a reward model on RewardBench 2 (allenai/reward-bench-2) — best-of-4 format.

RB2 adds a **Factuality** subset (detect hallucinations / confident errors) that RewardBench v1
lacks — the axis where a sycophantic RM ranks confident-wrong above honest answers. Each prompt has
1 chosen + 3 rejected completions; the RM is correct iff it scores the chosen above ALL rejected.

    python scripts/eval_rewardbench2.py --reward-model <ckpt> --subset Factuality --device cpu
"""

from __future__ import annotations

import argparse
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from rlhf.models import RewardModel, load_tokenizer
from rlhf.utils import get_logger, resolve_device, resolve_dtype
from rlhf.utils.generation import score_texts

log = get_logger("rlhf.rewardbench2")


def main():
    p = argparse.ArgumentParser(description="Evaluate a reward model on RewardBench 2")
    p.add_argument("--reward-model", required=True)
    p.add_argument("--subset", default=None, help="e.g. Factuality; default = all subsets")
    p.add_argument("--max-prompts", type=int, default=None)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="auto")
    p.add_argument("--head-weights", default=None, help="multi-head RM: comma-sep combine weights")
    args = p.parse_args()

    from datasets import load_dataset

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    tok = load_tokenizer(args.reward_model)
    rm = RewardModel.from_pretrained(args.reward_model, dtype=dtype).to(device).eval()
    if args.head_weights:
        rm.set_head_weights([float(x) for x in args.head_weights.split(",")])

    ds = load_dataset("allenai/reward-bench-2", split="test")
    if args.subset:
        ds = ds.filter(lambda e: e["subset"] == args.subset)
    if args.max_prompts:
        ds = ds.select(range(min(args.max_prompts, len(ds))))

    # flatten to (prompt, response) so we score every completion in batched passes
    flat_p, flat_r, spans, subs = [], [], [], []
    for ex in ds:
        comps = list(ex["chosen"]) + list(ex["rejected"])   # index 0..len(chosen)-1 are correct
        start = len(flat_p)
        flat_p += [ex["prompt"]] * len(comps)
        flat_r += comps
        spans.append((start, len(ex["chosen"]), len(comps)))
        subs.append(ex["subset"])
    log.info("RewardBench 2%s: %d prompts, %d completions",
             f"[{args.subset}]" if args.subset else "", len(spans), len(flat_p))

    sc = score_texts(rm, tok, flat_p, flat_r, device, max_length=args.max_length, batch_size=args.batch_size)

    by_sub = defaultdict(lambda: [0, 0])
    for (start, n_chosen, n_tot), sub in zip(spans, subs):
        s = sc[start:start + n_tot]
        best_chosen = s[:n_chosen].max().item()
        best_rejected = s[n_chosen:].max().item()
        ok = best_chosen > best_rejected
        by_sub[sub][0] += int(ok)
        by_sub[sub][1] += 1

    print(f"\nRewardBench 2 {'['+args.subset+']' if args.subset else '(all subsets)'}")
    tot_c = tot_n = 0
    for sub in sorted(by_sub):
        c, n = by_sub[sub]
        tot_c += c; tot_n += n
        print(f"  {sub:22s} {c/n:.4f}  (n={n})")
    print(f"  {'OVERALL':22s} {tot_c/max(1,tot_n):.4f}  (n={tot_n})")
    log.info("rb2 result: %s", {s: by_sub[s][0] / by_sub[s][1] for s in by_sub})


if __name__ == "__main__":
    main()
