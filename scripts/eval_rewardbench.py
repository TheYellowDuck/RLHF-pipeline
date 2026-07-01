"""Evaluate a reward model on RewardBench (allenai/reward-bench).

A non-saturated, discriminating RM yardstick (unlike HH-RLHF, where every UF-trained RM
sits at chance). Reports the per-category breakdown (Chat / Chat Hard / Safety / Reasoning),
the headline category-mean, and the raw micro-accuracy.

    python scripts/eval_rewardbench.py --reward-model checkpoints/reward_model --device cuda
"""

from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rlhf.eval import rewardbench_report
from rlhf.models import RewardModel, load_tokenizer
from rlhf.utils import get_logger, resolve_device, resolve_dtype
from rlhf.utils.generation import score_texts

log = get_logger("rlhf.rewardbench")


def main():
    p = argparse.ArgumentParser(description="Evaluate a reward model on RewardBench")
    p.add_argument("--reward-model", required=True)
    p.add_argument("--dataset", default="allenai/reward-bench")
    p.add_argument("--split", default="filtered")
    p.add_argument("--max-samples", type=int, default=None, help="cap examples (default: all)")
    p.add_argument("--max-length", type=int, default=1024, help="truncate prompt+response (RM trained at 512)")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--device", default="auto")
    p.add_argument("--dtype", default="auto")
    p.add_argument("--head-weights", default=None, help="multi-head RM: comma-sep combine weights, e.g. 1,1,0.5")
    args = p.parse_args()

    from datasets import load_dataset

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    tok = load_tokenizer(args.reward_model)
    rm = RewardModel.from_pretrained(args.reward_model, dtype=dtype).to(device).eval()
    if args.head_weights:
        rm.set_head_weights([float(x) for x in args.head_weights.split(",")])
        log.info("head weights set to %s", rm.head_weights.tolist())

    ds = load_dataset(args.dataset, split=args.split)
    if args.max_samples:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    prompts, chosen, rejected, subsets = ds["prompt"], ds["chosen"], ds["rejected"], ds["subset"]
    log.info("RewardBench %s[%s]: %d pairs", args.dataset, args.split, len(prompts))

    cs = score_texts(rm, tok, prompts, chosen, device, max_length=args.max_length, batch_size=args.batch_size)
    rs = score_texts(rm, tok, prompts, rejected, device, max_length=args.max_length, batch_size=args.batch_size)
    correct = (cs > rs).tolist()

    rep = rewardbench_report(subsets, correct)
    print(f"\nRewardBench  overall(category-mean)={rep['overall']:.4f}  "
          f"micro-acc={rep['accuracy_micro']:.4f}  n={rep['n']}")
    for c in ["Chat", "Chat Hard", "Safety", "Reasoning"]:
        if c in rep["per_category"]:
            print(f"  {c:10s} {rep['per_category'][c]:.4f}")
    if "safety_balanced" in rep:   # the over-refusal-aware view
        print(f"  Safety split: refuse-harm={rep['safety_refuse']:.4f}  "
              f"respond-benign={rep['safety_respond']:.4f}  "
              f"balanced(harm.mean)={rep['safety_balanced']:.4f}")
    log.info("rewardbench report: %s", rep)


if __name__ == "__main__":
    main()
