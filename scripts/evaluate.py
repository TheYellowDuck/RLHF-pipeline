"""Evaluate RLHF artifacts.

Reward-model accuracy on held-out preferences:
    python scripts/evaluate.py rm-accuracy --reward-model checkpoints/reward_model \
        --data Anthropic/hh-rlhf --split test --max-samples 1000

Score a policy's generations + (optionally) win-rate vs a baseline policy:
    python scripts/evaluate.py score-policy --policy checkpoints/ppo \
        --reward-model checkpoints/reward_model --data Anthropic/hh-rlhf \
        --num 200 --compare checkpoints/sft
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rlhf.algorithms import RewardTrainer
from rlhf.data import load_preference_dataset, load_prompt_dataset
from rlhf.models import ActorCriticPolicy, RewardModel, load_tokenizer
from rlhf.utils import Config, get_logger, resolve_device, resolve_dtype
from rlhf.utils.generation import generate_responses, score_texts

log = get_logger("rlhf.eval")


def rm_accuracy(args, device, dtype):
    tok = load_tokenizer(args.reward_model)
    rm = RewardModel.from_pretrained(args.reward_model, dtype=dtype)
    ds = load_preference_dataset(args.data, args.split, args.max_samples)
    cfg = Config(dict(output_dir="/tmp/eval", data=dict(max_length=args.max_length),
                      train=dict(batch_size=args.batch_size, bf16=False)))
    metrics = RewardTrainer(rm, tok, cfg, device).evaluate(ds)
    log.info("reward-model eval: %s", metrics)
    print(metrics)


def score_policy(args, device, dtype):
    tok = load_tokenizer(args.reward_model)
    rm = RewardModel.from_pretrained(args.reward_model, dtype=dtype).to(device).eval()
    prompts = load_prompt_dataset(args.data, args.split, args.num)["prompt"]

    def run(path):
        pol = ActorCriticPolicy.from_pretrained(path, dtype=dtype).to(device).eval()
        resp = generate_responses(pol, tok, prompts, device, max_new_tokens=args.max_new_tokens,
                                  do_sample=not args.greedy, temperature=args.temperature,
                                  max_prompt_length=args.max_length // 2, batch_size=args.batch_size)
        sc = score_texts(rm, tok, prompts, resp, device, max_length=args.max_length, batch_size=args.batch_size)
        return resp, sc

    resp, sc = run(args.policy)
    log.info("policy %s: mean reward %.4f +/- %.4f over %d prompts",
             args.policy, sc.mean().item(), sc.std().item(), len(sc))
    for p, r, s in list(zip(prompts, resp, sc))[:3]:
        print(f"\n--- reward={s.item():.3f}\nPROMPT:{p[-160:]}\nRESPONSE:{r[:200]}")

    if args.compare:
        _, base_sc = run(args.compare)
        win = (sc > base_sc).float().mean().item()
        log.info("WIN-RATE of %s vs %s under RM: %.1f%% (Δreward %.4f)",
                 args.policy, args.compare, 100 * win, (sc.mean() - base_sc.mean()).item())
        print(f"win_rate={100*win:.1f}%  base_mean={base_sc.mean().item():.4f}  policy_mean={sc.mean().item():.4f}")


def main():
    p = argparse.ArgumentParser(description="Evaluate RLHF artifacts")
    sub = p.add_subparsers(dest="mode", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--reward-model", required=True)
    common.add_argument("--data", default="Anthropic/hh-rlhf")
    common.add_argument("--split", default="test")
    common.add_argument("--max-length", type=int, default=512)
    common.add_argument("--batch-size", type=int, default=8)
    common.add_argument("--device", default="auto")
    common.add_argument("--dtype", default="auto")

    a = sub.add_parser("rm-accuracy", parents=[common])
    a.add_argument("--max-samples", type=int, default=1000)

    b = sub.add_parser("score-policy", parents=[common])
    b.add_argument("--policy", required=True)
    b.add_argument("--compare", default=None, help="baseline policy for win-rate")
    b.add_argument("--num", type=int, default=200)
    b.add_argument("--max-new-tokens", type=int, default=64)
    b.add_argument("--temperature", type=float, default=1.0)
    b.add_argument("--greedy", action="store_true")

    args = p.parse_args()
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    if args.mode == "rm-accuracy":
        rm_accuracy(args, device, dtype)
    else:
        score_policy(args, device, dtype)


if __name__ == "__main__":
    main()
