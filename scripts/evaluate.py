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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

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

    def run(path, best_of_n=1):
        pol = ActorCriticPolicy.from_pretrained(path, dtype=dtype).to(device).eval()

        def gen(sample):
            return generate_responses(pol, tok, prompts, device, max_new_tokens=args.max_new_tokens,
                                      do_sample=sample, temperature=args.temperature,
                                      max_prompt_length=args.max_length // 2, batch_size=args.batch_size)

        def sc_of(resp):
            return score_texts(rm, tok, prompts, resp, device, max_length=args.max_length, batch_size=args.batch_size)

        if best_of_n <= 1:
            resp = gen(not args.greedy)
            return resp, sc_of(resp)
        # Best-of-N: sample N times, keep the reward model's top pick per prompt.
        best_resp, best_sc = list(prompts), torch.full((len(prompts),), float("-inf"))
        for _ in range(best_of_n):
            r = gen(True)
            s = sc_of(r)
            for i in range(len(prompts)):
                if s[i] > best_sc[i]:
                    best_sc[i], best_resp[i] = s[i], r[i]
        return best_resp, best_sc

    resp, sc = run(args.policy, args.best_of_n)
    if args.best_of_n > 1:
        log.info("policy used Best-of-%d reranking by the reward model", args.best_of_n)
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


def judge_policy(args, device, dtype):
    """Independent LLM-as-judge win-rate of a policy vs a baseline (RM-free)."""
    from rlhf.eval import ClaudeJudge, judge_win_rate

    tok = load_tokenizer(args.policy)
    prompts = load_prompt_dataset(args.data, args.split, args.num)["prompt"]

    def gen(path):
        pol = ActorCriticPolicy.from_pretrained(path, dtype=dtype).to(device).eval()
        return generate_responses(pol, tok, prompts, device, max_new_tokens=args.max_new_tokens,
                                  do_sample=not args.greedy, temperature=args.temperature,
                                  max_prompt_length=args.max_length // 2, batch_size=args.batch_size)

    policy_resps, base_resps = gen(args.policy), gen(args.base)
    judge = ClaudeJudge(model=args.judge_model, thinking=args.thinking)
    result = judge_win_rate(judge, prompts, policy_resps, base_resps, swap=not args.no_swap)
    log.info("LLM-judge result: %s", result)
    print(result)


def main():
    p = argparse.ArgumentParser(description="Evaluate RLHF artifacts")
    sub = p.add_subparsers(dest="mode", required=True)

    shared = argparse.ArgumentParser(add_help=False)
    shared.add_argument("--data", default="Anthropic/hh-rlhf")
    shared.add_argument("--split", default="test")
    shared.add_argument("--max-length", type=int, default=512)
    shared.add_argument("--batch-size", type=int, default=8)
    shared.add_argument("--device", default="auto")
    shared.add_argument("--dtype", default="auto")

    common = argparse.ArgumentParser(add_help=False, parents=[shared])
    common.add_argument("--reward-model", required=True)

    a = sub.add_parser("rm-accuracy", parents=[common])
    a.add_argument("--max-samples", type=int, default=1000)

    b = sub.add_parser("score-policy", parents=[common])
    b.add_argument("--policy", required=True)
    b.add_argument("--compare", default=None, help="baseline policy for win-rate")
    b.add_argument("--num", type=int, default=200)
    b.add_argument("--max-new-tokens", type=int, default=64)
    b.add_argument("--temperature", type=float, default=1.0)
    b.add_argument("--greedy", action="store_true")
    b.add_argument("--best-of-n", type=int, default=1,
                   help="sample N per prompt, keep the RM's best (inference-time alignment)")

    j = sub.add_parser("judge", parents=[shared], help="independent Claude-as-judge win-rate")
    j.add_argument("--policy", required=True)
    j.add_argument("--base", required=True, help="baseline policy to compare against")
    j.add_argument("--num", type=int, default=100)
    j.add_argument("--max-new-tokens", type=int, default=64)
    j.add_argument("--temperature", type=float, default=1.0)
    j.add_argument("--greedy", action="store_true")
    j.add_argument("--judge-model", default="claude-opus-4-8")
    j.add_argument("--thinking", action="store_true", help="adaptive thinking (slower, stronger)")
    j.add_argument("--no-swap", action="store_true", help="disable position-bias swap")

    args = p.parse_args()
    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype, device)
    if args.mode == "rm-accuracy":
        rm_accuracy(args, device, dtype)
    elif args.mode == "score-policy":
        score_policy(args, device, dtype)
    else:
        judge_policy(args, device, dtype)


if __name__ == "__main__":
    main()
