"""GRPO RL fine-tuning (critic-free, group-relative advantages).

    python scripts/train_grpo.py --config configs/grpo.yaml \
        -o policy.name_or_path=checkpoints/sft \
        -o reward_model.name_or_path=checkpoints/reward_model
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rlhf.algorithms import GRPOTrainer
from rlhf.cli import base_parser, init, make_logger
from rlhf.data import load_prompt_dataset
from rlhf.models import ActorCriticPolicy, RewardModel, load_causal_lm, load_tokenizer
from rlhf.utils import resolve_dtype


def main():
    args = base_parser("GRPO RL fine-tuning", "configs/grpo.yaml").parse_args()
    cfg, device = init(args)
    dtype = resolve_dtype(cfg.policy.get("dtype", "auto"), device)
    use_lora = cfg.policy.get("use_lora", False)

    tok = load_tokenizer(cfg.policy.name_or_path)
    policy = ActorCriticPolicy.from_pretrained_lm(
        cfg.policy.name_or_path, dtype=dtype, use_lora=use_lora, lora_cfg=cfg.policy.get("lora", {}))
    reward_model = RewardModel.from_pretrained(
        cfg.reward_model.name_or_path,
        dtype=resolve_dtype(cfg.reward_model.get("dtype", "auto"), device))
    ref = None if use_lora else load_causal_lm(cfg.policy.name_or_path, dtype=dtype)

    prompt_ds = load_prompt_dataset(cfg.data.name, cfg.data.train_split, cfg.data.get("max_samples"))

    logger = make_logger(cfg, args, run_name="grpo")
    GRPOTrainer(policy, reward_model, tok, cfg, device, ref_model=ref, metric_logger=logger).train(prompt_ds)
    logger.close()


if __name__ == "__main__":
    main()
