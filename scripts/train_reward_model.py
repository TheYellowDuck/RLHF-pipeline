"""Train a reward model on a preference dataset.

    python scripts/train_reward_model.py --config configs/reward_model.yaml \
        -o model.name_or_path=EleutherAI/pythia-410m -o data.max_samples=20000
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rlhf.algorithms import RewardTrainer
from rlhf.cli import base_parser, init, make_logger
from rlhf.data import load_preference_dataset
from rlhf.models import RewardModel, load_tokenizer
from rlhf.utils import resolve_dtype


def main():
    args = base_parser("Train a reward model", "configs/reward_model.yaml").parse_args()
    cfg, device = init(args)
    dtype = resolve_dtype(cfg.model.get("dtype", "auto"), device)

    tok = load_tokenizer(cfg.model.name_or_path)
    rm = RewardModel.from_backbone(
        cfg.model.name_or_path, dtype=dtype,
        use_lora=cfg.model.get("use_lora", False), lora_cfg=cfg.model.get("lora", {}),
    )

    train_ds = load_preference_dataset(cfg.data.name, cfg.data.train_split, cfg.data.get("max_samples"))
    eval_ds = None
    if cfg.data.get("eval_split"):
        eval_ds = load_preference_dataset(cfg.data.name, cfg.data.eval_split, cfg.data.get("max_eval_samples"))

    logger = make_logger(cfg, args, run_name="reward_model")
    RewardTrainer(rm, tok, cfg, device, metric_logger=logger).train(train_ds, eval_ds)
    logger.close()


if __name__ == "__main__":
    main()
