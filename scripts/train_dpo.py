"""Direct Preference Optimization (RM-free alternative to PPO).

    python scripts/train_dpo.py --config configs/dpo.yaml -o model.name_or_path=checkpoints/sft
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rlhf.algorithms import DPOTrainer
from rlhf.cli import base_parser, init, make_logger
from rlhf.data import load_preference_dataset
from rlhf.models import apply_lora, load_causal_lm, load_tokenizer
from rlhf.utils import resolve_dtype


def main():
    args = base_parser("Direct Preference Optimization", "configs/dpo.yaml").parse_args()
    cfg, device = init(args)
    dtype = resolve_dtype(cfg.model.get("dtype", "auto"), device)
    use_lora = cfg.model.get("use_lora", False)

    tok = load_tokenizer(cfg.model.name_or_path)
    model = load_causal_lm(cfg.model.name_or_path, dtype=dtype)
    if use_lora:
        model = apply_lora(model, cfg.model.get("lora", {}), task_type="CAUSAL_LM")
    ref = None if use_lora else load_causal_lm(cfg.model.name_or_path, dtype=dtype)

    train_ds = load_preference_dataset(cfg.data.name, cfg.data.train_split, cfg.data.get("max_samples"))
    eval_ds = None
    if cfg.data.get("eval_split"):
        eval_ds = load_preference_dataset(cfg.data.name, cfg.data.eval_split, cfg.data.get("max_eval_samples"))

    logger = make_logger(cfg, args, run_name="dpo")
    DPOTrainer(model, ref, tok, cfg, device, metric_logger=logger).train(train_ds, eval_ds)
    logger.close()


if __name__ == "__main__":
    main()
