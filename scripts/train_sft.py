"""Supervised fine-tune a causal LM on demonstrations (the SFT precursor to PPO).

    python scripts/train_sft.py --config configs/sft.yaml \
        -o model.name_or_path=Qwen/Qwen2.5-0.5B -o data.max_samples=20000
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rlhf.algorithms import SFTTrainer
from rlhf.cli import base_parser, init, make_logger
from rlhf.data import load_sft_dataset
from rlhf.models import apply_lora, load_causal_lm, load_tokenizer
from rlhf.utils import resolve_dtype


def main():
    parser = base_parser("Supervised fine-tuning", "configs/sft.yaml")
    parser.add_argument("--accelerate", action="store_true", help="enable accelerate (multi-GPU/DDP)")
    args = parser.parse_args()
    cfg, device = init(args)
    dtype = resolve_dtype(cfg.model.get("dtype", "auto"), device)
    acc = None
    if args.accelerate:
        from accelerate import Accelerator

        acc = Accelerator()

    tok = load_tokenizer(cfg.model.name_or_path)
    model = load_causal_lm(cfg.model.name_or_path, dtype=dtype)
    if cfg.model.get("use_lora", False):
        model = apply_lora(model, cfg.model.get("lora", {}), task_type="CAUSAL_LM")

    train_ds = load_sft_dataset(cfg.data.name, cfg.data.train_split, cfg.data.get("max_samples"))
    eval_ds = None
    if cfg.data.get("eval_split"):
        eval_ds = load_sft_dataset(cfg.data.name, cfg.data.eval_split, cfg.data.get("max_eval_samples"))

    is_main = acc is None or acc.is_main_process
    logger = make_logger(cfg, args, run_name="sft") if is_main else None
    SFTTrainer(model, tok, cfg, device, metric_logger=logger, accelerator=acc).train(train_ds, eval_ds)
    if logger:
        logger.close()


if __name__ == "__main__":
    main()
