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
    parser = base_parser("Direct Preference Optimization", "configs/dpo.yaml")
    parser.add_argument("--accelerate", action="store_true", help="enable accelerate (multi-GPU/DDP)")
    args = parser.parse_args()
    cfg, device = init(args)
    dtype = resolve_dtype(cfg.model.get("dtype", "auto"), device)
    use_lora = cfg.model.get("use_lora", False)
    acc = None
    if args.accelerate:
        from accelerate import Accelerator

        acc = Accelerator()

    tok = load_tokenizer(cfg.model.name_or_path)
    model = load_causal_lm(cfg.model.name_or_path, dtype=dtype)
    if use_lora:
        model = apply_lora(model, cfg.model.get("lora", {}), task_type="CAUSAL_LM")
    ref = None if use_lora else load_causal_lm(cfg.model.name_or_path, dtype=dtype)

    train_ds = load_preference_dataset(
        cfg.data.name, cfg.data.train_split, cfg.data.get("max_samples"),
        max_pair_similarity=cfg.data.get("max_pair_similarity", 1.0),
        contrast_metric=cfg.data.get("contrast_metric", "jaccard"),
        embedding_model=cfg.data.get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2"))
    eval_ds = None
    if cfg.data.get("eval_split"):
        eval_ds = load_preference_dataset(cfg.data.name, cfg.data.eval_split, cfg.data.get("max_eval_samples"))

    is_main = acc is None or acc.is_main_process
    logger = make_logger(cfg, args, run_name="dpo") if is_main else None
    DPOTrainer(model, ref, tok, cfg, device, metric_logger=logger, accelerator=acc).train(train_ds, eval_ds)
    if logger:
        logger.close()


if __name__ == "__main__":
    main()
