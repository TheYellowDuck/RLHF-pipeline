"""Shared command-line plumbing for the training scripts."""

from __future__ import annotations

import argparse

from .utils import load_config, resolve_device, set_seed
from .utils.metrics import MetricLogger


def base_parser(description: str, default_config: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--config", default=default_config, help="path to a YAML config")
    p.add_argument("-o", "--override", action="append", default=[],
                   help="dotted config override, e.g. -o ppo.lr=2e-6 (repeatable)")
    p.add_argument("--report-to", default="none", choices=["none", "tensorboard", "wandb"])
    return p


def init(args):
    cfg = load_config(args.config, args.override)
    set_seed(int(cfg.get("seed", 0)))
    device = resolve_device(cfg.get("device", "auto"))
    return cfg, device


def make_logger(cfg, args, run_name: str) -> MetricLogger:
    backends = () if args.report_to == "none" else (args.report_to,)
    return MetricLogger(cfg.output_dir, backends=backends, run_name=run_name, config=cfg.to_dict())
