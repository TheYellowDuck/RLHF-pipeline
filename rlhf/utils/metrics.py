"""Lightweight metric logging: console + JSONL + optional TensorBoard/W&B."""

from __future__ import annotations

import json
import os
import time
from typing import Any

from .common import get_logger
from .tensor_ops import flatten_dict


class MetricLogger:
    """Logs scalar metrics to stdout, a JSONL file, and optional backends.

    backends: any subset of {"tensorboard", "wandb"} (created lazily; missing
    packages are skipped with a warning rather than crashing training).
    """

    def __init__(
        self,
        output_dir: str,
        backends: tuple[str, ...] = (),
        project: str = "rlhf-pipeline",
        run_name: str | None = None,
        config: dict | None = None,
    ):
        self.log = get_logger("rlhf.metrics")
        os.makedirs(output_dir, exist_ok=True)
        self.jsonl_path = os.path.join(output_dir, "metrics.jsonl")
        self._jsonl = open(self.jsonl_path, "a")
        self._t0 = time.time()
        self._tb = None
        self._wandb = None

        if "tensorboard" in backends:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self._tb = SummaryWriter(os.path.join(output_dir, "tb"))
            except Exception as e:  # noqa: BLE001
                self.log.warning("TensorBoard unavailable (%s); skipping.", e)
        if "wandb" in backends:
            try:
                import wandb

                self._wandb = wandb
                wandb.init(project=project, name=run_name, config=config or {})
            except Exception as e:  # noqa: BLE001
                self.log.warning("wandb unavailable (%s); skipping.", e)

    def log_metrics(self, metrics: dict[str, Any], step: int, prefix: str = "") -> None:
        flat = flatten_dict(metrics)
        if prefix:
            flat = {f"{prefix}/{k}": v for k, v in flat.items()}
        flat = {k: (float(v) if hasattr(v, "__float__") else v) for k, v in flat.items()}

        record = {"step": step, "elapsed_s": round(time.time() - self._t0, 1), **flat}
        self._jsonl.write(json.dumps(record) + "\n")
        self._jsonl.flush()

        if self._tb is not None:
            for k, v in flat.items():
                if isinstance(v, (int, float)):
                    self._tb.add_scalar(k, v, step)
        if self._wandb is not None:
            self._wandb.log({**flat, "step": step})

        pretty = "  ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in flat.items()
        )
        self.log.info("step %-6d | %s", step, pretty)

    def close(self) -> None:
        try:
            self._jsonl.close()
        except Exception:  # noqa: BLE001
            pass
        if self._tb is not None:
            self._tb.close()
        if self._wandb is not None:
            self._wandb.finish()
