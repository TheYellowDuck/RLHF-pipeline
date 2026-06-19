"""Minimal YAML config with attribute access and CLI dot-overrides.

    cfg = load_config("configs/ppo.yaml", overrides=["ppo.lr=2e-6", "policy.use_lora=true"])
    print(cfg.ppo.lr, cfg.policy.name_or_path)

Nested dicts become nested `Config` objects; lists/scalars pass through. Unknown
keys raise on attribute access so typos surface early.
"""

from __future__ import annotations

import json
from typing import Any, Iterable

import yaml


class Config:
    """Attribute-accessible view over a (possibly nested) dict."""

    def __init__(self, data: dict | None = None):
        object.__setattr__(self, "_data", {})
        for k, v in (data or {}).items():
            self[k] = v

    # --- mapping-ish access -------------------------------------------------
    def __setitem__(self, key: str, value: Any):
        self._data[key] = Config(value) if isinstance(value, dict) else value

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __setattr__(self, key: str, value: Any):
        self[key] = value

    def __getattr__(self, key: str) -> Any:
        try:
            return self._data[key]
        except KeyError as e:
            raise AttributeError(
                f"Config has no key '{key}'. Available: {list(self._data)}"
            ) from e

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def keys(self):
        return self._data.keys()

    def items(self):
        return self._data.items()

    def to_dict(self) -> dict:
        out = {}
        for k, v in self._data.items():
            out[k] = v.to_dict() if isinstance(v, Config) else v
        return out

    def __repr__(self) -> str:
        return f"Config({json.dumps(self.to_dict(), default=str, indent=2)})"


def _coerce(value: str) -> Any:
    """Parse a CLI string override into a typed python value."""
    low = value.lower()
    if low in ("none", "null"):
        return None
    if low == "true":
        return True
    if low == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def apply_overrides(cfg: Config, overrides: Iterable[str] | None) -> Config:
    """Apply ['a.b.c=value', ...] dotted overrides in place; returns cfg."""
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override '{item}' must be of the form key.sub=value")
        key, raw = item.split("=", 1)
        parts = key.split(".")
        node = cfg
        for p in parts[:-1]:
            if p not in node or not isinstance(node[p], Config):
                node[p] = Config()
            node = node[p]
        node[parts[-1]] = _coerce(raw)
    return cfg


def load_config(path: str, overrides: Iterable[str] | None = None) -> Config:
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    cfg = Config(data)
    return apply_overrides(cfg, overrides)
