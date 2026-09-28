"""Shared YAML configuration loading with environment expansion and overlays."""

from __future__ import annotations

import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

_ENV_PATTERN = re.compile(r"\$\{([^}:]+)(:-([^}]*))?\}")


def _expand_environment(raw: str) -> str:
    def replace(match: re.Match[str]) -> str:
        name = match.group(1)
        default = match.group(3)
        value = os.environ.get(name, "")
        if value:
            return value
        return default if default is not None else ""

    return os.path.expandvars(_ENV_PATTERN.sub(replace, raw))


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _load_yaml_config(path: Path, stack: tuple[Path, ...]) -> dict[str, Any]:
    path = path.resolve()
    if path in stack:
        chain = " -> ".join(str(item) for item in (*stack, path))
        raise ValueError(f"Cyclic config extends chain: {chain}")
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    loaded = yaml.safe_load(_expand_environment(path.read_text(encoding="utf-8"))) or {}
    if not isinstance(loaded, dict):
        raise TypeError(f"Config root must be a mapping: {path}")

    extends = loaded.pop("extends", [])
    if isinstance(extends, str):
        extends = [extends]
    if not isinstance(extends, list) or not all(isinstance(item, str) for item in extends):
        raise ValueError(f"Config extends must be a path or list of paths: {path}")

    merged: dict[str, Any] = {}
    next_stack = (*stack, path)
    for parent in extends:
        parent_path = Path(parent)
        if not parent_path.is_absolute():
            parent_path = path.parent / parent_path
        merged = _deep_merge(merged, _load_yaml_config(parent_path, next_stack))
    return _deep_merge(merged, loaded)


def load_yaml_config(config_path: str | Path) -> dict[str, Any]:
    """Load YAML, expanding environment variables and recursive ``extends``."""
    return _load_yaml_config(Path(config_path), ())
