"""Resolve shared and strategy-specific AutoSaddler settings."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any

from autosaddler.v1.proposer.autosaddler.strategy import StrategySpec, resolve_strategy


@dataclass(frozen=True)
class ResolvedStrategySettings:
    """Effective settings consumed by the proposer and prompt bundle."""

    strategy: StrategySpec
    eta: float
    softmax_temperature: float
    min_prob: float
    pattern_extraction_timeout: float
    arm_scoring_timeout: float

    def __post_init__(self) -> None:
        if not 0.0 < self.eta <= 1.0:
            raise ValueError("eta must be in (0, 1]")
        if self.softmax_temperature <= 0.0:
            raise ValueError("softmax_temperature must be positive")
        if not 0.0 <= self.min_prob < 1.0:
            raise ValueError("min_prob must be in [0, 1)")
        if self.pattern_extraction_timeout <= 0.0:
            raise ValueError("pattern_extraction_timeout must be positive")
        if self.arm_scoring_timeout <= 0.0:
            raise ValueError("arm_scoring_timeout must be positive")

    def to_fingerprint_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["strategy"] = self.strategy.name.value
        return payload


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def _lookup(mapping: dict[str, Any], path: tuple[str, ...]) -> tuple[bool, Any]:
    current: Any = mapping
    for key in path:
        if not isinstance(current, dict) or key not in current:
            return False, None
        current = current[key]
    return True, current


def _resolve_value(
    config: dict[str, Any],
    effective_nested: dict[str, Any],
    *,
    flat_key: str,
    nested_path: tuple[str, ...],
    default: Any,
) -> Any:
    nested_present, nested_value = _lookup(effective_nested, nested_path)
    flat_present = flat_key in config
    flat_value = config.get(flat_key)
    if nested_present and flat_present and nested_value != flat_value:
        dotted = ".".join(nested_path)
        raise ValueError(
            f"Conflicting strategy setting: autosaddler.{flat_key}={flat_value!r} "
            f"but autosaddler.{dotted}={nested_value!r}"
        )
    if nested_present:
        return nested_value
    if flat_present:
        return flat_value
    return default


def resolve_strategy_settings(config: dict[str, Any] | None) -> ResolvedStrategySettings:
    """Resolve shared nested settings, optional overrides, and legacy flat keys."""
    config = config or {}
    strategy = resolve_strategy(config.get("sampling_strategy", "autosaddler"))

    common_nested = {key: deepcopy(config[key]) for key in ("pattern", "bandit") if isinstance(config.get(key), dict)}
    overrides = config.get("strategy_overrides") or {}
    if not isinstance(overrides, dict):
        raise TypeError("autosaddler.strategy_overrides must be a mapping")
    selected_override = overrides.get(strategy.name.value) or {}
    if not isinstance(selected_override, dict):
        raise TypeError(f"autosaddler.strategy_overrides.{strategy.name.value} must be a mapping")
    effective_nested = _deep_merge(common_nested, selected_override)
    legacy_aggregation_present, _ = _lookup(
        effective_nested,
        ("bandit", "scoring", "agent_aggregation"),
    )
    if "agent_score_aggregation" in config or legacy_aggregation_present:
        raise ValueError(
            "agent_score_aggregation has been removed; Agent scoring always "
            "uses (severity + fixability + breadth + (1 - side_effect)) / 4"
        )

    resolved = ResolvedStrategySettings(
        strategy=strategy,
        eta=float(
            _resolve_value(
                config,
                effective_nested,
                flat_key="eta",
                nested_path=("bandit", "scoring", "ema_eta"),
                default=0.3,
            )
        ),
        softmax_temperature=float(
            _resolve_value(
                config,
                effective_nested,
                flat_key="softmax_temperature",
                nested_path=("bandit", "selection", "softmax_temperature"),
                default=0.15,
            )
        ),
        min_prob=float(
            _resolve_value(
                config,
                effective_nested,
                flat_key="min_prob",
                nested_path=("bandit", "selection", "min_prob"),
                default=0.02,
            )
        ),
        pattern_extraction_timeout=float(
            _resolve_value(
                config,
                effective_nested,
                flat_key="pattern_extraction_timeout",
                nested_path=("pattern", "extraction_timeout"),
                default=3600.0,
            )
        ),
        arm_scoring_timeout=float(
            _resolve_value(
                config,
                effective_nested,
                flat_key="arm_scoring_timeout",
                nested_path=("bandit", "scoring", "agent_timeout"),
                default=3600.0,
            )
        ),
    )
    return resolved
