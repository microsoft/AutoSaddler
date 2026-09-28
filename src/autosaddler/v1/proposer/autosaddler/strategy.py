"""Canonical sampling-strategy capabilities for AutoSaddler."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class SamplingStrategy(StrEnum):
    """Supported sampling policies."""

    AUTOSADDLER = "autosaddler"
    ACTIVESADDLER = "activesaddler"


SessionNumber = int | float
UNSEEN_SCENARIO_EXPLORATION_SESSION: SessionNumber = 3.5


@dataclass(frozen=True)
class StrategySpec:
    """Capabilities that vary independently across sampling strategies."""

    name: SamplingStrategy
    sampler_family: str
    scoring: str
    arm_creation: str
    enabled_sessions: tuple[SessionNumber, ...]
    extra_skills: tuple[str, ...]
    pattern_cli_capabilities: frozenset[str]

    @property
    def pattern_sampling(self) -> bool:
        return self.sampler_family == "modern_bandit"

    @property
    def agent_scoring(self) -> bool:
        return self.scoring == "agent"

    @property
    def agent_arm_creation(self) -> bool:
        return self.arm_creation == "agent"

    def pattern_cli_capabilities_for_session(
        self,
        session: SessionNumber,
    ) -> frozenset[str]:
        """Return the pattern commands exposed to one agent session."""
        if session == 3 and session in self.enabled_sessions:
            return _PATTERN_EXTRACTION_COMMANDS
        if session == 4 and self.agent_scoring:
            return _QUERY_PATTERN_COMMANDS | {"rate"}
        if session == UNSEEN_SCENARIO_EXPLORATION_SESSION and self.agent_arm_creation:
            return _QUERY_PATTERN_COMMANDS | {"decide"}
        return frozenset()


_QUERY_PATTERN_COMMANDS = frozenset(
    {"list", "show", "history", "score", "scenarios"}
)
_PATTERN_EXTRACTION_COMMANDS = _QUERY_PATTERN_COMMANDS | {"register", "tag"}

_STRATEGY_SPECS = {
    SamplingStrategy.AUTOSADDLER: StrategySpec(
        name=SamplingStrategy.AUTOSADDLER,
        sampler_family="epoch",
        scoring="none",
        arm_creation="epoch",
        enabled_sessions=(0, 1, 2),
        extra_skills=(),
        pattern_cli_capabilities=frozenset(),
    ),
    SamplingStrategy.ACTIVESADDLER: StrategySpec(
        name=SamplingStrategy.ACTIVESADDLER,
        sampler_family="modern_bandit",
        scoring="agent",
        arm_creation="agent",
        enabled_sessions=(0, 1, 2, 3, UNSEEN_SCENARIO_EXPLORATION_SESSION, 4),
        extra_skills=(
            "symptom-extract",
            "symptom-normalize",
            "progress-scoring",
        ),
        pattern_cli_capabilities=_PATTERN_EXTRACTION_COMMANDS | {"rate", "decide"},
    ),
}


def canonicalize_sampling_strategy(value: str | SamplingStrategy) -> SamplingStrategy:
    """Validate a strategy name."""
    if isinstance(value, SamplingStrategy):
        return value
    normalized = str(value).strip().lower()
    try:
        return SamplingStrategy(normalized)
    except ValueError as exc:
        choices = ", ".join(strategy.value for strategy in SamplingStrategy)
        raise ValueError(f"Unknown sampling_strategy={value!r}; expected one of: {choices}") from exc


def resolve_strategy(value: str | SamplingStrategy) -> StrategySpec:
    """Return the immutable capability specification for ``value``."""
    return _STRATEGY_SPECS[canonicalize_sampling_strategy(value)]
