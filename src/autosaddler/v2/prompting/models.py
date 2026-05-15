from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Callable, Literal, Mapping, TypeAlias

from jsonschema import ValidationError
from jsonschema.validators import validator_for

from autosaddler.v2.core.domain import Cost, JsonValue, freeze_json_mapping, to_json_value

SessionKind: TypeAlias = str
Capability: TypeAlias = Literal["read_workspace", "edit_workspace", "run_commands", "load_skills", "network"]
SessionStatus: TypeAlias = Literal["completed", "timeout", "failed", "interrupted"]
UsageStatus: TypeAlias = Literal["success", "failed", "cancelled", "timeout", "interrupted"]
InputTokenSemantics: TypeAlias = Literal["includes_cached_tokens", "excludes_cached_tokens", "unknown"]
AgentScope: TypeAlias = Literal["main", "subagent", "session_inclusive", "unknown"]


@dataclass(frozen=True, slots=True)
class SessionSpec:
    kind: SessionKind
    system_context: str
    task_prompt: str
    skills: Mapping[str, str]
    output_schema: Mapping[str, JsonValue]
    workspace_files: Mapping[str, str]
    capabilities: frozenset[Capability]
    mutation_label: str | None = None

    def __post_init__(self) -> None:
        if not self.kind or not self.task_prompt:
            raise ValueError("SessionSpec kind and task_prompt must be non-empty")
        if self.mutation_label is not None and not self.mutation_label:
            raise ValueError("SessionSpec.mutation_label must be non-empty when set")
        for relative_path in self.workspace_files:
            path = PurePosixPath(relative_path)
            if path.is_absolute() or ".." in path.parts:
                raise ValueError(f"Unsafe workspace file path: {relative_path}")
        schema = to_json_value(self.output_schema)
        assert isinstance(schema, dict)
        validator_for(schema).check_schema(schema)
        object.__setattr__(self, "skills", MappingProxyType(dict(self.skills)))
        object.__setattr__(self, "workspace_files", MappingProxyType(dict(self.workspace_files)))
        object.__setattr__(self, "output_schema", freeze_json_mapping(self.output_schema))


@dataclass(frozen=True, slots=True)
class SessionRequest:
    session_id: str
    operation_id: str
    spec: SessionSpec
    workspace: Path
    timeout_seconds: float
    usage_observer: Callable[["Usage"], None] | None = None
    trace_dir: Path | None = None

    def __post_init__(self) -> None:
        if not self.session_id or not self.operation_id:
            raise ValueError("SessionRequest IDs must be non-empty")
        if self.timeout_seconds <= 0:
            raise ValueError("SessionRequest.timeout_seconds must be positive")


@dataclass(frozen=True, slots=True)
class ToolCall:
    tool: str
    arguments: Mapping[str, JsonValue]
    result_preview: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "arguments", freeze_json_mapping(self.arguments))


@dataclass(frozen=True, slots=True)
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    model: str | None = None
    role: str = "optimizer"
    cached_input_tokens: int = 0
    uncached_input_tokens: int | None = None
    reasoning_tokens: int = 0
    total_tokens: int | None = None
    provider_cost: float | None = None
    provider_cost_unit: str | None = None
    provider_nano_aiu: float | None = None
    provider_ai_credits: float | None = None
    duration_seconds: float | None = None
    status: UsageStatus = "success"
    error_type: str | None = None
    usage_incomplete: bool = False
    provider_correlation_id: str | None = None
    agent_id: str | None = None
    agent_scope: AgentScope = "unknown"
    input_token_semantics: InputTokenSemantics = "unknown"
    provider_reported_input_tokens: int | None = None
    provider_reported_total_tokens: int | None = None
    total_tokens_is_inferred: bool = False
    configured_settings: Mapping[str, JsonValue] = MappingProxyType({})
    reported_settings: Mapping[str, JsonValue] = MappingProxyType({})
    provider_metadata: Mapping[str, JsonValue] = MappingProxyType({})

    def __post_init__(self) -> None:
        token_values = (
            self.input_tokens,
            self.cached_input_tokens,
            self.uncached_input_tokens,
            self.output_tokens,
            self.reasoning_tokens,
            self.provider_reported_input_tokens,
            self.provider_reported_total_tokens,
        )
        if any(value is not None and value < 0 for value in token_values):
            raise ValueError("Token counts cannot be negative")
        total_tokens = self.input_tokens + self.output_tokens if self.total_tokens is None else self.total_tokens
        if total_tokens < 0:
            raise ValueError("Usage.total_tokens cannot be negative")
        object.__setattr__(self, "total_tokens", total_tokens)
        uncached_input_tokens = (
            self.input_tokens - self.cached_input_tokens
            if self.uncached_input_tokens is None
            else self.uncached_input_tokens
        )
        if uncached_input_tokens < 0:
            raise ValueError("Usage.uncached_input_tokens cannot be negative")
        object.__setattr__(self, "uncached_input_tokens", uncached_input_tokens)
        if self.provider_cost is not None and self.provider_cost < 0:
            raise ValueError("Usage.provider_cost cannot be negative")
        if self.provider_nano_aiu is not None and self.provider_nano_aiu < 0:
            raise ValueError("Usage.provider_nano_aiu cannot be negative")
        if self.provider_ai_credits is not None and self.provider_ai_credits < 0:
            raise ValueError("Usage.provider_ai_credits cannot be negative")
        if self.provider_nano_aiu is not None:
            ai_credits = self.provider_nano_aiu / 1_000_000_000
            if self.provider_ai_credits is not None and self.provider_ai_credits != ai_credits:
                raise ValueError("Usage.provider_ai_credits must equal provider_nano_aiu / 1e9")
            object.__setattr__(self, "provider_ai_credits", ai_credits)
        if self.duration_seconds is not None and self.duration_seconds < 0:
            raise ValueError("Usage.duration_seconds cannot be negative")
        if (self.provider_cost is None) != (self.provider_cost_unit is None):
            raise ValueError("Provider cost requires both a value and a unit")
        if not self.role:
            raise ValueError("Usage.role must be non-empty")
        if self.cached_input_tokens + uncached_input_tokens != self.input_tokens:
            raise ValueError("Usage input_tokens must equal cached_input_tokens + uncached_input_tokens")
        if self.reasoning_tokens > self.output_tokens:
            raise ValueError("Usage.reasoning_tokens must be a subset of output_tokens")
        if total_tokens != self.input_tokens + self.output_tokens:
            raise ValueError("Usage.total_tokens must equal input_tokens + output_tokens")
        object.__setattr__(self, "configured_settings", freeze_json_mapping(self.configured_settings))
        object.__setattr__(self, "reported_settings", freeze_json_mapping(self.reported_settings))
        object.__setattr__(self, "provider_metadata", freeze_json_mapping(self.provider_metadata))

    @property
    def normalized_input_tokens(self) -> int:
        assert self.total_tokens is not None
        return self.total_tokens - self.output_tokens

    @property
    def normalized_nonreasoning_output_tokens(self) -> int:
        return self.output_tokens - self.reasoning_tokens


@dataclass(frozen=True, slots=True)
class SessionResult:
    status: SessionStatus
    structured_output: Mapping[str, JsonValue] | None
    raw_response: str
    tool_calls: tuple[ToolCall, ...]
    usage: tuple[Usage, ...]
    cost: Cost
    error: str | None = None

    def __post_init__(self) -> None:
        if self.status == "completed" and self.error is not None:
            raise ValueError("A completed session cannot carry an error")
        if self.status == "failed" and not self.error:
            raise ValueError("A failed session requires an error")
        if self.structured_output is not None:
            object.__setattr__(self, "structured_output", freeze_json_mapping(self.structured_output))


def session_output_validation_error(
    schema: Mapping[str, JsonValue],
    output: Mapping[str, JsonValue],
) -> str | None:
    schema_value = to_json_value(schema)
    output_value = to_json_value(output)
    assert isinstance(schema_value, dict) and isinstance(output_value, dict)
    validator = validator_for(schema_value)(schema_value)
    try:
        validator.validate(output_value)
    except ValidationError as error:
        location = "$" + "".join(f"[{part!r}]" for part in error.absolute_path)
        return f"Structured output violates the session schema at {location}: {error.message}"
    return None