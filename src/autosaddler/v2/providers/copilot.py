from __future__ import annotations

import json
import logging
import os
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from uuid import NAMESPACE_URL, uuid5

from autosaddler.v2.core.domain import JsonValue, to_json_value
from autosaddler.v2.prompting.models import ToolCall, Usage
from autosaddler.v2.providers.base import AgentTransport, BaseAgentProvider, TransportOutcome, observe_usage
from autosaddler.v2.providers.workspace_renderer import RenderedSession, copilot_renderer

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class CopilotCustomProviderConfig:
    type: str
    base_url: str
    wire_api: str
    model_id: str
    wire_model: str

    def __post_init__(self) -> None:
        for name in ("type", "base_url", "wire_api", "model_id", "wire_model"):
            if not getattr(self, name):
                raise ValueError(f"Copilot custom provider {name} cannot be empty")

    def as_mapping(self) -> dict[str, str]:
        return {
            "type": self.type,
            "base_url": self.base_url,
            "wire_api": self.wire_api,
            "model_id": self.model_id,
            "wire_model": self.wire_model,
        }


@dataclass(frozen=True, slots=True)
class CopilotProviderConfig:
    model: str = "gpt-5.4"
    reasoning_effort: str | None = "high"
    provider: CopilotCustomProviderConfig | None = None


class CopilotAgentProvider(BaseAgentProvider):
    def __init__(
        self,
        config: CopilotProviderConfig | None = None,
        *,
        transport: AgentTransport | None = None,
    ) -> None:
        resolved_config = config or CopilotProviderConfig()
        super().__init__(copilot_renderer(), transport or CopilotSdkTransport(resolved_config))


class CopilotSdkTransport:
    def __init__(self, config: CopilotProviderConfig) -> None:
        self.config = config

    async def run(self, session: RenderedSession, timeout_seconds: float) -> TransportOutcome:
        from copilot import CopilotClient, PermissionHandler
        from copilot.generated.session_events import SessionEventType

        tool_calls: list[ToolCall] = []
        usage: list[Usage] = []
        aggregate_metrics: dict[str, JsonValue] | None = None
        aggregate_metrics_error: dict[str, JsonValue] | None = None
        model_call_failure_type = getattr(SessionEventType, "MODEL_CALL_FAILURE", None)

        def handle_event(event: Any) -> None:
            if event.type == SessionEventType.TOOL_EXECUTION_START:
                arguments = _tool_arguments(getattr(event.data, "arguments", {}))
                tool_calls.append(
                    ToolCall(tool=getattr(event.data, "tool_name", "unknown"), arguments=arguments)
                )
            elif event.type == SessionEventType.TOOL_EXECUTION_COMPLETE and tool_calls:
                result = getattr(event.data, "result", None)
                preview = getattr(result, "content", result)
                previous = tool_calls[-1]
                tool_calls[-1] = ToolCall(
                    tool=previous.tool,
                    arguments=previous.arguments,
                    result_preview=str(preview)[:500],
                )
            elif event.type == SessionEventType.ASSISTANT_USAGE:
                item = _copilot_usage(
                    event.data,
                    self.config,
                    agent_id=getattr(event, "agent_id", None),
                    interaction_type=(
                        getattr(event, "interaction_type", None)
                        or getattr(event.data, "interaction_type", None)
                    ),
                )
                usage.append(item)
                observe_usage(item)
            elif model_call_failure_type is not None and event.type == model_call_failure_type:
                status_code = getattr(event.data, "status_code", None)
                agent_id = _text(getattr(event, "agent_id", None))
                interaction_type = _text(
                    getattr(event, "interaction_type", None)
                    or getattr(event.data, "interaction_type", None)
                )
                item = Usage(
                    role="optimizer",
                    model=getattr(event.data, "model", self.config.model),
                    duration_seconds=_seconds(getattr(event.data, "duration", None)),
                    status="failed",
                    error_type=f"http_{status_code}" if status_code is not None else type(event.data).__name__,
                    usage_incomplete=True,
                    provider_correlation_id=_correlation_id(event.data),
                    agent_id=agent_id,
                    agent_scope=_copilot_agent_scope(
                        agent_id=agent_id,
                        initiator=getattr(event.data, "initiator", None),
                        interaction_type=interaction_type,
                    ),
                    configured_settings=_without_none(
                        {
                            "model": self.config.model,
                            "reasoning_effort": self.config.reasoning_effort,
                        }
                    ),
                    provider_metadata={
                        **_attributes(event.data, ("initiator", "source", "status_code")),
                        **({"interaction_type": interaction_type} if interaction_type is not None else {}),
                    },
                )
                usage.append(item)
                observe_usage(item)

        client = CopilotClient(working_directory=str(session.workspace))
        sdk_session_id: str | None = None
        try:
            await client.start()
            session_config: dict[str, Any] = {
                "model": self.config.model,
                "on_permission_request": PermissionHandler.approve_all,
                "working_directory": str(session.workspace),
                "available_tools": list(session.allowed_tools),
                "system_message": {"mode": "append", "content": session.system_context},
                "enable_skills": True,
                "skill_directories": [str(session.skill_directory)],
            }
            if self.config.reasoning_effort is not None:
                session_config["reasoning_effort"] = self.config.reasoning_effort
            if self.config.provider is not None:
                session_config["provider"] = _copilot_provider_mapping(self.config.provider)
            requested_session_id = (
                str(uuid5(NAMESPACE_URL, f"autosaddler:{session.session_id}"))
                if session.session_id is not None
                else None
            )
            sdk_session = await client.create_session(
                **session_config,
                session_id=requested_session_id,
            )
            sdk_session_id = str(sdk_session.session_id)
            sdk_session.on(handle_event)
            response = await sdk_session.send_and_wait(session.task_prompt, timeout=timeout_seconds)
            try:
                aggregate_metrics = _copilot_session_metrics(await sdk_session.rpc.usage.get_metrics())
            except Exception as error:
                aggregate_metrics_error = _error_summary(error)
                logger.warning("Failed to reconcile Copilot session usage", exc_info=True)
            raw_response = response.data.content if response and hasattr(response.data, "content") else ""
            return TransportOutcome(
                raw_response=raw_response,
                tool_calls=tuple(tool_calls),
                usage=tuple(usage),
                usage_streamed=True,
            )
        finally:
            try:
                await client.stop()
            except Exception:
                logger.warning("Failed to stop Copilot client", exc_info=True)
            finally:
                if sdk_session_id is not None and session.trace_dir is not None:
                    try:
                        _export_copilot_session_state(
                            sdk_session_id=sdk_session_id,
                            destination_root=session.trace_dir / "copilot-session-state",
                            aggregate_metrics=aggregate_metrics,
                            aggregate_metrics_error=aggregate_metrics_error,
                        )
                    except Exception:
                        logger.warning("Failed to write Copilot trace export manifest", exc_info=True)


def _copilot_provider_mapping(config: CopilotCustomProviderConfig) -> dict[str, str]:
    mapping = config.as_mapping()
    if config.type != "openai" or urlparse(config.base_url).hostname != "api.openai.com":
        return mapping
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY must be set for the official OpenAI provider")
    return {**mapping, "api_key": api_key}


def _tool_arguments(value: Any) -> Mapping[str, JsonValue]:
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            value = {"raw": value}
    if not isinstance(value, Mapping):
        value = {"raw": str(value)}
    converted = to_json_value(value)
    assert isinstance(converted, dict)
    return converted


def _copilot_usage(
    value: Any,
    config: CopilotProviderConfig | str,
    *,
    agent_id: str | None = None,
    interaction_type: str | None = None,
) -> Usage:
    resolved_config = config if isinstance(config, CopilotProviderConfig) else CopilotProviderConfig(model=config)
    agent_id = _text(agent_id)
    interaction_type = _text(interaction_type)
    input_tokens = int(getattr(value, "input_tokens", 0) or 0)
    output_tokens = int(getattr(value, "output_tokens", 0) or 0)
    provider_cost = _float(getattr(value, "cost", None))
    copilot_usage = getattr(value, "copilot_usage", None)
    nano_aiu_value = getattr(copilot_usage, "total_nano_aiu", None)
    if nano_aiu_value is None:
        nano_aiu_value = getattr(copilot_usage, "totalNanoAiu", None)
    nano_aiu = _float(nano_aiu_value)
    cached_input_tokens = int(getattr(value, "cache_read_tokens", 0) or 0)
    cache_creation_input_tokens = int(getattr(value, "cache_write_tokens", 0) or 0)
    initiator = _text(getattr(value, "initiator", None))
    total_tokens = getattr(value, "total_tokens", None)
    total_tokens_is_inferred = total_tokens is None
    return Usage(
        role="optimizer",
        model=getattr(value, "model", resolved_config.model),
        input_tokens=input_tokens,
        cached_input_tokens=cached_input_tokens,
        uncached_input_tokens=input_tokens - cached_input_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=int(getattr(value, "reasoning_tokens", 0) or 0),
        total_tokens=input_tokens + output_tokens,
        provider_cost=provider_cost,
        provider_cost_unit="premium_request_multiplier" if provider_cost is not None else None,
        provider_nano_aiu=nano_aiu,
        duration_seconds=_seconds(getattr(value, "duration", None)),
        provider_correlation_id=_correlation_id(value),
        agent_id=agent_id,
        agent_scope=_copilot_agent_scope(
            agent_id=agent_id,
            initiator=initiator,
            interaction_type=interaction_type,
        ),
        input_token_semantics="includes_cached_tokens",
        provider_reported_input_tokens=input_tokens,
        provider_reported_total_tokens=int(total_tokens) if total_tokens is not None else None,
        total_tokens_is_inferred=total_tokens_is_inferred,
        configured_settings=_without_none(
            {
                "model": resolved_config.model,
                "reasoning_effort": resolved_config.reasoning_effort,
            }
        ),
        reported_settings=_attributes(
            value,
            (
                "reasoning_effort",
                "temperature",
                "top_p",
                "max_tokens",
                "max_output_tokens",
                "seed",
                "frequency_penalty",
                "presence_penalty",
            ),
        ),
        provider_metadata={
            **_attributes(
                value,
                (
                    "api_endpoint",
                    "content_filter_triggered",
                    "finish_reason",
                    "initiator",
                    "inter_token_latency",
                    "parent_tool_call_id",
                    "time_to_first_token",
                ),
            ),
            **({"interaction_type": interaction_type} if interaction_type is not None else {}),
            **(
                {"cache_creation_input_tokens": cache_creation_input_tokens}
                if cache_creation_input_tokens
                else {}
            ),
        },
    )


def _copilot_agent_scope(
    *,
    agent_id: str | None,
    initiator: str | None,
    interaction_type: str | None,
) -> str:
    is_subagent = bool(agent_id) or initiator == "sub-agent" or bool(
        interaction_type and "subagent" in interaction_type
    )
    return "subagent" if is_subagent else "main"


def _attributes(value: Any, names: tuple[str, ...]) -> dict[str, JsonValue]:
    observed: dict[str, JsonValue] = {}
    for name in names:
        item = getattr(value, name, None)
        if item is None:
            continue
        if isinstance(item, timedelta):
            item = item.total_seconds()
        elif isinstance(item, Enum):
            item = item.value
        try:
            converted = to_json_value(item)
        except (TypeError, ValueError):
            logger.warning("Ignoring unsupported Copilot usage attribute %s", name, exc_info=True)
            continue
        observed[name] = converted
    return observed


def _text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, Enum):
        value = value.value
    return str(value)


def _without_none(values: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    return {key: value for key, value in values.items() if value is not None}


def _error_summary(error: Exception) -> dict[str, JsonValue]:
    return {"error_type": type(error).__name__, "error": str(error)[:2_000]}


def _export_copilot_session_state(
    *,
    sdk_session_id: str,
    destination_root: Path,
    source_root: Path | None = None,
    aggregate_metrics: Mapping[str, JsonValue] | None = None,
    aggregate_metrics_error: Mapping[str, JsonValue] | None = None,
) -> None:
    source = (source_root or Path.home() / ".copilot" / "session-state") / sdk_session_id
    destination_root.mkdir(parents=True, exist_ok=True)
    destination = destination_root / sdk_session_id
    manifest_path = destination_root / "export-manifest.json"
    manifest: dict[str, JsonValue] = {
        "schema_version": "autosaddler-copilot-trace-export/v1",
        "copilot_session_id": sdk_session_id,
        "source": str(source),
        "destination": destination.name,
        "sensitive": True,
    }
    if aggregate_metrics is not None:
        manifest["session_usage"] = dict(aggregate_metrics)
    if aggregate_metrics_error is not None:
        manifest["session_usage_error"] = dict(aggregate_metrics_error)
    try:
        if source.is_symlink():
            raise ValueError("Copilot session-state root cannot be a symlink")
        if not source.is_dir():
            raise FileNotFoundError(f"Copilot session-state directory does not exist: {source}")
        symlinks = [path for path in source.rglob("*") if path.is_symlink()]
        if symlinks:
            raise ValueError(f"Copilot session-state contains symlinks: {symlinks[0]}")
        if destination.exists():
            shutil.rmtree(destination)
        shutil.copytree(
            source,
            destination,
            symlinks=False,
            ignore=shutil.ignore_patterns("inuse.*.lock"),
        )
        files = sorted(path.relative_to(destination).as_posix() for path in destination.rglob("*") if path.is_file())
        manifest.update(
            {
                "status": "exported",
                "files": files,
                "summary": _copilot_trace_summary(destination / "events.jsonl"),
            }
        )
    except Exception as error:
        manifest.update({"status": "failed", **_error_summary(error)})
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _copilot_session_metrics(value: Any) -> dict[str, JsonValue]:
    total_nano_aiu = _float(getattr(value, "total_nano_aiu", None))
    models: dict[str, JsonValue] = {}
    for model, metric in getattr(value, "model_metrics", {}).items():
        requests = metric.requests
        native_usage = metric.usage
        input_tokens = int(native_usage.input_tokens)
        cached_input_tokens = int(native_usage.cache_read_tokens)
        cache_creation_input_tokens = int(native_usage.cache_write_tokens)
        model_nano_aiu = _float(metric.total_nano_aiu)
        model_value: dict[str, JsonValue] = {
            "request_count": int(requests.count),
            "premium_request_cost": float(requests.cost),
            "input_tokens": input_tokens,
            "cached_input_tokens": cached_input_tokens,
            "uncached_input_tokens": input_tokens - cached_input_tokens,
            "output_tokens": int(native_usage.output_tokens),
            "reasoning_tokens": int(native_usage.reasoning_tokens or 0),
            "total_tokens": input_tokens + int(native_usage.output_tokens),
        }
        if model_nano_aiu is not None:
            model_value["total_nano_aiu"] = model_nano_aiu
            model_value["ai_credits"] = model_nano_aiu / 1_000_000_000
        if cache_creation_input_tokens:
            model_value["provider_metadata"] = {
                "cache_creation_input_tokens": cache_creation_input_tokens,
            }
        models[str(model)] = model_value
    result: dict[str, JsonValue] = {
        "scope": "session_inclusive",
        "includes_subagents": True,
        "total_premium_request_cost": float(value.total_premium_request_cost),
        "total_user_requests": int(value.total_user_requests),
        "total_api_duration_seconds": float(value.total_api_duration_ms) / 1_000,
        "last_call_input_tokens": int(value.last_call_input_tokens),
        "last_call_output_tokens": int(value.last_call_output_tokens),
        "models": models,
    }
    current_model = getattr(value, "current_model", None)
    if current_model is not None:
        result["current_model"] = str(current_model)
    if total_nano_aiu is not None:
        result["total_nano_aiu"] = total_nano_aiu
        result["total_ai_credits"] = total_nano_aiu / 1_000_000_000
    return result


def _copilot_trace_summary(events_path: Path) -> dict[str, JsonValue]:
    if not events_path.is_file():
        return {"event_count": 0, "tool_call_count": 0}
    event_count = 0
    tool_call_count = 0
    timestamps: list[datetime] = []
    session_settings: dict[str, JsonValue] = {}
    for line in events_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        if not isinstance(event, Mapping):
            raise TypeError("Copilot trace events must be JSON objects")
        event_count += 1
        if event.get("type") == "tool.execution_start":
            tool_call_count += 1
        if event.get("type") == "session.start" and isinstance(event.get("data"), Mapping):
            data = event["data"]
            assert isinstance(data, Mapping)
            for key in (
                "selectedModel",
                "reasoningEffort",
                "contextTier",
                "temperature",
                "topP",
                "maxTokens",
                "maxOutputTokens",
                "seed",
            ):
                value = data.get(key)
                if isinstance(value, (str, int, float, bool)):
                    session_settings[key] = value
        timestamp = event.get("timestamp")
        if isinstance(timestamp, str):
            timestamps.append(datetime.fromisoformat(timestamp.replace("Z", "+00:00")))
    summary: dict[str, JsonValue] = {
        "event_count": event_count,
        "tool_call_count": tool_call_count,
    }
    if timestamps:
        summary.update(
            {
                "started_at": min(timestamps).isoformat(),
                "ended_at": max(timestamps).isoformat(),
                "duration_seconds": (max(timestamps) - min(timestamps)).total_seconds(),
            }
        )
    if session_settings:
        summary["session_settings"] = session_settings
    return summary


def _correlation_id(value: Any) -> str | None:
    for field in ("api_call_id", "provider_call_id", "service_request_id"):
        candidate = getattr(value, field, None)
        if candidate is not None:
            return str(candidate)
    return None


def _seconds(value: Any) -> float | None:
    if isinstance(value, timedelta):
        return value.total_seconds()
    return _float(value)


def _float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None