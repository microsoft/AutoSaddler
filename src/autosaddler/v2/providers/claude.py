from __future__ import annotations

import json
import logging
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from autosaddler.v2.core.domain import JsonValue, to_json_value
from autosaddler.v2.prompting.models import ToolCall, Usage
from autosaddler.v2.providers.base import AgentTransport, BaseAgentProvider, TransportOutcome, observe_usage
from autosaddler.v2.providers.workspace_renderer import RenderedSession, claude_renderer

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ClaudeProviderConfig:
    model: str = "claude-opus-4.6"
    effort: str | None = "max"
    permission_mode: str = "bypassPermissions"
    base_url: str = "https://api.anthropic.com"


class ClaudeAgentProvider(BaseAgentProvider):
    def __init__(
        self,
        config: ClaudeProviderConfig | None = None,
        *,
        transport: AgentTransport | None = None,
    ) -> None:
        resolved_config = config or ClaudeProviderConfig()
        super().__init__(claude_renderer(), transport or ClaudeSdkTransport(resolved_config))


class ClaudeSdkTransport:
    def __init__(self, config: ClaudeProviderConfig) -> None:
        self.config = config

    async def run(self, session: RenderedSession, _timeout_seconds: float) -> TransportOutcome:
        from claude_agent_sdk import AssistantMessage, ClaudeAgentOptions, ResultMessage, query
        from claude_agent_sdk.types import TextBlock, ToolUseBlock

        options = ClaudeAgentOptions(
            model=self.config.model,
            effort=self.config.effort,
            permission_mode=self.config.permission_mode,
            cwd=str(session.workspace),
            env={
                "ANTHROPIC_BASE_URL": self.config.base_url,
            },
            system_prompt=session.system_context,
            allowed_tools=list(session.allowed_tools),
            setting_sources=["project"],
        )
        raw_response = ""
        tool_calls: list[ToolCall] = []
        usage: list[Usage] = []
        claude_session_id: str | None = None
        session_summary: dict[str, JsonValue] | None = None

        try:
            async for message in query(prompt=session.task_prompt, options=options):
                if isinstance(message, AssistantMessage):
                    for block in message.content or ():
                        if isinstance(block, TextBlock):
                            raw_response = block.text
                        elif isinstance(block, ToolUseBlock):
                            arguments = _json_mapping(block.input or {})
                            tool_calls.append(ToolCall(tool=block.name, arguments=arguments))
                elif isinstance(message, ResultMessage):
                    claude_session_id = message.session_id
                    if message.result:
                        raw_response = message.result
                    session_summary = _claude_session_summary(message)
                    if message.model_usage:
                        items = _claude_model_usages(
                            message.model_usage,
                            configured_model=self.config.model,
                            configured_effort=self.config.effort,
                            message=message,
                        )
                    elif message.usage:
                        items = (
                            _usage(
                                message.usage,
                                self.config.model,
                                provider_cost=getattr(message, "total_cost_usd", None),
                                duration_seconds=_duration_seconds(message),
                                configured_settings={
                                    "model": self.config.model,
                                    "effort": self.config.effort,
                                },
                                agent_scope="session_inclusive",
                                extra_provider_metadata=_claude_result_metadata(message),
                            ),
                        )
                    else:
                        items = ()
                    for item in items:
                        usage.append(item)
                        observe_usage(item)
        finally:
            if claude_session_id is not None and session.trace_dir is not None:
                try:
                    _export_claude_session_state(
                        claude_session_id=claude_session_id,
                        workspace=session.workspace,
                        destination_root=session.trace_dir / "claude-session-state",
                        session_summary=session_summary,
                    )
                except Exception:
                    logger.warning("Failed to write Claude trace export manifest", exc_info=True)

        return TransportOutcome(
            raw_response=raw_response,
            tool_calls=tuple(tool_calls),
            usage=tuple(usage),
            usage_streamed=True,
        )


def _json_mapping(value: Mapping[str, Any]) -> Mapping[str, JsonValue]:
    converted = to_json_value(value)
    assert isinstance(converted, dict)
    return converted


def _usage(
    value: Mapping[str, Any],
    model: str,
    *,
    provider_cost: Any = None,
    duration_seconds: float | None = None,
    configured_settings: Mapping[str, JsonValue] | None = None,
    agent_scope: str = "unknown",
    extra_provider_metadata: Mapping[str, JsonValue] | None = None,
) -> Usage:
    input_tokens = int(value.get("input_tokens", 0) or 0)
    cached_input_tokens = int(value.get("cache_read_input_tokens", 0) or 0)
    cache_creation_input_tokens = int(value.get("cache_creation_input_tokens", 0) or 0)
    uncached_input_tokens = input_tokens + cache_creation_input_tokens
    normalized_input_tokens = cached_input_tokens + uncached_input_tokens
    output_tokens = int(value.get("output_tokens", 0) or 0)
    provider_total_tokens = value.get("total_tokens")
    cost = float(provider_cost) if isinstance(provider_cost, (int, float)) and not isinstance(provider_cost, bool) else None
    return Usage(
        role="optimizer",
        input_tokens=normalized_input_tokens,
        cached_input_tokens=cached_input_tokens,
        uncached_input_tokens=uncached_input_tokens,
        output_tokens=output_tokens,
        total_tokens=normalized_input_tokens + output_tokens,
        model=model,
        provider_cost=cost,
        provider_cost_unit="usd" if cost is not None else None,
        duration_seconds=duration_seconds,
        agent_scope=agent_scope,
        input_token_semantics="excludes_cached_tokens",
        provider_reported_input_tokens=input_tokens,
        provider_reported_total_tokens=int(provider_total_tokens) if provider_total_tokens is not None else None,
        total_tokens_is_inferred=provider_total_tokens is None,
        configured_settings={
            key: item
            for key, item in (configured_settings or {"model": model}).items()
            if item is not None
        },
        reported_settings={
            key: to_json_value(value[key])
            for key in (
                "reasoning_effort",
                "temperature",
                "top_p",
                "max_tokens",
                "max_output_tokens",
            )
            if value.get(key) is not None
        },
        provider_metadata={
            **{
                key: to_json_value(value[key])
                for key in ("finish_reason", "stop_reason")
                if value.get(key) is not None
            },
            **(
                {"cache_creation_input_tokens": cache_creation_input_tokens}
                if cache_creation_input_tokens
                else {}
            ),
            **dict(extra_provider_metadata or {}),
        },
    )


def _claude_model_usages(
    model_usage: Mapping[str, Any],
    *,
    configured_model: str,
    configured_effort: str | None,
    message: Any,
) -> tuple[Usage, ...]:
    result: list[Usage] = []
    metadata = _claude_result_metadata(message)
    for model, value in model_usage.items():
        if not isinstance(value, Mapping):
            raise TypeError(f"Claude model_usage[{model!r}] must be a mapping")
        normalized = {
            "input_tokens": _mapping_value(value, "inputTokens", "input_tokens"),
            "cache_read_input_tokens": _mapping_value(
                value, "cacheReadInputTokens", "cache_read_input_tokens"
            ),
            "cache_creation_input_tokens": _mapping_value(
                value, "cacheCreationInputTokens", "cache_creation_input_tokens"
            ),
            "output_tokens": _mapping_value(value, "outputTokens", "output_tokens"),
        }
        provider_metadata = dict(metadata)
        for key in ("webSearchRequests", "contextWindow", "maxOutputTokens", "canonicalModel", "provider"):
            if value.get(key) is not None:
                provider_metadata[key] = to_json_value(value[key])
        result.append(
            _usage(
                normalized,
                str(model),
                provider_cost=_mapping_value(value, "costUSD", "cost_usd", default=None),
                configured_settings={
                    "model": configured_model,
                    "effort": configured_effort,
                },
                agent_scope="session_inclusive",
                extra_provider_metadata=provider_metadata,
            )
        )
    return tuple(result)


def _mapping_value(
    value: Mapping[str, Any],
    primary: str,
    fallback: str,
    *,
    default: Any = 0,
) -> Any:
    return value.get(primary, value.get(fallback, default))


def _claude_result_metadata(message: Any) -> dict[str, JsonValue]:
    metadata: dict[str, JsonValue] = {
        "includes_subagents": True,
        "session_id": str(message.session_id),
        "session_duration_seconds": float(message.duration_ms) / 1_000,
        "duration_api_seconds": float(message.duration_api_ms) / 1_000,
        "num_turns": int(message.num_turns),
    }
    for key in ("stop_reason", "subtype", "api_error_status"):
        value = getattr(message, key, None)
        if value is not None:
            metadata[key] = to_json_value(value)
    total_cost = getattr(message, "total_cost_usd", None)
    if isinstance(total_cost, (int, float)) and not isinstance(total_cost, bool):
        metadata["session_total_cost_usd"] = float(total_cost)
    return metadata


def _claude_session_summary(message: Any) -> dict[str, JsonValue]:
    return {
        "scope": "session_inclusive",
        **_claude_result_metadata(message),
    }


def _export_claude_session_state(
    *,
    claude_session_id: str,
    workspace: Path,
    destination_root: Path,
    session_summary: Mapping[str, JsonValue] | None = None,
    source_project_dir: Path | None = None,
) -> None:
    if source_project_dir is None:
        from claude_agent_sdk._internal.sessions import _find_project_dir

        source_project_dir = _find_project_dir(str(workspace))
    destination_root.mkdir(parents=True, exist_ok=True)
    destination = destination_root / claude_session_id
    manifest_path = destination_root / "export-manifest.json"
    manifest: dict[str, JsonValue] = {
        "schema_version": "autosaddler-claude-trace-export/v1",
        "claude_session_id": claude_session_id,
        "destination": destination.name,
        "sensitive": True,
    }
    if session_summary is not None:
        manifest["session_usage"] = dict(session_summary)
    try:
        if source_project_dir is None:
            raise FileNotFoundError(f"Claude project directory does not exist for {workspace}")
        source_main = source_project_dir / f"{claude_session_id}.jsonl"
        source_subagents = source_project_dir / claude_session_id / "subagents"
        if source_main.is_symlink() or source_subagents.is_symlink():
            raise ValueError("Claude session state roots cannot be symlinks")
        if not source_main.is_file():
            raise FileNotFoundError(f"Claude transcript does not exist: {source_main}")
        paths = [source_main]
        if source_subagents.exists():
            paths.extend(source_subagents.rglob("*"))
        symlinks = [path for path in paths if path.is_symlink()]
        if symlinks:
            raise ValueError(f"Claude session state contains symlinks: {symlinks[0]}")
        if destination.exists():
            shutil.rmtree(destination)
        destination.mkdir()
        shutil.copy2(source_main, destination / source_main.name)
        if source_subagents.is_dir():
            shutil.copytree(source_subagents, destination / "subagents")
        transcript_files = sorted(destination.rglob("*.jsonl"))
        subagent_files = [path for path in transcript_files if "subagents" in path.parts]
        manifest.update(
            {
                "status": "exported",
                "files": [path.relative_to(destination).as_posix() for path in sorted(destination.rglob("*")) if path.is_file()],
                "summary": {
                    "main_message_count": _jsonl_line_count(destination / source_main.name),
                    "subagent_transcript_count": len(subagent_files),
                    "subagent_message_count": sum(_jsonl_line_count(path) for path in subagent_files),
                },
            }
        )
    except Exception as error:
        manifest.update({"status": "failed", "error_type": type(error).__name__, "error": str(error)[:2_000]})
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def _jsonl_line_count(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def _duration_seconds(message: Any) -> float | None:
    duration_ms = getattr(message, "duration_ms", None)
    if isinstance(duration_ms, (int, float)) and not isinstance(duration_ms, bool):
        return float(duration_ms) / 1_000
    return None