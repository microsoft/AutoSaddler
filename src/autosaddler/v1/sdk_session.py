"""SDK session runner for AutoSaddler.

Provides ``run_sdk_session()`` — the single entry point for all SDK
interactions (diagnosis, patch execution, reflection, candidate selection).

Supports two backends:
- ``"claude"``: Claude Agent SDK (``claude_agent_sdk``)
- ``"copilot"``: GitHub Copilot SDK (``copilot``)
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class RateLimitError(Exception):
    """Raised when the API returns a rate-limit (429) error."""


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SdkConfig:
    """SDK configuration for Claude Agent SDK and GitHub Copilot SDK."""

    # Backend selection: "claude" or "copilot"
    backend: str = "claude"

    # Claude Agent SDK settings
    claude_base_url: str = "https://api.anthropic.com"
    claude_api_key: str = ""
    claude_permission_mode: str = "bypassPermissions"
    claude_model: str | None = None

    # Claude Code session settings
    claude_effort: str | None = "max"
    claude_allowed_tools: list[str] | None = None
    claude_setting_sources: list[str] | None = None
    claude_mcp_servers: dict | None = None
    claude_plugins: list | None = None

    # GitHub Copilot SDK settings
    copilot_model: str | None = None
    copilot_effort: str | None = "max"
    copilot_allowed_tools: list[str] | None = None


# ---------------------------------------------------------------------------
# Effort mapping
# ---------------------------------------------------------------------------

_CLAUDE_TO_COPILOT_EFFORT = {
    "max": "high",
    "high": "high",
    "medium": "medium",
    "low": "low",
    "xhigh": "xhigh",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def run_sdk_session(
    cwd: str | Path,
    prompt: str,
    *,
    model: str = "Claude Opus 4.6",
    timeout: float = 600.0,
    sdk_config: SdkConfig | None = None,
    track_events: bool = False,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    """Run an SDK session (Claude Agent SDK or GitHub Copilot SDK).

    Parameters
    ----------
    cwd:
        Working directory for the session.
    prompt:
        The task/instruction prompt to send.
    model:
        Model name (e.g. ``"Claude Opus 4.6"`` or ``"claude-opus-4.6"``).
    timeout:
        Session timeout in seconds.
    sdk_config:
        SDK configuration. Uses defaults if ``None``.
    track_events:
        If ``True``, capture tool calls, turn counts, and usage info.
    system_prompt:
        If set, appended to the built-in system prompt.

    Returns
    -------
    dict with keys:
        - ``raw_response`` (str): The final text response.
        - ``tool_calls`` (list[dict]): Tool invocations (if tracked).
        - ``turns`` (int): Number of assistant turns.
        - ``usage`` (list[dict] | None): Token usage info.
    """
    if sdk_config is None:
        sdk_config = SdkConfig()

    if sdk_config.backend == "copilot":
        return await _run_copilot_session(
            cwd=cwd,
            prompt=prompt,
            model=sdk_config.copilot_model or model,
            timeout=timeout,
            track_events=track_events,
            reasoning_effort=sdk_config.copilot_effort,
            excluded_tools=None,
            allowed_tools=sdk_config.copilot_allowed_tools,
            system_prompt=system_prompt,
        )

    return await _run_claude_session(
        cwd=cwd,
        prompt=prompt,
        model=sdk_config.claude_model or model,
        timeout=timeout,
        base_url=sdk_config.claude_base_url,
        api_key=sdk_config.claude_api_key,
        permission_mode=sdk_config.claude_permission_mode,
        track_events=track_events,
        effort=sdk_config.claude_effort,
        allowed_tools=sdk_config.claude_allowed_tools,
        setting_sources=sdk_config.claude_setting_sources,
        mcp_servers=sdk_config.claude_mcp_servers,
        plugins=sdk_config.claude_plugins,
        system_prompt=system_prompt,
    )


# ---------------------------------------------------------------------------
# Claude Agent SDK implementation
# ---------------------------------------------------------------------------

async def _run_claude_session(
    *,
    cwd: str | Path,
    prompt: str,
    model: str,
    timeout: float,
    base_url: str,
    api_key: str,
    permission_mode: str,
    track_events: bool,
    effort: str | None = None,
    allowed_tools: list[str] | None = None,
    setting_sources: list[str] | None = None,
    mcp_servers: dict | None = None,
    plugins: list | None = None,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    """Run a session via the Claude Agent SDK (``claude_agent_sdk``)."""
    from claude_agent_sdk import (  # type: ignore[import-untyped]
        AssistantMessage,
        ClaudeAgentOptions,
        ResultMessage,
        query,
    )
    from claude_agent_sdk.types import TextBlock, ToolUseBlock  # type: ignore[import-untyped]

    # Normalize model name: "Claude Opus 4.6" -> "claude-opus-4.6"
    _MODEL_ALIASES = {
        "opus": "claude-opus-4.6",
        "sonnet": "claude-sonnet-4.6",
        "haiku": "claude-haiku-4.5",
    }
    cli_model = model.lower().replace(" ", "-") if model else model
    cli_model = _MODEL_ALIASES.get(cli_model, cli_model)

    tool_calls: list[dict[str, Any]] = []
    turns = 0
    usage_info: list[dict[str, Any]] = []
    raw_response = ""
    _saw_rate_limit = False  # Track rate-limit signals across messages

    stderr_lines: list[str] = []

    def _capture_stderr(line: str) -> None:
        stderr_lines.append(line)
        logger.debug("claude-cli stderr: %s", line.rstrip())

    sdk_kwargs: dict[str, Any] = {
        "model": cli_model,
        "permission_mode": permission_mode,
        "cwd": str(cwd),
        "env": {
            "ANTHROPIC_BASE_URL": base_url,
            "ANTHROPIC_API_KEY": api_key,
        },
        "stderr": _capture_stderr,
        "debug_stderr": None,
    }
    if effort is not None:
        sdk_kwargs["effort"] = effort
    if allowed_tools is not None:
        sdk_kwargs["allowed_tools"] = allowed_tools
    if setting_sources is not None:
        sdk_kwargs["setting_sources"] = setting_sources
    if mcp_servers is not None:
        sdk_kwargs["mcp_servers"] = mcp_servers
    if plugins is not None:
        sdk_kwargs["plugins"] = plugins
    if system_prompt is not None:
        sdk_kwargs["system_prompt"] = system_prompt

    options = ClaudeAgentOptions(**sdk_kwargs)

    async def _stream() -> str:
        nonlocal turns, _saw_rate_limit
        last_text = ""

        async for message in query(prompt=prompt, options=options):
            if isinstance(message, AssistantMessage):
                if track_events:
                    turns += 1
                    for block in (message.content or []):
                        if isinstance(block, ToolUseBlock):
                            entry: dict[str, Any] = {
                                "tool": block.name,
                            }
                            if block.input:
                                entry["arguments"] = block.input
                            tool_calls.append(entry)
                        elif isinstance(block, TextBlock):
                            last_text = block.text
                # Detect rate-limit in assistant text (fires before ResultMessage)
                for block in (message.content or []):
                    if isinstance(block, TextBlock):
                        txt = block.text
                        if "429" in txt and "rate_limit" in txt.lower():
                            _saw_rate_limit = True

            elif isinstance(message, ResultMessage):
                if message.result:
                    last_text = message.result
                    # Detect rate-limit errors surfaced as result text
                    if "429" in last_text and "rate_limit" in last_text.lower():
                        raise RateLimitError(last_text)
                if track_events and message.usage:
                    usage_info.append(message.usage)

        return last_text

    try:
        raw_response = await asyncio.wait_for(
            _stream(), timeout=timeout,
        )
    except asyncio.TimeoutError:
        logger.warning(
            "Claude SDK session timed out after %.0fs", timeout,
        )
    except RateLimitError:
        raise
    except Exception as exc:
        if stderr_lines:
            logger.error(
                "Claude SDK session failed. stderr:\n%s",
                "\n".join(stderr_lines[-50:]),
            )
        # Detect rate-limit errors (429) from the CLI error message or
        # from streamed messages captured during _stream().
        err_str = str(exc)
        if (
            _saw_rate_limit
            or "429" in err_str
            or "rate_limit" in err_str.lower()
        ):
            raise RateLimitError(
                f"Rate-limited (429): {err_str}"
            ) from exc
        raise

    return {
        "raw_response": raw_response,
        "tool_calls": tool_calls,
        "turns": turns,
        "usage": usage_info or None,
    }


# ---------------------------------------------------------------------------
# GitHub Copilot SDK implementation
# ---------------------------------------------------------------------------

async def _run_copilot_session(
    *,
    cwd: str | Path,
    prompt: str,
    model: str,
    timeout: float,
    track_events: bool,
    reasoning_effort: str | None = None,
    excluded_tools: list[str] | None = None,
    allowed_tools: list[str] | None = None,
    system_prompt: str | None = None,
) -> dict[str, Any]:
    """Run a session via the GitHub Copilot SDK (``copilot``)."""
    from copilot import CopilotClient, PermissionHandler  # type: ignore[import-untyped]
    from copilot.generated.session_events import SessionEventType  # type: ignore[import-untyped]

    # Normalize model name: "Claude Opus 4.6" -> "claude-opus-4.6"
    cli_model = model.lower().replace(" ", "-") if model else model

    tool_calls: list[dict[str, Any]] = []
    turns = 0
    usage_info: list[dict[str, Any]] = []
    raw_response = ""
    _saw_rate_limit = False

    def _event_handler(event: Any) -> None:
        nonlocal turns, _saw_rate_limit
        if event.type == SessionEventType.TOOL_EXECUTION_START:
            entry: dict[str, Any] = {"tool": "unknown"}
            data = event.data
            # SDK 1.0 uses "tool_name"
            val = getattr(data, "tool_name", None)
            if val:
                entry["tool"] = val
            # SDK 1.0 returns arguments as a dict (or its repr string)
            args_val = getattr(data, "arguments", None)
            if args_val is not None:
                if isinstance(args_val, dict):
                    entry["arguments"] = args_val
                elif isinstance(args_val, str):
                    try:
                        entry["arguments"] = json.loads(args_val)
                    except (json.JSONDecodeError, TypeError):
                        import ast
                        try:
                            entry["arguments"] = ast.literal_eval(args_val)
                        except Exception:
                            entry["arguments"] = args_val[:300]
            tool_calls.append(entry)

        elif event.type == SessionEventType.TOOL_EXECUTION_COMPLETE:
            data = event.data
            preview = None
            # SDK 1.0: result is a ToolExecutionCompleteResult with .content
            result_obj = getattr(data, "result", None)
            if result_obj is not None:
                content = getattr(result_obj, "content", None)
                if content is not None:
                    preview = str(content)[:500]
                else:
                    preview = str(result_obj)[:500]
            if tool_calls:
                tool_calls[-1]["result_preview"] = preview

        elif event.type == SessionEventType.ASSISTANT_TURN_START:
            turns += 1

        elif event.type == SessionEventType.ASSISTANT_USAGE:
            data = event.data
            info: dict[str, Any] = {}
            for attr in (
                # Copilot SDK naming
                "input_tokens", "output_tokens",
                "cache_read_tokens", "cache_write_tokens",
                "cost", "model", "duration",
                "reasoning_effort", "reasoning_tokens",
                # Claude SDK naming (fallback)
                "promptTokens", "completionTokens", "totalTokens",
                "prompt_tokens", "completion_tokens", "total_tokens",
            ):
                val = getattr(data, attr, None)
                if val is not None:
                    # Convert timedelta to seconds for JSON serialization
                    import datetime
                    if isinstance(val, datetime.timedelta):
                        val = val.total_seconds()
                    info[attr] = val
            if info:
                usage_info.append(info)

        elif event.type == SessionEventType.SESSION_ERROR:
            err_msg = getattr(event.data, "message", str(event.data))
            if "429" in err_msg or "rate_limit" in err_msg.lower():
                _saw_rate_limit = True

    # Build session config
    session_config: dict[str, Any] = {
        "model": cli_model,
        "on_permission_request": PermissionHandler.approve_all,
        "working_directory": str(cwd),
    }
    if reasoning_effort is not None:
        # Map Claude effort values to Copilot reasoning_effort
        mapped = _CLAUDE_TO_COPILOT_EFFORT.get(reasoning_effort, reasoning_effort)
        session_config["reasoning_effort"] = mapped
    if allowed_tools is not None:
        session_config["available_tools"] = allowed_tools
    elif excluded_tools is not None:
        session_config["excluded_tools"] = excluded_tools
    if system_prompt is not None:
        session_config["system_message"] = {
            "mode": "append",
            "content": system_prompt,
        }

    client = CopilotClient(working_directory=str(cwd))
    try:
        await client.start()
        session = await client.create_session(**session_config)
        if track_events:
            session.on(_event_handler)

        resp = await session.send_and_wait(prompt, timeout=timeout)
        raw_response = resp.data.content if resp and hasattr(resp.data, "content") else ""

        if _saw_rate_limit:
            raise RateLimitError("Rate-limited (429) during Copilot session")

    except asyncio.TimeoutError:
        logger.warning(
            "Copilot SDK session timed out after %.0fs", timeout,
        )
    except RateLimitError:
        raise
    except Exception as exc:
        err_str = str(exc)
        if (
            _saw_rate_limit
            or "429" in err_str
            or "rate_limit" in err_str.lower()
        ):
            raise RateLimitError(
                f"Rate-limited (429): {err_str}"
            ) from exc
        raise
    finally:
        await client.stop()

    return {
        "raw_response": raw_response,
        "tool_calls": tool_calls,
        "turns": turns,
        "usage": usage_info or None,
    }
