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
import re
import shlex
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

NANO_AIU_PER_USD = 100_000_000_000


class RateLimitError(Exception):
    """Raised when the API returns a rate-limit (429) error."""


class SDKSessionTimeoutError(Exception):
    """Raised after preserving metrics from a timed-out SDK session."""


class ContentFilterError(Exception):
    """Raised when a provider blocks a request or response by policy."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        error_code: str | None = None,
        error_type: str | None = None,
        provider_call_id: str | None = None,
        service_request_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code
        self.error_type = error_type
        self.provider_call_id = provider_call_id
        self.service_request_id = service_request_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "message": str(self),
            "status_code": self.status_code,
            "error_code": self.error_code,
            "error_type": self.error_type,
            "provider_call_id": self.provider_call_id,
            "service_request_id": self.service_request_id,
        }


def classify_content_filter_error(
    message: str,
    *,
    status_code: int | None = None,
    error_code: str | None = None,
    error_type: str | None = None,
    provider_call_id: str | None = None,
    service_request_id: str | None = None,
) -> ContentFilterError | None:
    """Return a structured policy error when provider evidence is conclusive."""
    evidence = " ".join(
        value for value in (message, error_code, error_type) if value
    ).lower()
    policy_markers = (
        "content_filter",
        "content filter",
        "content management policy",
        "responsibleaipolicyviolation",
        "filtered due to the prompt",
        "filtered due to the response",
    )
    if not any(marker in evidence for marker in policy_markers):
        return None
    if status_code not in (None, 400):
        return None
    return ContentFilterError(
        message,
        status_code=status_code,
        error_code=error_code,
        error_type=error_type,
        provider_call_id=provider_call_id,
        service_request_id=service_request_id,
    )


# ---------------------------------------------------------------------------
# Sub-agent usage capture
# ---------------------------------------------------------------------------
# ``claude_agent_sdk`` parses the CLI ``result`` event but keeps only the
# top-level ``usage`` field, which reflects the MAIN agent only.  The CLI also
# emits ``modelUsage`` -- per-model token/cost totals that INCLUDE every spawned
# sub-agent (Task tool).  We monkeypatch ``parse_message`` to stash that dict
# (keyed by session_id) so the recorder can persist authoritative, sub-agent-
# inclusive usage.  Best-effort and idempotent: if the SDK internals change the
# patch silently no-ops and normal operation continues.

_MODEL_USAGE_BY_SESSION: dict[str, Any] = {}


def _install_modelusage_capture() -> None:
    """Monkeypatch ``parse_message`` to preserve the CLI ``modelUsage`` field."""
    try:
        from claude_agent_sdk._internal import message_parser as _mp
    except Exception:
        return
    if getattr(_mp, "_modelusage_capture_installed", False):
        return
    _orig_parse = _mp.parse_message

    def _parse_with_capture(data: Any):
        msg = _orig_parse(data)
        try:
            if isinstance(data, dict) and data.get("type") == "result":
                model_usage = data.get("modelUsage")
                session_id = data.get("session_id")
                if model_usage and session_id:
                    _MODEL_USAGE_BY_SESSION[session_id] = model_usage
        except Exception:
            pass
        return msg

    _mp.parse_message = _parse_with_capture
    # ``_internal/client.py`` binds ``parse_message`` at import time, so patch
    # that module reference too (covers the ``query()`` code path).
    try:
        from claude_agent_sdk._internal import client as _client
        if hasattr(_client, "parse_message"):
            _client.parse_message = _parse_with_capture
    except Exception:
        pass
    _mp._modelusage_capture_installed = True


def _pop_model_usage(session_id: str | None) -> Any:
    """Retrieve (and clear) captured ``modelUsage`` for a finished session."""
    if not session_id:
        return None
    return _MODEL_USAGE_BY_SESSION.pop(session_id, None)


def aggregate_model_usage(model_usage: Any) -> dict[str, int | float] | None:
    """Aggregate Claude CLI per-model usage, including spawned sub-agents."""
    if not isinstance(model_usage, dict) or not model_usage:
        return None

    metric_aliases = {
        "input_tokens": ("input_tokens", "inputTokens"),
        "output_tokens": ("output_tokens", "outputTokens"),
        "cache_read_input_tokens": (
            "cache_read_input_tokens",
            "cacheReadInputTokens",
            "cache_read_tokens",
        ),
        "cache_creation_input_tokens": (
            "cache_creation_input_tokens",
            "cacheCreationInputTokens",
            "cache_write_tokens",
        ),
        "total_cost_usd": ("total_cost_usd", "cost_usd", "costUSD", "cost"),
    }
    top_level_metric = any(
        alias in model_usage
        for aliases in metric_aliases.values()
        for alias in aliases
    )
    entries = [model_usage] if top_level_metric else list(model_usage.values())
    totals: dict[str, int | float] = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_input_tokens": 0,
        "cache_creation_input_tokens": 0,
        "total_cost_usd": 0.0,
    }
    found = False
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        for metric, aliases in metric_aliases.items():
            for alias in aliases:
                value = entry.get(alias)
                if isinstance(value, int | float) and not isinstance(value, bool):
                    totals[metric] += value
                    found = True
                    break
    return totals if found else None


def _tiered_price(
    pricing: dict[str, Any],
    base_key: str,
    input_tokens: int,
) -> float | None:
    """Resolve a LiteLLM per-token rate, including long-context tiers."""
    candidates: list[tuple[int, float]] = []
    base_value = pricing.get(base_key)
    if isinstance(base_value, int | float) and not isinstance(base_value, bool):
        candidates.append((0, float(base_value)))

    prefix = f"{base_key}_above_"
    for key, value in pricing.items():
        if not key.startswith(prefix):
            continue
        if not isinstance(value, int | float) or isinstance(value, bool):
            continue
        match = re.fullmatch(r"(\d+)([km])_tokens", key.removeprefix(prefix))
        if match is None:
            continue
        multiplier = 1_000 if match.group(2) == "k" else 1_000_000
        candidates.append((int(match.group(1)) * multiplier, float(value)))

    applicable = [
        item for item in candidates if item[0] == 0 or input_tokens > item[0]
    ]
    return max(applicable, default=(0, None), key=lambda item: item[0])[1]


def estimate_copilot_usage_cost_usd(
    usage_info: list[dict[str, Any]],
) -> float | None:
    """Estimate complete Copilot usage from LiteLLM's versioned tariff table."""
    if not usage_info:
        return None
    try:
        import litellm
    except ImportError:
        return None

    total = 0.0
    for usage in usage_info:
        model = str(usage.get("model") or "")
        pricing = (
            litellm.model_cost.get(model)
            or litellm.model_cost.get(f"azure/{model}")
            or litellm.model_cost.get(f"azure_ai/{model}")
        )
        if not isinstance(pricing, dict):
            return None

        input_tokens = int(usage.get("input_tokens", 0) or 0)
        output_tokens = int(usage.get("output_tokens", 0) or 0)
        cache_read_tokens = int(
            usage.get("cache_read_input_tokens", 0)
            or usage.get("cache_read_tokens", 0)
            or 0
        )
        cache_creation_tokens = int(
            usage.get("cache_creation_input_tokens", 0)
            or usage.get("cache_write_tokens", 0)
            or 0
        )
        uncached_input_tokens = max(
            input_tokens - cache_read_tokens - cache_creation_tokens,
            0,
        )
        priced_buckets = (
            (
                uncached_input_tokens,
                _tiered_price(pricing, "input_cost_per_token", input_tokens),
            ),
            (
                output_tokens,
                _tiered_price(pricing, "output_cost_per_token", input_tokens),
            ),
            (
                cache_read_tokens,
                _tiered_price(
                    pricing,
                    "cache_read_input_token_cost",
                    input_tokens,
                ),
            ),
            (
                cache_creation_tokens,
                _tiered_price(
                    pricing,
                    "cache_creation_input_token_cost",
                    input_tokens,
                ),
            ),
        )
        if any(tokens > 0 and rate is None for tokens, rate in priced_buckets):
            return None
        total += sum(tokens * float(rate or 0.0) for tokens, rate in priced_buckets)
    return total


def summarize_copilot_usage(
    usage_info: list[dict[str, Any]],
) -> dict[str, Any]:
    """Aggregate request-level Copilot usage without treating ``cost`` as USD."""
    unique_usage: list[dict[str, Any]] = []
    seen_keys: set[tuple[str, str]] = set()
    duplicate_events = 0
    for usage in usage_info:
        key = None
        for identity_field in ("event_id", "api_call_id"):
            value = usage.get(identity_field)
            if value:
                key = (identity_field, str(value))
                break
        if key is not None and key in seen_keys:
            duplicate_events += 1
            continue
        if key is not None:
            seen_keys.add(key)
        unique_usage.append(usage)

    model_usage: dict[str, dict[str, Any]] = {}
    total_nano_aiu = 0.0
    has_nano_aiu = False
    for usage in unique_usage:
        model = str(usage.get("model") or "unknown")
        entry = model_usage.setdefault(
            model,
            {
                "llm_calls": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
                "reasoning_tokens": 0,
                "copilot_nano_aiu": 0.0,
                "metered_cost_usd": None,
                "estimated_cost_usd": None,
            },
        )
        entry["llm_calls"] += 1
        for metric_name in (
            "input_tokens",
            "output_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
            "reasoning_tokens",
        ):
            entry[metric_name] += int(usage.get(metric_name, 0) or 0)
        nano_aiu = usage.get("copilot_nano_aiu")
        if isinstance(nano_aiu, int | float) and not isinstance(nano_aiu, bool):
            has_nano_aiu = True
            entry["copilot_nano_aiu"] += float(nano_aiu)
            total_nano_aiu += float(nano_aiu)

    for model, entry in model_usage.items():
        if entry["copilot_nano_aiu"] > 0:
            entry["metered_cost_usd"] = (
                entry["copilot_nano_aiu"] / NANO_AIU_PER_USD
            )
        model_rows = [
            usage
            for usage in unique_usage
            if str(usage.get("model") or "unknown") == model
        ]
        entry["estimated_cost_usd"] = estimate_copilot_usage_cost_usd(model_rows)

    metered_cost_usd = (
        total_nano_aiu / NANO_AIU_PER_USD if has_nano_aiu else None
    )
    estimated_cost_usd = estimate_copilot_usage_cost_usd(unique_usage)
    if metered_cost_usd is not None:
        total_cost_usd = metered_cost_usd
        cost_source = "copilot_nano_aiu"
    else:
        total_cost_usd = estimated_cost_usd
        cost_source = (
            "litellm_pricing_estimate"
            if estimated_cost_usd is not None
            else "unavailable"
        )
    return {
        "usage": unique_usage,
        "usage_event_count": len(unique_usage),
        "duplicate_usage_event_count": duplicate_events,
        "llm_call_count": len(unique_usage),
        "copilot_nano_aiu": total_nano_aiu if has_nano_aiu else None,
        "metered_cost_usd": metered_cost_usd,
        "estimated_cost_usd": estimated_cost_usd,
        "total_cost_usd": total_cost_usd,
        "cost_source": cost_source,
        "cost_is_estimate": total_cost_usd is not None,
        "model_usage": model_usage or None,
    }


_install_modelusage_capture()


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SdkRetryConfig:
    """Retry policy shared by SDK-backed AutoSaddler sessions."""

    content_filter_max_retries: int = 3
    content_filter_retry_delays_s: tuple[float, ...] = (2.0, 5.0, 10.0)

    def delay_for_content_filter_retry(self, retry_index: int) -> float:
        if not self.content_filter_retry_delays_s:
            return 0.0
        index = min(
            max(retry_index - 1, 0),
            len(self.content_filter_retry_delays_s) - 1,
        )
        return max(0.0, float(self.content_filter_retry_delays_s[index]))


def build_sdk_retry_config(sdk_cfg: dict[str, Any]) -> SdkRetryConfig:
    """Build and validate the nested ``sdk.retry`` configuration."""
    retry_cfg = sdk_cfg.get("retry", {})
    if not isinstance(retry_cfg, dict):
        raise TypeError("sdk.retry must be a mapping")
    max_retries = int(retry_cfg.get("content_filter_max_retries", 3))
    if max_retries < 0:
        raise ValueError("sdk.retry.content_filter_max_retries must be >= 0")
    raw_delays = retry_cfg.get(
        "content_filter_retry_delays_s",
        (2.0, 5.0, 10.0),
    )
    if not isinstance(raw_delays, list | tuple):
        raise TypeError("sdk.retry.content_filter_retry_delays_s must be a list")
    delays = tuple(float(delay) for delay in raw_delays)
    if any(delay < 0 for delay in delays):
        raise ValueError(
            "sdk.retry.content_filter_retry_delays_s values must be >= 0"
        )
    return SdkRetryConfig(
        content_filter_max_retries=max_retries,
        content_filter_retry_delays_s=delays,
    )


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
    claude_auth_mode: str = "api_key"
    claude_azure_config_dir: str | None = None
    claude_azure_resource: str | None = None
    claude_custom_headers: dict[str, str] = field(default_factory=dict)
    claude_token_helper_ttl_ms: int = 2_700_000

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
    # BYOK: route the Copilot SDK's LLM calls through a custom
    # OpenAI-compatible endpoint (e.g. a copilot-api server). ``None`` uses
    # the default GitHub Copilot authentication/endpoint.
    copilot_provider: dict[str, Any] | None = None

    retry: SdkRetryConfig = field(default_factory=SdkRetryConfig)


def build_copilot_provider(
    copilot_cfg: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Build a BYOK provider config from an ``sdk.copilot`` config block.

    Returns a ``ProviderConfig``-shaped dict that routes the Copilot SDK's LLM
    calls through a custom OpenAI-compatible endpoint (e.g. a copilot-api
    server), or ``None`` when no ``base_url`` is configured — in which case the
    default GitHub Copilot authentication/endpoint is used.
    """
    if not isinstance(copilot_cfg, dict):
        return None
    provider = copilot_cfg.get("provider")
    if not isinstance(provider, dict):
        return None
    base_url = str(provider.get("base_url") or "").strip()
    if not base_url:
        return None
    out: dict[str, Any] = {"base_url": base_url}
    for key in (
        "type", "wire_api", "api_key", "bearer_token",
        "headers", "model_id", "wire_model",
    ):
        val = provider.get(key)
        if val not in (None, ""):
            out[key] = val
    out.setdefault("type", "openai")
    return out


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
        - ``wall_clock_s`` (float): Measured wall-clock duration of the session.
        - ``result_meta`` (dict | None): SDK-reported session metrics
          (``duration_ms``, ``duration_api_ms``, ``num_turns``,
          ``total_cost_usd``, ``session_id``) when tracked.
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
            provider=sdk_config.copilot_provider,
            system_prompt=system_prompt,
        )

    return await _run_claude_session(
        cwd=cwd,
        prompt=prompt,
        model=sdk_config.claude_model or model,
        timeout=timeout,
        base_url=sdk_config.claude_base_url,
        api_key=sdk_config.claude_api_key,
        auth_mode=sdk_config.claude_auth_mode,
        azure_config_dir=sdk_config.claude_azure_config_dir,
        azure_resource=sdk_config.claude_azure_resource,
        custom_headers=sdk_config.claude_custom_headers,
        token_helper_ttl_ms=sdk_config.claude_token_helper_ttl_ms,
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
    auth_mode: str,
    azure_config_dir: str | None,
    azure_resource: str | None,
    custom_headers: dict[str, str],
    token_helper_ttl_ms: int,
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
    result_meta: dict[str, Any] = {}
    raw_response = ""
    _saw_rate_limit = False  # Track rate-limit signals across messages

    stderr_lines: list[str] = []

    def _capture_stderr(line: str) -> None:
        stderr_lines.append(line)
        logger.debug("claude-cli stderr: %s", line.rstrip())

    sdk_env, sdk_settings = _build_sdk_auth(
        base_url=base_url,
        api_key=api_key,
        auth_mode=auth_mode,
        azure_config_dir=azure_config_dir,
        azure_resource=azure_resource,
        custom_headers=custom_headers,
        token_helper_ttl_ms=token_helper_ttl_ms,
    )
    sdk_kwargs: dict[str, Any] = {
        "model": cli_model,
        "permission_mode": permission_mode,
        "cwd": str(cwd),
        "env": sdk_env,
        "stderr": _capture_stderr,
        "debug_stderr": None,
    }
    if sdk_settings is not None:
        sdk_kwargs["settings"] = sdk_settings
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
        nonlocal turns, _saw_rate_limit, result_meta
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
                if track_events:
                    if message.usage:
                        usage_info.append(message.usage)
                    # Capture authoritative session metrics reported by the SDK
                    sdk_cost_usd = message.total_cost_usd
                    result_meta = {
                        "duration_ms": message.duration_ms,
                        "duration_api_ms": message.duration_api_ms,
                        "num_turns": message.num_turns,
                        "llm_call_count": message.num_turns,
                        "reported_cost_usd": None,
                        "metered_cost_usd": None,
                        "estimated_cost_usd": sdk_cost_usd,
                        "total_cost_usd": sdk_cost_usd,
                        "cost_source": (
                            "claude_sdk_list_price_estimate"
                            if sdk_cost_usd is not None
                            else "unavailable"
                        ),
                        "cost_is_estimate": sdk_cost_usd is not None,
                        "session_id": message.session_id,
                        # Per-model usage incl. sub-agents (None if SDK omits it).
                        "model_usage": _pop_model_usage(message.session_id),
                    }

        return last_text

    _t0 = time.perf_counter()
    try:
        raw_response = await asyncio.wait_for(
            _stream(), timeout=timeout,
        )
    except TimeoutError:
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
        content_filter_error = classify_content_filter_error(err_str)
        if content_filter_error is not None:
            raise content_filter_error from exc
        raise

    wall_clock_s = time.perf_counter() - _t0

    return {
        "raw_response": raw_response,
        "tool_calls": tool_calls,
        "turns": turns,
        "usage": usage_info or None,
        "wall_clock_s": wall_clock_s,
        "result_meta": result_meta or None,
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
    provider: dict[str, Any] | None = None,
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
    session_error: dict[str, Any] = {}
    seen_event_ids: set[str] = set()

    def _event_handler(event: Any) -> None:
        nonlocal turns, _saw_rate_limit, session_error
        event_id_value = getattr(event, "id", None)
        event_id = str(event_id_value) if event_id_value is not None else None
        if event_id is not None:
            if event_id in seen_event_ids:
                return
            seen_event_ids.add(event_id)
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
            aliases = {
                "input_tokens": ("input_tokens", "prompt_tokens", "promptTokens"),
                "output_tokens": (
                    "output_tokens",
                    "completion_tokens",
                    "completionTokens",
                ),
                "cache_read_input_tokens": (
                    "cache_read_tokens",
                    "cache_read_input_tokens",
                ),
                "cache_creation_input_tokens": (
                    "cache_write_tokens",
                    "cache_creation_input_tokens",
                ),
                "sdk_cost_multiplier": ("cost",),
                "model": ("model",),
                "duration_s": ("duration",),
                "reasoning_effort": ("reasoning_effort",),
                "reasoning_tokens": ("reasoning_tokens",),
                "api_call_id": ("api_call_id",),
                "api_endpoint": ("api_endpoint",),
                "provider_call_id": ("provider_call_id",),
                "service_request_id": ("service_request_id",),
                "initiator": ("initiator",),
                "parent_tool_call_id": ("parent_tool_call_id",),
                "finish_reason": ("finish_reason",),
                "content_filter_triggered": ("content_filter_triggered",),
            }
            for output_name, source_names in aliases.items():
                for source_name in source_names:
                    value = getattr(data, source_name, None)
                    if value is None:
                        continue
                    import datetime
                    if isinstance(value, datetime.timedelta):
                        value = value.total_seconds()
                    elif hasattr(value, "value"):
                        value = value.value
                    info[output_name] = value
                    break
            if event_id is not None:
                info["event_id"] = event_id
            timestamp = getattr(event, "timestamp", None)
            if timestamp is not None:
                info["timestamp"] = (
                    timestamp.isoformat()
                    if hasattr(timestamp, "isoformat")
                    else str(timestamp)
                )
            for event_attr in ("agent_id", "parent_id"):
                value = getattr(event, event_attr, None)
                if value is not None:
                    info[event_attr] = str(value)
            copilot_usage = getattr(data, "copilot_usage", None)
            if copilot_usage is not None:
                nano_aiu = getattr(copilot_usage, "total_nano_aiu", None)
                if nano_aiu is not None:
                    info["copilot_nano_aiu"] = nano_aiu
                details = getattr(copilot_usage, "_token_details", None)
                if details is not None:
                    info["copilot_token_details"] = [
                        detail.to_dict() if hasattr(detail, "to_dict") else str(detail)
                        for detail in details
                    ]
            if info:
                usage_info.append(info)

        elif event.type == SessionEventType.SESSION_ERROR:
            data = event.data
            session_error = {
                "message": getattr(data, "message", str(data)),
                "status_code": getattr(data, "status_code", None),
                "error_code": getattr(data, "error_code", None),
                "error_type": getattr(data, "error_type", None),
                "provider_call_id": getattr(data, "provider_call_id", None),
                "service_request_id": getattr(data, "service_request_id", None),
            }
            err_msg = session_error["message"]
            if "429" in err_msg or "rate_limit" in err_msg.lower():
                _saw_rate_limit = True

    # Build session config
    session_config: dict[str, Any] = {
        "model": cli_model,
        "on_permission_request": PermissionHandler.approve_all,
        "working_directory": str(cwd),
    }
    if provider:
        # BYOK: route LLM calls through a custom OpenAI-compatible endpoint.
        session_config["provider"] = provider
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
    session = None
    started_at = time.perf_counter()
    pending_error: BaseException | None = None
    outcome = "success"
    try:
        await client.start()
        session = await client.create_session(**session_config)
        session.on(_event_handler)

        resp = await session.send_and_wait(prompt, timeout=timeout)
        raw_response = resp.data.content if resp and hasattr(resp.data, "content") else ""

        if _saw_rate_limit:
            raise RateLimitError("Rate-limited (429) during Copilot session")

    except TimeoutError:
        logger.warning(
            "Copilot SDK session timed out after %.0fs", timeout,
        )
        outcome = "timeout"
        pending_error = SDKSessionTimeoutError(
            f"Copilot SDK session timed out after {timeout:.0f}s"
        )
    except RateLimitError as exc:
        outcome = "rate_limit"
        pending_error = exc
    except Exception as exc:
        err_str = str(exc)
        if (
            _saw_rate_limit
            or "429" in err_str
            or "rate_limit" in err_str.lower()
        ):
            outcome = "rate_limit"
            pending_error = RateLimitError(
                f"Rate-limited (429): {err_str}"
            )
            pending_error.__cause__ = exc
        else:
            content_filter_error = classify_content_filter_error(
                session_error.get("message") or err_str,
                status_code=session_error.get("status_code"),
                error_code=session_error.get("error_code"),
                error_type=session_error.get("error_type"),
                provider_call_id=session_error.get("provider_call_id"),
                service_request_id=session_error.get("service_request_id"),
            )
            if content_filter_error is not None:
                outcome = "content_filter"
                pending_error = content_filter_error
                pending_error.__cause__ = exc
            else:
                outcome = "error"
                pending_error = exc
    finally:
        if session is not None:
            get_events = getattr(session, "get_events", None)
            if callable(get_events):
                try:
                    for event in await get_events():
                        _event_handler(event)
                except Exception:
                    logger.debug(
                        "Failed to reconcile Copilot session event history",
                        exc_info=True,
                    )
            disconnect = getattr(session, "disconnect", None)
            if callable(disconnect):
                try:
                    await disconnect()
                except Exception:
                    logger.debug("Failed to disconnect Copilot session", exc_info=True)
        await client.stop()

    wall_clock_s = time.perf_counter() - started_at
    usage_summary = summarize_copilot_usage(usage_info)
    result = {
        "raw_response": raw_response,
        "tool_calls": tool_calls,
        "turns": turns,
        "usage": usage_summary["usage"] or None,
        "wall_clock_s": wall_clock_s,
        "result_meta": {
            "outcome": outcome,
            "duration_ms": round(wall_clock_s * 1000),
            "duration_api_ms": 0,
            "num_turns": turns,
            "llm_call_count": usage_summary["llm_call_count"],
            "usage_event_count": usage_summary["usage_event_count"],
            "duplicate_usage_event_count": usage_summary[
                "duplicate_usage_event_count"
            ],
            "copilot_nano_aiu": usage_summary["copilot_nano_aiu"],
            "reported_cost_usd": None,
            "metered_cost_usd": usage_summary["metered_cost_usd"],
            "estimated_cost_usd": usage_summary["estimated_cost_usd"],
            "total_cost_usd": usage_summary["total_cost_usd"],
            "cost_source": usage_summary["cost_source"],
            "cost_is_estimate": usage_summary["cost_is_estimate"],
            "session_id": getattr(session, "session_id", None),
            "model_usage": usage_summary["model_usage"],
        },
    }
    if pending_error is not None:
        pending_error.session_result = result
        raise pending_error
    return result


def _build_sdk_auth(
    *,
    base_url: str,
    api_key: str,
    auth_mode: str,
    azure_config_dir: str | None,
    azure_resource: str | None,
    custom_headers: dict[str, str],
    token_helper_ttl_ms: int,
) -> tuple[dict[str, str], str | None]:
    """Build Claude Code environment and optional credential helper settings."""
    if auth_mode == "api_key":
        return {
            "ANTHROPIC_BASE_URL": base_url,
            "ANTHROPIC_API_KEY": api_key,
        }, None

    if auth_mode != "azure_cli_helper":
        raise ValueError(f"Unsupported Claude SDK auth mode: {auth_mode}")
    if not azure_config_dir:
        raise ValueError("sdk.claude.azure_config_dir is required for azure_cli_helper")
    if not azure_resource:
        raise ValueError("sdk.claude.azure_resource is required for azure_cli_helper")
    if token_helper_ttl_ms <= 0:
        raise ValueError("sdk.claude.token_helper_ttl_ms must be positive")

    profile_dir = Path(azure_config_dir).expanduser().resolve()
    if not profile_dir.is_dir():
        raise ValueError(f"Azure CLI profile directory not found: {profile_dir}")

    header_lines = []
    for name, value in custom_headers.items():
        if "\n" in name or "\n" in value:
            raise ValueError("Claude SDK custom headers cannot contain newlines")
        header_lines.append(f"{name}: {value}")

    sdk_env = {
        "AZURE_CONFIG_DIR": str(profile_dir),
        "ANTHROPIC_BASE_URL": base_url,
        "ANTHROPIC_API_KEY": "",
        "ANTHROPIC_AUTH_TOKEN": "",
        "CLAUDE_CODE_API_KEY_HELPER_TTL_MS": str(token_helper_ttl_ms),
    }
    if header_lines:
        sdk_env["ANTHROPIC_CUSTOM_HEADERS"] = "\n".join(header_lines)

    helper_command = shlex.join(
        [
            "az",
            "account",
            "get-access-token",
            "--resource",
            azure_resource,
            "--query",
            "accessToken",
            "--output",
            "tsv",
        ]
    )
    return sdk_env, json.dumps({"apiKeyHelper": helper_command})
