from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from autosaddler.v2.core.domain import JsonValue, to_json_value
from autosaddler.v2.prompting.models import Usage
from autosaddler.v2.providers.base import AgentTransport, BaseAgentProvider, TransportOutcome, observe_usage
from autosaddler.v2.providers.workspace_renderer import RenderedSession, taskferry_renderer

logger = logging.getLogger(__name__)

_RESULT_FIELDS = (
    "message,tokens,cost,sessionId,exitCode,signal,spawnError,failureReason,failureDetail,incomplete"
)
_SETTLED_STATUSES = frozenset({"done", "crashed", "cancelled", "unknown"})
_UNSETTLED_STATUSES = frozenset({"queued", "running"})


@dataclass(frozen=True, slots=True)
class TaskferryProviderConfig:
    model: str = "openai/gpt-5.6"
    variant: str | None = None
    executor: str | None = None
    sandboxed: bool = True
    executable: str = "taskferry"

    def __post_init__(self) -> None:
        for name in ("model", "executable"):
            if not getattr(self, name):
                raise ValueError(f"Taskferry {name} cannot be empty")
        if self.executor is not None and self.executor not in ("opencode", "pi"):
            raise ValueError(f"Taskferry executor must be 'opencode' or 'pi', got {self.executor!r}")


class TaskferryAgentProvider(BaseAgentProvider):
    def __init__(
        self,
        config: TaskferryProviderConfig | None = None,
        *,
        transport: AgentTransport | None = None,
    ) -> None:
        resolved_config = config or TaskferryProviderConfig()
        super().__init__(taskferry_renderer(), transport or TaskferryCliTransport(resolved_config))


class TaskferryCliTransport:
    """Dispatch optimizer sessions through the taskferry CLI.

    Every session is dispatched with ``--no-overlay`` so the worker's writes
    land directly in the session workspace instead of a copy-on-write overlay
    awaiting an accept step: the session workspace is AutoSaddler-owned
    scratch space and the structured output file must be readable by the
    engine immediately after settlement.
    """

    def __init__(self, config: TaskferryProviderConfig) -> None:
        self.config = config

    async def run(self, session: RenderedSession, timeout_seconds: float) -> TransportOutcome:
        started_wall = time.monotonic()
        payload = await self._dispatch(session)
        task_id = payload.get("id")
        if not isinstance(task_id, str) or not task_id:
            raise RuntimeError(f"taskferry dispatch returned no task id: {payload!r}")
        result_payload: Mapping[str, JsonValue] | None = None
        error_summary: Mapping[str, JsonValue] | None = None
        try:
            status_payload = await self._run_cli(("wait", task_id, "--timeout", _wait_duration(timeout_seconds)))
            status = str(status_payload.get("status") or "")
            if status in _UNSETTLED_STATUSES:
                raise TimeoutError(f"taskferry task {task_id} did not settle within {timeout_seconds:.0f}s")
            result_payload = await self._run_cli(("result", task_id, "--fields", _RESULT_FIELDS))
            status = str(result_payload.get("status") or "")
            if status != "done":
                raise RuntimeError(
                    f"taskferry task {task_id} ended with status {status!r}: {_failure_detail(result_payload)}"
                )
        except BaseException as error:
            error_summary = _error_summary(error)
            await self._cancel_best_effort(task_id)
            raise
        finally:
            if session.trace_dir is not None:
                try:
                    _export_taskferry_session_state(
                        task_id=task_id,
                        destination_root=session.trace_dir / "taskferry-session-state",
                        config=self.config,
                        result_payload=result_payload,
                        error_summary=error_summary,
                    )
                except Exception:
                    logger.warning("Failed to write taskferry trace export manifest", exc_info=True)
        usage = _taskferry_usage(
            result_payload.get("tokens"),
            self.config,
            duration_seconds=time.monotonic() - started_wall,
            incomplete=result_payload.get("incomplete"),
            session_id=_text(result_payload.get("sessionId")),
        )
        for item in usage:
            observe_usage(item)
        return TransportOutcome(
            raw_response=_text(result_payload.get("message")) or "",
            tool_calls=(),
            usage=tuple(usage),
            usage_streamed=True,
        )

    async def _dispatch(self, session: RenderedSession) -> Mapping[str, JsonValue]:
        arguments: list[str] = [
            "dispatch",
            "--prompt",
            "-",
            "--directory",
            str(session.workspace),
            "--model",
            self.config.model,
            "--no-overlay",
            "--class",
            "autosaddler-v2",
        ]
        if self.config.variant is not None:
            arguments.extend(("--variant", self.config.variant))
        if self.config.executor is not None:
            arguments.extend(("--executor", self.config.executor))
        if not self.config.sandboxed:
            arguments.append("--no-sandbox")
        payload = await self._run_cli(tuple(arguments), stdin=_dispatch_prompt(session))
        if not isinstance(payload, Mapping):
            raise ValueError("taskferry dispatch returned a non-object response")
        return payload

    async def _run_cli(
        self,
        arguments: Sequence[str],
        *,
        stdin: str | None = None,
    ) -> Mapping[str, JsonValue]:
        process = await asyncio.create_subprocess_exec(
            self.config.executable,
            *arguments,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_bytes, stderr_bytes = await process.communicate(input=None if stdin is None else stdin.encode("utf-8"))
        stdout = stdout_bytes.decode("utf-8", errors="replace")
        stderr = stderr_bytes.decode("utf-8", errors="replace")
        if process.returncode != 0:
            raise RuntimeError(
                f"taskferry {' '.join(arguments)} failed (exit {process.returncode}): {stderr[-2000:]}"
            )
        return _decode_toon(stdout)

    async def _cancel_best_effort(self, task_id: str) -> None:
        try:
            process = await asyncio.create_subprocess_exec(
                self.config.executable,
                "cancel",
                task_id,
                stdout=asyncio.subprocess.DEVNULL,
                stderr=asyncio.subprocess.DEVNULL,
            )
            await process.wait()
        except Exception:
            logger.warning("Failed to cancel taskferry task %s", task_id, exc_info=True)


def _dispatch_prompt(session: RenderedSession) -> str:
    prompt = "\n\n".join(
        part for part in (session.system_context.rstrip(), session.task_prompt.rstrip()) if part
    )
    return (
        f"{prompt}\n\n"
        "Follow `.autosaddler/session_output_schema.json` and write the final structured output "
        "as a JSON object to `.autosaddler/session_output.json`.\n"
    )


def _wait_duration(timeout_seconds: float) -> str:
    seconds = max(1, int(timeout_seconds))
    return f"{seconds}s"


def _failure_detail(payload: Mapping[str, JsonValue]) -> str:
    fields = {
        key: _text(payload.get(key))
        for key in ("failureReason", "failureDetail", "spawnError", "exitCode", "signal")
        if payload.get(key) is not None
    }
    if not fields:
        return "no failure detail reported"
    return " ".join(f"{key}={value}" for key, value in fields.items())


def _error_summary(error: BaseException) -> dict[str, JsonValue]:
    return {"error_type": type(error).__name__, "error": str(error)[:2_000]}


def _taskferry_usage(
    tokens: JsonValue | None,
    config: TaskferryProviderConfig,
    *,
    duration_seconds: float | None = None,
    incomplete: JsonValue | None = None,
    session_id: str | None = None,
) -> tuple[Usage, ...]:
    if not isinstance(tokens, Mapping):
        if tokens is not None:
            logger.warning("Ignoring unsupported taskferry tokens payload of type %s", type(tokens).__name__)
        return ()
    cached_input = _int_or(_nested(tokens, "cacheRead"), _nested(tokens, "cache", "read"), 0)
    cache_creation = _int_or(_nested(tokens, "cacheWrite"), _nested(tokens, "cache", "write"), 0)
    reported_input = _int_or(_nested(tokens, "input"), 0)
    output = _int_or(_nested(tokens, "output"), 0)
    reasoning = _int_or(_nested(tokens, "reasoning"), 0)
    provider_total = _int_or(_nested(tokens, "totalTokens"), _nested(tokens, "total"), None)
    cost_value = _nested(tokens, "cost", "total")
    if cost_value is None:
        cost_value = _nested(tokens, "cost")
    provider_cost = _float_or(cost_value)
    uncached_input = reported_input + cache_creation
    input_tokens = cached_input + uncached_input
    incomplete_flag = bool(incomplete)
    return (
        Usage(
            role="optimizer",
            model=config.model,
            input_tokens=input_tokens,
            cached_input_tokens=cached_input,
            uncached_input_tokens=uncached_input,
            output_tokens=output,
            reasoning_tokens=min(reasoning, output),
            total_tokens=input_tokens + output,
            provider_cost=provider_cost,
            provider_cost_unit="usd" if provider_cost is not None else None,
            duration_seconds=duration_seconds,
            status="failed" if incomplete_flag else "success",
            usage_incomplete=incomplete_flag,
            provider_correlation_id=session_id,
            input_token_semantics="excludes_cached_tokens",
            provider_reported_input_tokens=reported_input,
            provider_reported_total_tokens=provider_total,
            total_tokens_is_inferred=provider_total is None,
            configured_settings=_without_none(
                {
                    "model": config.model,
                    "variant": config.variant,
                    "executor": config.executor,
                }
            ),
            provider_metadata=(
                {"incomplete": True} if incomplete_flag else {}
            ),
        ),
    )


def _nested(value: Mapping[str, Any], *keys: str) -> JsonValue | None:
    current: Any = value
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
        if current is None:
            return None
    if isinstance(current, (str, int, float, bool)) or current is None:
        return current
    return None


def _int_or(*values: JsonValue | None) -> int | None:
    for value in values:
        if isinstance(value, bool):
            continue
        if isinstance(value, int):
            return value
        if isinstance(value, float) and value.is_integer():
            return int(value)
    return None


def _float_or(value: JsonValue | None) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _text(value: JsonValue | None) -> str | None:
    if value is None:
        return None
    return str(value)


def _without_none(values: Mapping[str, JsonValue | None]) -> dict[str, JsonValue]:
    return {key: value for key, value in values.items() if value is not None}


def _export_taskferry_session_state(
    *,
    task_id: str,
    destination_root: Path,
    config: TaskferryProviderConfig,
    result_payload: Mapping[str, JsonValue] | None = None,
    error_summary: Mapping[str, JsonValue] | None = None,
) -> None:
    destination_root.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, JsonValue] = {
        "schema_version": "autosaddler-taskferry-trace-export/v1",
        "taskferry_task_id": task_id,
        "model": config.model,
        "executor": config.executor,
        "variant": config.variant,
        "sensitive": True,
        "transcript": f"Available via: taskferry result {task_id} --full",
    }
    if result_payload is not None:
        manifest.update(
            {
                "status": _text(result_payload.get("status")),
                "session_id": _text(result_payload.get("sessionId")),
                "exit_code": _nested(result_payload, "exitCode"),
                "incomplete": bool(result_payload.get("incomplete")),
                "tokens": result_payload.get("tokens"),
                "message": _text(result_payload.get("message")),
            }
        )
        for key in ("failureReason", "failureDetail", "spawnError"):
            value = result_payload.get(key)
            if value is not None:
                manifest[key] = value
    else:
        manifest.update(
            {
                "status": "unsettled",
                **dict(error_summary or {}),
            }
        )
    (destination_root / "export-manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Minimal TOON decoder for taskferry CLI output
# ---------------------------------------------------------------------------

_LITERALS = {"null": None, "true": True, "false": False}
_NUMBER_RE = re.compile(r"^-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?$")
_ARRAY_HEADER_RE = re.compile(r"^(.+)\[(\d+)\]$")
_KEY_VALUE_RE = re.compile(r"^([^:]+):\s?(.*)$")


def _decode_toon(text: str) -> Mapping[str, JsonValue]:
    """Decode the TOON object emitted on stdout by taskferry CLI commands.

    Only the subset the CLI emits for dispatch/wait/result is supported:
    objects with bare or JSON-quoted scalar values, nested objects, and
    single-line arrays. Unknown shapes raise ValueError.
    """
    lines = [line for line in text.splitlines() if line.strip()]
    if not lines:
        raise ValueError("taskferry CLI produced empty output")
    value, _ = _toon_block(lines, 0, 0)
    if not isinstance(value, dict):
        raise ValueError("taskferry CLI output must be a single TOON object")
    return value


def _toon_block(lines: list[str], index: int, indent: int) -> tuple[JsonValue, int]:
    result: dict[str, JsonValue] = {}
    count = len(lines)
    while index < count:
        line = lines[index]
        current_indent = len(line) - len(line.lstrip(" "))
        if current_indent < indent:
            break
        if current_indent != indent:
            raise ValueError(f"unexpected indentation in taskferry CLI output: {line!r}")
        content = line.strip()
        match = _KEY_VALUE_RE.match(content)
        if not match:
            raise ValueError(f"unsupported line in taskferry CLI output: {line!r}")
        key, value_text = match.group(1), match.group(2)
        array_match = _ARRAY_HEADER_RE.match(key)
        if array_match:
            key = array_match.group(1)
        if value_text == "":
            if array_match:
                raise ValueError(f"unsupported multiline array in taskferry CLI output: {key!r}")
            child, index = _toon_block(lines, index + 1, indent + 2)
            result[key] = to_json_value(child)
            continue
        if array_match:
            result[key] = to_json_value(_toon_list(value_text))
        else:
            result[key] = _toon_scalar(value_text)
        index += 1
    return result, index


def _toon_scalar(text: str) -> JsonValue:
    if text == "":
        return ""
    if text in _LITERALS:
        return _LITERALS[text]
    if text.startswith('"'):
        try:
            value = json.loads(text)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid quoted string in taskferry CLI output: {text[:80]!r}") from error
        return value
    if _NUMBER_RE.match(text):
        return float(text) if "." in text or "e" in text.lower() else int(text)
    return text


def _toon_list(text: str) -> list[JsonValue]:
    parts: list[str] = []
    current = ""
    quoted = False
    index = 0
    while index < len(text):
        char = text[index]
        if char == "\\" and quoted and index + 1 < len(text):
            current += text[index : index + 2]
            index += 2
            continue
        if char == '"':
            current += char
            quoted = not quoted
        elif char == "," and not quoted:
            parts.append(current)
            current = ""
        else:
            current += char
        index += 1
    parts.append(current)
    return [_toon_scalar(part) for part in parts]