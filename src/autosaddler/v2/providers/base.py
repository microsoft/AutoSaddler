from __future__ import annotations

import asyncio
import json
from contextvars import ContextVar
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from autosaddler.v2.core.domain import Cost, JsonValue
from autosaddler.v2.prompting.models import (
    SessionRequest,
    SessionResult,
    ToolCall,
    Usage,
    session_output_validation_error,
)
from autosaddler.v2.providers.workspace_renderer import RenderedSession, WorkspaceRenderer

_CURRENT_USAGE_OBSERVER: ContextVar[object | None] = ContextVar(
    "autosaddler_v2_usage_observer",
    default=None,
)


@dataclass(frozen=True, slots=True)
class TransportOutcome:
    raw_response: str
    structured_output: Mapping[str, JsonValue] | None = None
    tool_calls: tuple[ToolCall, ...] = ()
    usage: tuple[Usage, ...] = ()
    usage_streamed: bool = False


class AgentTransport(Protocol):
    async def run(self, session: RenderedSession, timeout_seconds: float) -> TransportOutcome: ...


class BaseAgentProvider:
    def __init__(self, renderer: WorkspaceRenderer, transport: AgentTransport) -> None:
        self._renderer = renderer
        self._transport = transport

    async def run(self, request: SessionRequest) -> SessionResult:
        observer_token = _CURRENT_USAGE_OBSERVER.set(request.usage_observer)
        try:
            rendered = self._renderer.render(
                request.spec,
                request.workspace,
                session_id=request.session_id,
                trace_dir=request.trace_dir,
            )
            outcome = await asyncio.wait_for(
                self._transport.run(rendered, request.timeout_seconds),
                timeout=request.timeout_seconds,
            )
            if not outcome.usage_streamed:
                for usage in outcome.usage:
                    observe_usage(usage)
            structured_output = (
                outcome.structured_output
                if outcome.structured_output is not None
                else _read_structured_output(rendered, outcome.raw_response)
            )
            if request.spec.output_schema and structured_output is None:
                return _failed_result(outcome, "Provider completed without the required structured output")
            if structured_output is not None:
                validation_error = session_output_validation_error(request.spec.output_schema, structured_output)
                if validation_error is not None:
                    return _failed_result(outcome, validation_error)
            return SessionResult(
                status="completed",
                structured_output=structured_output,
                raw_response=outcome.raw_response,
                tool_calls=outcome.tool_calls,
                usage=outcome.usage,
                cost=_session_cost(outcome.usage),
            )
        except TimeoutError:
            failure_usage = Usage(
                role="optimizer",
                status="timeout",
                error_type="TimeoutError",
                usage_incomplete=True,
            )
            observe_usage(failure_usage)
            return SessionResult(
                status="timeout",
                structured_output=None,
                raw_response="",
                tool_calls=(),
                usage=(failure_usage,),
                cost=Cost(sessions=1),
                error="Provider session timed out",
            )
        except Exception as exc:
            failure_usage = Usage(
                role="optimizer",
                status="failed",
                error_type=type(exc).__name__,
                usage_incomplete=True,
            )
            observe_usage(failure_usage)
            return SessionResult(
                status="failed",
                structured_output=None,
                raw_response="",
                tool_calls=(),
                usage=(failure_usage,),
                cost=Cost(sessions=1),
                error=f"{type(exc).__name__}: {exc}",
            )
        finally:
            _CURRENT_USAGE_OBSERVER.reset(observer_token)


def observe_usage(usage: Usage) -> None:
    observer = _CURRENT_USAGE_OBSERVER.get()
    if observer is None:
        return
    observer(usage)  # type: ignore[operator]


def _read_structured_output(
    session: RenderedSession,
    raw_response: str,
) -> Mapping[str, JsonValue] | None:
    payload: object
    if session.output_path.exists():
        payload = json.loads(session.output_path.read_text(encoding="utf-8"))
    else:
        try:
            payload = json.loads(raw_response)
        except json.JSONDecodeError:
            return None
    if not isinstance(payload, dict):
        raise TypeError("Structured session output must be a JSON object")
    return payload


def _session_cost(usage: tuple[Usage, ...]) -> Cost:
    return Cost(
        sessions=1,
        input_tokens=sum(item.input_tokens for item in usage),
        output_tokens=sum(item.output_tokens for item in usage),
    )


def _failed_result(outcome: TransportOutcome, error: str) -> SessionResult:
    return SessionResult(
        status="failed",
        structured_output=None,
        raw_response=outcome.raw_response,
        tool_calls=outcome.tool_calls,
        usage=outcome.usage,
        cost=_session_cost(outcome.usage),
        error=error,
    )