from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
from dataclasses import dataclass
from typing import Mapping

from autosaddler.v2.core.domain import JsonValue
from autosaddler.v2.prompting.models import ToolCall, Usage
from autosaddler.v2.providers.base import AgentTransport, BaseAgentProvider, TransportOutcome, observe_usage
from autosaddler.v2.providers.workspace_renderer import RenderedSession, codex_renderer


@dataclass(frozen=True, slots=True)
class CodexProviderConfig:
    model: str
    reasoning_effort: str | None = None
    executable: str = "codex"


class CodexAgentProvider(BaseAgentProvider):
    def __init__(
        self,
        config: CodexProviderConfig | None = None,
        *,
        transport: AgentTransport | None = None,
    ) -> None:
        if transport is None:
            if config is None:
                raise ValueError("Codex requires a provider configuration")
            transport = CodexCliTransport(config)
        super().__init__(codex_renderer(), transport)


def codex_runtime(executable: str) -> Mapping[str, JsonValue]:
    """Resolve the CLI version before initializing or resuming a run."""
    result = subprocess.run(
        [executable, "--version"],
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    cli_version = result.stdout.strip()
    if not cli_version.startswith("codex-cli "):
        raise ValueError("Codex executable did not report a codex-cli version")
    return {"executable": executable, "version": cli_version}


class CodexCliTransport:
    def __init__(self, config: CodexProviderConfig) -> None:
        self.config = config

    async def run(self, session: RenderedSession, timeout_seconds: float) -> TransportOutcome:
        trace_root = (session.trace_dir or session.workspace / ".autosaddler") / "codex-session-state"
        trace_root.mkdir(parents=True, exist_ok=True)
        events = _CodexEvents(self.config)
        status = "failed"
        process: asyncio.subprocess.Process | None = None
        try:
            with (
                (trace_root / "events.jsonl").open("wb") as transcript,
                (trace_root / "stderr.log").open("wb") as stderr,
            ):
                async with asyncio.timeout(timeout_seconds):
                    process = await asyncio.create_subprocess_exec(
                        *self._command(session),
                        cwd=session.workspace,
                        stdin=asyncio.subprocess.PIPE,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=stderr,
                        start_new_session=os.name == "posix",
                        limit=4 * 1024 * 1024,
                    )
                    assert process.stdin is not None and process.stdout is not None
                    process.stdin.write(session.task_prompt.encode("utf-8"))
                    await process.stdin.drain()
                    process.stdin.close()
                    while line := await process.stdout.readline():
                        transcript.write(line)
                        transcript.flush()
                        events.consume(json.loads(line))
                    returncode = await process.wait()
                    if returncode != 0:
                        raise RuntimeError(f"Codex exited with status {returncode}; see the session trace")
                    if not events.completed:
                        raise RuntimeError("Codex exited without a completed turn")
                    status = "completed"
                    return TransportOutcome(
                        raw_response=events.response,
                        tool_calls=tuple(events.tool_calls),
                        usage=tuple(events.usage),
                        usage_streamed=True,
                    )
        except TimeoutError:
            status = "timeout"
            raise
        except asyncio.CancelledError:
            status = "interrupted"
            raise
        finally:
            if process is not None:
                await _stop_process(process)
            (trace_root / "export-manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": "autosaddler-codex-session-state/v1",
                        "status": status,
                        "session_id": session.session_id,
                        "codex_thread_id": events.thread_id,
                        "files": ["events.jsonl", "stderr.log"],
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )

    def _command(self, session: RenderedSession) -> list[str]:
        writable = "edit_workspace" in session.allowed_tools
        network = "network" in session.allowed_tools
        instructions = session.instruction_path.read_text(encoding="utf-8")
        if not writable:
            instructions = (
                f"{session.system_context.rstrip()}\n\n## Structured output\n\n"
                "Follow `.autosaddler/session_output_schema.json` and return the final JSON object "
                "as your final response. Do not write workspace files.\n"
            )
        command = [
            self.config.executable,
            "exec",
            "--json",
            "--color",
            "never",
            "--skip-git-repo-check",
            "--ephemeral",
            "--ignore-user-config",
            "--model",
            self.config.model,
            "--sandbox",
            "workspace-write" if writable else "read-only",
            "-c",
            'approval_policy="never"',
            "-c",
            "project_doc_max_bytes=0",
            "-c",
            "developer_instructions=" + json.dumps(instructions, ensure_ascii=False),
            "-c",
            'web_search="live"' if network else 'web_search="disabled"',
            "-c",
            "sandbox_workspace_write.network_access=" + str(network).lower(),
            "-c",
            "features.multi_agent=false",
        ]
        if self.config.reasoning_effort is not None:
            command.extend(["-c", "model_reasoning_effort=" + json.dumps(self.config.reasoning_effort)])
        command.append("-")
        return command


class _CodexEvents:
    def __init__(self, config: CodexProviderConfig) -> None:
        self.config = config
        self.thread_id: str | None = None
        self.completed = False
        self.response = ""
        self.usage: list[Usage] = []
        self.tool_calls: list[ToolCall] = []

    def consume(self, event: object) -> None:
        if not isinstance(event, dict) or not isinstance(event.get("type"), str):
            raise ValueError("Malformed Codex event")
        event_type = event["type"]
        if event_type == "thread.started":
            self.thread_id = event["thread_id"]
        elif event_type in {"turn.failed", "error"}:
            raise RuntimeError("Codex reported a failed turn; see the session trace")
        elif event_type == "turn.completed":
            if self.completed:
                raise ValueError("Codex reported duplicate turn completion")
            usage = _codex_usage(event["usage"], self.config, self.thread_id)
            self.usage.append(usage)
            observe_usage(usage)
            self.completed = True
        elif event_type == "item.completed":
            item = event["item"]
            kind = item["type"]
            if kind == "agent_message":
                self.response = item["text"]
            elif kind == "command_execution":
                self.tool_calls.append(
                    ToolCall(
                        tool=kind,
                        arguments={"command": item["command"]},
                        result_preview=str(item.get("aggregated_output", ""))[:500],
                    )
                )
            elif kind in {"file_change", "mcp_tool_call", "web_search"}:
                self.tool_calls.append(
                    ToolCall(
                        tool=kind,
                        arguments={key: value for key, value in item.items() if key not in {"id", "type", "result"}},
                        result_preview=str(item.get("result", item.get("status", "")))[:500],
                    )
                )


def _codex_usage(value: Mapping[str, JsonValue], config: CodexProviderConfig, thread_id: str | None) -> Usage:
    def tokens(name: str, default: int | None = None) -> int:
        count = value.get(name, default)
        if type(count) is not int or count < 0:
            raise ValueError(f"Invalid Codex usage counter: {name}")
        return count

    inputs = tokens("input_tokens")
    outputs = tokens("output_tokens")
    return Usage(
        model=config.model,
        input_tokens=inputs,
        cached_input_tokens=tokens("cached_input_tokens", 0),
        output_tokens=outputs,
        reasoning_tokens=tokens("reasoning_output_tokens", 0),
        provider_reported_input_tokens=inputs,
        input_token_semantics="includes_cached_tokens",
        total_tokens_is_inferred=True,
        provider_correlation_id=thread_id,
        agent_scope="main",
        configured_settings={"model": config.model, "reasoning_effort": config.reasoning_effort},
        provider_metadata={"cache_write_input_tokens": tokens("cache_write_input_tokens", 0)},
    )


async def _stop_process(process: asyncio.subprocess.Process) -> None:
    """Reap the CLI and stop its tools when a session is cancelled or fails."""
    if os.name == "posix":
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    elif process.returncode is None:
        killer = await asyncio.create_subprocess_exec(
            "taskkill",
            "/PID",
            str(process.pid),
            "/T",
            "/F",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        await killer.wait()
    await process.wait()
