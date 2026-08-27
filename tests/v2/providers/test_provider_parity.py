from __future__ import annotations

import asyncio
import json
import sys
from dataclasses import replace
from datetime import timedelta
from enum import Enum
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace

import pytest
import yaml

from autosaddler.v2.prompting.models import SessionRequest, SessionSpec, ToolCall, Usage
from autosaddler.v2.providers.base import TransportOutcome
from autosaddler.v2.providers.claude import (
    ClaudeAgentProvider,
    ClaudeProviderConfig,
    ClaudeSdkTransport,
    _claude_model_usages,
    _export_claude_session_state,
    _usage as claude_usage,
)
from autosaddler.v2.providers.copilot import (
    CopilotAgentProvider,
    CopilotCustomProviderConfig,
    CopilotProviderConfig,
    CopilotSdkTransport,
    _copilot_provider_mapping,
    _copilot_usage,
    _copilot_session_metrics,
    _export_copilot_session_state,
)
from autosaddler.v2.providers.taskferry import TaskferryAgentProvider


class CapturingTransport:
    def __init__(self) -> None:
        self.sessions = []

    async def run(self, session, timeout_seconds):
        self.sessions.append((session, timeout_seconds))
        return TransportOutcome(
            raw_response='{"change": "tighten instructions"}',
            structured_output={"change": "tighten instructions"},
            tool_calls=(ToolCall(tool="edit", arguments={"path": "candidate.json"}),),
            usage=(Usage(input_tokens=10, output_tokens=4, model="fake"),),
        )


def test_usage_preserves_provider_native_metrics() -> None:
    usage = Usage(
        role="optimizer",
        model="fake",
        input_tokens=10,
        cached_input_tokens=3,
        uncached_input_tokens=7,
        output_tokens=4,
        reasoning_tokens=1,
        total_tokens=14,
        provider_cost=0.25,
        provider_cost_unit="premium_request_multiplier",
        provider_nano_aiu=12.5,
        duration_seconds=1.5,
        status="failed",
        error_type="TimeoutError",
        usage_incomplete=True,
        provider_correlation_id="request-1",
    )

    assert usage.total_tokens == 14
    assert usage.provider_cost_unit == "premium_request_multiplier"
    assert usage.provider_ai_credits == 12.5e-9
    assert usage.status == "failed"
    assert usage.usage_incomplete is True


def test_copilot_usage_normalizes_native_cost_cache_and_correlation() -> None:
    class ApiEndpoint(str, Enum):
        RESPONSES = "responses"

    class InteractionType(str, Enum):
        SUBAGENT = "conversation-subagent"

    usage = _copilot_usage(
        SimpleNamespace(
            input_tokens=10,
            cache_read_tokens=3,
            cache_write_tokens=2,
            output_tokens=4,
            reasoning_tokens=1,
            cost=2.0,
            model="gpt-test",
            duration=timedelta(seconds=1.25),
            copilot_usage=SimpleNamespace(total_nano_aiu=42.0),
            api_call_id="api-call",
            reasoning_effort="high",
            api_endpoint=ApiEndpoint.RESPONSES,
            finish_reason="stop",
            time_to_first_token=timedelta(milliseconds=250),
        ),
        CopilotProviderConfig(model="fallback", reasoning_effort="medium"),
        agent_id="agent-1",
        interaction_type=InteractionType.SUBAGENT,
    )

    assert usage.total_tokens == 14
    assert usage.cached_input_tokens == 3
    assert usage.uncached_input_tokens == 7
    assert usage.provider_cost == 2.0
    assert usage.provider_cost_unit == "premium_request_multiplier"
    assert usage.provider_nano_aiu == 42.0
    assert usage.provider_ai_credits == 42e-9
    assert usage.agent_id == "agent-1"
    assert usage.agent_scope == "subagent"
    assert usage.duration_seconds == 1.25
    assert usage.provider_correlation_id == "api-call"
    assert usage.configured_settings == {"model": "fallback", "reasoning_effort": "medium"}
    assert usage.reported_settings == {"reasoning_effort": "high"}
    assert usage.provider_metadata == {
        "api_endpoint": "responses",
        "cache_creation_input_tokens": 2,
        "finish_reason": "stop",
        "interaction_type": "conversation-subagent",
        "time_to_first_token": 0.25,
    }
    assert usage.provider_reported_input_tokens == 10


def test_claude_usage_normalizes_cache_and_currency_cost() -> None:
    usage = claude_usage(
        {
            "input_tokens": 10,
            "cache_read_input_tokens": 3,
            "cache_creation_input_tokens": 2,
            "output_tokens": 4,
        },
        "claude-test",
        provider_cost=0.75,
        duration_seconds=2.5,
    )

    assert usage.input_tokens == 15
    assert usage.total_tokens == 19
    assert usage.provider_reported_input_tokens == 10
    assert usage.input_token_semantics == "excludes_cached_tokens"
    assert usage.cached_input_tokens == 3
    assert usage.uncached_input_tokens == 12
    assert usage.provider_metadata == {"cache_creation_input_tokens": 2}
    assert usage.provider_cost == 0.75
    assert usage.provider_cost_unit == "usd"
    assert usage.duration_seconds == 2.5


def test_claude_model_usage_preserves_subagent_inclusive_usd_by_model() -> None:
    message = SimpleNamespace(
        session_id="claude-session",
        duration_ms=4_000,
        duration_api_ms=3_000,
        num_turns=2,
        stop_reason="end_turn",
        subtype="success",
        api_error_status=None,
        total_cost_usd=1.25,
    )

    usage = _claude_model_usages(
        {
            "claude-a": {
                "inputTokens": 10,
                "outputTokens": 4,
                "cacheReadInputTokens": 3,
                "cacheCreationInputTokens": 2,
                "costUSD": 1.25,
                "contextWindow": 200_000,
            }
        },
        configured_model="claude-a",
        configured_effort="max",
        message=message,
    )

    assert len(usage) == 1
    assert usage[0].input_tokens == 15
    assert usage[0].cached_input_tokens == 3
    assert usage[0].uncached_input_tokens == 12
    assert usage[0].provider_cost == 1.25
    assert usage[0].provider_cost_unit == "usd"
    assert usage[0].agent_scope == "session_inclusive"
    assert usage[0].provider_metadata["includes_subagents"] is True
    assert usage[0].provider_metadata["session_total_cost_usd"] == 1.25

    unknown_cost = _claude_model_usages(
        {"claude-a": {"inputTokens": 1, "outputTokens": 1}},
        configured_model="claude-a",
        configured_effort="max",
        message=message,
    )[0]
    assert unknown_cost.provider_cost is None
    assert unknown_cost.provider_cost_unit is None


def test_claude_trace_export_copies_exact_main_and_subagent_transcripts(tmp_path: Path) -> None:
    project_dir = tmp_path / "projects" / "workspace"
    session_id = "11111111-1111-4111-8111-111111111111"
    project_dir.mkdir(parents=True)
    (project_dir / f"{session_id}.jsonl").write_text('{"type":"main"}\n')
    subagents = project_dir / session_id / "subagents"
    subagents.mkdir(parents=True)
    (subagents / "agent-a.jsonl").write_text('{"type":"subagent"}\n{"type":"subagent"}\n')
    (subagents / "agent-a.meta.json").write_text('{"agentType":"research"}\n')
    (project_dir / "22222222-2222-4222-8222-222222222222.jsonl").write_text("unrelated\n")

    destination = tmp_path / "export"
    _export_claude_session_state(
        claude_session_id=session_id,
        workspace=tmp_path,
        destination_root=destination,
        source_project_dir=project_dir,
        session_summary={"session_total_cost_usd": 1.25},
    )

    manifest = json.loads((destination / "export-manifest.json").read_text())
    assert manifest["status"] == "exported"
    assert manifest["claude_session_id"] == session_id
    assert manifest["session_usage"] == {"session_total_cost_usd": 1.25}
    assert manifest["summary"] == {
        "main_message_count": 1,
        "subagent_message_count": 2,
        "subagent_transcript_count": 1,
    }
    assert (destination / session_id / f"{session_id}.jsonl").is_file()
    assert (destination / session_id / "subagents/agent-a.jsonl").is_file()
    assert not (destination / session_id / "22222222-2222-4222-8222-222222222222.jsonl").exists()


def test_copilot_trace_export_copies_exact_session_and_omits_lock(tmp_path: Path) -> None:
    source_root = tmp_path / "session-state"
    wanted = source_root / "sdk-session-a"
    unrelated = source_root / "sdk-session-b"
    wanted.mkdir(parents=True)
    unrelated.mkdir()
    (wanted / "events.jsonl").write_text(
        '{"type":"session.start","timestamp":"2026-08-07T09:00:00Z",'
        '"data":{"selectedModel":"gpt-test","reasoningEffort":"high","prompt":"secret"}}\n'
        '{"type":"tool.execution_start","timestamp":"2026-08-07T09:00:02Z"}\n'
        '{"type":"session.shutdown","timestamp":"2026-08-07T09:00:05Z"}\n'
    )
    (wanted / "inuse.123.lock").write_text("")
    (unrelated / "events.jsonl").write_text("unrelated\n")

    destination = tmp_path / "export"
    _export_copilot_session_state(
        sdk_session_id="sdk-session-a",
        destination_root=destination,
        source_root=source_root,
    )

    manifest = json.loads((destination / "export-manifest.json").read_text())
    assert manifest["status"] == "exported"
    assert manifest["copilot_session_id"] == "sdk-session-a"
    assert manifest["summary"]["event_count"] == 3
    assert manifest["summary"]["tool_call_count"] == 1
    assert manifest["summary"]["duration_seconds"] == 5.0
    assert manifest["summary"]["session_settings"] == {
        "selectedModel": "gpt-test",
        "reasoningEffort": "high",
    }
    assert "secret" not in json.dumps(manifest)
    assert (destination / "sdk-session-a/events.jsonl").is_file()
    assert not (destination / "sdk-session-a/inuse.123.lock").exists()
    assert not (destination / "sdk-session-b").exists()


def test_copilot_session_metrics_reconcile_main_and_subagent_costs() -> None:
    aggregate = _copilot_session_metrics(
        SimpleNamespace(
            current_model="gpt-test",
            last_call_input_tokens=8,
            last_call_output_tokens=2,
            total_api_duration_ms=2_500,
            total_premium_request_cost=3.0,
            total_user_requests=2,
            total_nano_aiu=1_500_000_000.0,
            model_metrics={
                "gpt-test": SimpleNamespace(
                    requests=SimpleNamespace(count=2, cost=3.0),
                    usage=SimpleNamespace(
                        input_tokens=20,
                        cache_read_tokens=6,
                        cache_write_tokens=4,
                        output_tokens=5,
                        reasoning_tokens=1,
                    ),
                    total_nano_aiu=1_500_000_000.0,
                )
            },
        )
    )

    assert aggregate["scope"] == "session_inclusive"
    assert aggregate["includes_subagents"] is True
    assert aggregate["total_ai_credits"] == 1.5
    assert aggregate["total_premium_request_cost"] == 3.0
    assert aggregate["models"]["gpt-test"] == {
        "request_count": 2,
        "premium_request_cost": 3.0,
        "input_tokens": 20,
        "cached_input_tokens": 6,
        "uncached_input_tokens": 14,
        "output_tokens": 5,
        "reasoning_tokens": 1,
        "total_tokens": 25,
        "total_nano_aiu": 1_500_000_000.0,
        "ai_credits": 1.5,
        "provider_metadata": {"cache_creation_input_tokens": 4},
    }


def test_copilot_reconciliation_failure_does_not_fail_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class FakeSessionEventType:
        TOOL_EXECUTION_START = "tool.execution_start"
        TOOL_EXECUTION_COMPLETE = "tool.execution_complete"
        ASSISTANT_USAGE = "assistant.usage"
        MODEL_CALL_FAILURE = "model.call_failure"

    class FakeSession:
        session_id = "sdk-session"

        def __init__(self) -> None:
            self.rpc = SimpleNamespace(
                usage=SimpleNamespace(get_metrics=self.get_metrics),
            )

        async def get_metrics(self):
            raise RuntimeError("metrics unavailable")

        def on(self, _handler) -> None:
            pass

        async def send_and_wait(self, _prompt, timeout):
            assert timeout == 5
            return SimpleNamespace(data=SimpleNamespace(content='{"change":"done"}'))

    class FakeClient:
        def __init__(self, **_kwargs) -> None:
            pass

        async def start(self) -> None:
            pass

        async def create_session(self, **_kwargs):
            return FakeSession()

        async def stop(self) -> None:
            raise RuntimeError("client cleanup failed")

    copilot_module = ModuleType("copilot")
    copilot_module.CopilotClient = FakeClient
    copilot_module.PermissionHandler = SimpleNamespace(approve_all=object())
    events_module = ModuleType("copilot.generated.session_events")
    events_module.SessionEventType = FakeSessionEventType
    monkeypatch.setitem(sys.modules, "copilot", copilot_module)
    monkeypatch.setitem(sys.modules, "copilot.generated.session_events", events_module)
    exported: dict[str, object] = {}
    monkeypatch.setattr(
        "autosaddler.v2.providers.copilot._export_copilot_session_state",
        lambda **values: exported.update(values),
    )
    session = SimpleNamespace(
        workspace=tmp_path,
        allowed_tools=(),
        system_context="system",
        skill_directory=tmp_path,
        session_id="session-1",
        task_prompt="prompt",
        trace_dir=tmp_path / "traces",
    )

    outcome = asyncio.run(CopilotSdkTransport(CopilotProviderConfig()).run(session, 5))

    assert outcome.raw_response == '{"change":"done"}'
    assert exported["aggregate_metrics"] is None
    assert exported["aggregate_metrics_error"] == {
        "error_type": "RuntimeError",
        "error": "metrics unavailable",
    }


def test_claude_export_failure_does_not_fail_response(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    captured_options = None

    class FakeAssistantMessage:
        pass

    class FakeResultMessage:
        session_id = "claude-session"
        result = '{"change":"done"}'
        model_usage = {}
        usage = {}
        duration_ms = 1_000
        duration_api_ms = 750
        num_turns = 1
        total_cost_usd = None

    class FakeOptions:
        def __init__(self, **kwargs) -> None:
            nonlocal captured_options
            captured_options = kwargs

    async def fake_query(**_kwargs):
        yield FakeResultMessage()

    claude_module = ModuleType("claude_agent_sdk")
    claude_module.AssistantMessage = FakeAssistantMessage
    claude_module.ClaudeAgentOptions = FakeOptions
    claude_module.ResultMessage = FakeResultMessage
    claude_module.query = fake_query
    types_module = ModuleType("claude_agent_sdk.types")
    types_module.TextBlock = type("TextBlock", (), {})
    types_module.ToolUseBlock = type("ToolUseBlock", (), {})
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", claude_module)
    monkeypatch.setitem(sys.modules, "claude_agent_sdk.types", types_module)
    monkeypatch.setattr(
        "autosaddler.v2.providers.claude._export_claude_session_state",
        lambda **_values: (_ for _ in ()).throw(OSError("read-only trace root")),
    )
    session = SimpleNamespace(
        workspace=tmp_path,
        allowed_tools=(),
        system_context="system",
        task_prompt="prompt",
        trace_dir=tmp_path / "traces",
    )

    outcome = asyncio.run(ClaudeSdkTransport(ClaudeProviderConfig()).run(session, 5))

    assert outcome.raw_response == '{"change":"done"}'
    assert captured_options is not None
    assert captured_options["env"] == {"ANTHROPIC_BASE_URL": "https://api.anthropic.com"}


def session_spec() -> SessionSpec:
    return SessionSpec(
        kind="diagnose_patch",
        system_context="Optimize only the declared candidate surface.",
        task_prompt="Diagnose the training failures and emit one mutation.",
        skills={
            "diagnose": (
                "---\n"
                "name: diagnose\n"
                'description: "Diagnose training failures from staged evidence."\n'
                "---\n\n"
                "# Diagnose\n\n"
                "Use training evidence only.\n"
            )
        },
        output_schema={
            "type": "object",
            "required": ["change"],
            "properties": {"change": {"type": "string"}},
        },
        workspace_files={"evidence/train.json": "{}\n"},
        capabilities=frozenset({"read_workspace", "edit_workspace", "load_skills"}),
    )


@pytest.mark.parametrize(
    ("provider_type", "instruction_file", "skill_root"),
    [
        (ClaudeAgentProvider, "CLAUDE.md", ".claude/skills"),
        (CopilotAgentProvider, "AGENTS.md", ".copilot/skills"),
        (TaskferryAgentProvider, "AGENTS.md", ".opencode/skills"),
    ],
)
def test_provider_parity_preserves_semantic_session_assets(
    tmp_path: Path,
    provider_type,
    instruction_file: str,
    skill_root: str,
) -> None:
    transport = CapturingTransport()
    workspace = tmp_path / provider_type.__name__
    request = SessionRequest(
        session_id="session-1",
        operation_id="iteration-1:diagnose",
        spec=session_spec(),
        workspace=workspace,
        timeout_seconds=15,
    )
    stale_skill = workspace / skill_root / "stale" / "SKILL.md"
    stale_skill.parent.mkdir(parents=True)
    stale_skill.write_text("unvalidated stale skill\n")

    result = asyncio.run(provider_type(transport=transport).run(request))
    rendered, timeout = transport.sessions[0]

    assert result.status == "completed"
    assert result.structured_output == {"change": "tighten instructions"}
    assert result.cost.sessions == 1
    assert result.cost.input_tokens == 10
    assert result.tool_calls[0].arguments == {"path": "candidate.json"}
    assert timeout == 15
    assert rendered.task_prompt == request.spec.task_prompt
    assert rendered.system_context == request.spec.system_context
    assert request.spec.system_context in (workspace / instruction_file).read_text()
    rendered_skill = (workspace / skill_root / "diagnose" / "SKILL.md").read_text()
    assert rendered_skill == request.spec.skills["diagnose"]
    _, frontmatter, _ = rendered_skill.split("---\n", 2)
    assert yaml.safe_load(frontmatter) == {
        "name": "diagnose",
        "description": "Diagnose training failures from staged evidence.",
    }
    assert not stale_skill.exists()
    assert json.loads((workspace / ".autosaddler/session_output_schema.json").read_text()) == dict(
        request.spec.output_schema
    )
    assert (workspace / "evidence/train.json").read_text() == "{}\n"
    assert rendered.allowed_tools


@pytest.mark.parametrize("provider_type", [ClaudeAgentProvider, CopilotAgentProvider, TaskferryAgentProvider])
def test_provider_rejects_skill_without_discovery_frontmatter(tmp_path: Path, provider_type) -> None:
    spec = replace(session_spec(), skills={"diagnose": "# Diagnose\n"})
    transport = CapturingTransport()
    request = SessionRequest(
        session_id="session-invalid-skill",
        operation_id="iteration-1:diagnose",
        spec=spec,
        workspace=tmp_path / provider_type.__name__,
        timeout_seconds=15,
    )

    result = asyncio.run(provider_type(transport=transport).run(request))

    assert result.status == "failed"
    assert result.error is not None
    assert "must start with YAML frontmatter" in result.error
    assert transport.sessions == []


@pytest.mark.parametrize(
    ("provider_type", "skill_root"),
    [
        (ClaudeAgentProvider, ".claude/skills"),
        (CopilotAgentProvider, ".copilot/skills"),
        (TaskferryAgentProvider, ".opencode/skills"),
    ],
)
def test_provider_rejects_workspace_file_in_skill_directory(tmp_path: Path, provider_type, skill_root: str) -> None:
    spec = replace(
        session_spec(),
        workspace_files={f"{skill_root}/injected/SKILL.md": "unvalidated injected skill\n"},
    )
    transport = CapturingTransport()
    request = SessionRequest(
        session_id="session-injected-skill",
        operation_id="iteration-1:diagnose",
        spec=spec,
        workspace=tmp_path / provider_type.__name__,
        timeout_seconds=15,
    )

    result = asyncio.run(provider_type(transport=transport).run(request))

    assert result.status == "failed"
    assert result.error is not None
    assert "collides with provider skill directory" in result.error
    assert transport.sessions == []


def test_provider_fails_closed_when_required_output_is_missing(tmp_path: Path) -> None:
    class MissingOutputTransport:
        async def run(self, _session, _timeout_seconds):
            return TransportOutcome(raw_response="No JSON was produced")

    request = SessionRequest(
        session_id="session-missing",
        operation_id="iteration-1:reflect",
        spec=session_spec(),
        workspace=tmp_path,
        timeout_seconds=15,
    )
    result = asyncio.run(ClaudeAgentProvider(transport=MissingOutputTransport()).run(request))

    assert result.status == "failed"
    assert result.error == "Provider completed without the required structured output"


def test_provider_observes_zero_token_failure(tmp_path: Path) -> None:
    class FailingTransport:
        async def run(self, _session, _timeout_seconds):
            raise ConnectionError("provider unavailable")

    observed: list[Usage] = []
    request = SessionRequest(
        session_id="session-failed",
        operation_id="iteration-1:diagnose:attempt:1",
        spec=session_spec(),
        workspace=tmp_path,
        timeout_seconds=15,
        usage_observer=observed.append,
    )

    result = asyncio.run(ClaudeAgentProvider(transport=FailingTransport()).run(request))

    assert result.status == "failed"
    assert len(observed) == 1
    assert observed[0].total_tokens == 0
    assert observed[0].status == "failed"
    assert observed[0].error_type == "ConnectionError"
    assert observed[0].usage_incomplete is True


def test_provider_does_not_reuse_structured_output_from_prior_attempt(tmp_path: Path) -> None:
    class MissingOutputTransport:
        async def run(self, session, _timeout_seconds):
            assert not session.output_path.exists()
            return TransportOutcome(raw_response="No JSON was produced")

    output = tmp_path / ".autosaddler/session_output.json"
    output.parent.mkdir(parents=True)
    output.write_text('{"change":"stale"}\n')
    request = SessionRequest(
        session_id="session-retry",
        operation_id="iteration-1:diagnose:attempt:2",
        spec=session_spec(),
        workspace=tmp_path,
        timeout_seconds=15,
    )

    result = asyncio.run(ClaudeAgentProvider(transport=MissingOutputTransport()).run(request))

    assert result.status == "failed"
    assert result.error == "Provider completed without the required structured output"


def test_provider_rejects_structured_output_that_violates_schema(tmp_path: Path) -> None:
    class InvalidOutputTransport:
        async def run(self, _session, _timeout_seconds):
            return TransportOutcome(raw_response='{"change":"ignored fallback"}', structured_output={})

    request = SessionRequest(
        session_id="session-invalid",
        operation_id="iteration-1:diagnose:attempt:1",
        spec=session_spec(),
        workspace=tmp_path,
        timeout_seconds=15,
    )

    result = asyncio.run(ClaudeAgentProvider(transport=InvalidOutputTransport()).run(request))

    assert result.status == "failed"
    assert "'change' is a required property" in result.error


def test_copilot_custom_provider_is_complete_and_serializable() -> None:
    provider = CopilotCustomProviderConfig(
        type="openai",
        base_url="http://127.0.0.1:4144",
        wire_api="responses",
        model_id="gpt-5.5",
        wire_model="gpt-5.5",
    )

    assert provider.as_mapping() == {
        "type": "openai",
        "base_url": "http://127.0.0.1:4144",
        "wire_api": "responses",
        "model_id": "gpt-5.5",
        "wire_model": "gpt-5.5",
    }


def test_copilot_official_openai_provider_uses_environment_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = CopilotCustomProviderConfig(
        type="openai",
        base_url="https://api.openai.com/v1",
        wire_api="responses",
        model_id="gpt-5.5",
        wire_model="gpt-5.5",
    )
    monkeypatch.setenv("OPENAI_API_KEY", "test-openai-key")

    assert _copilot_provider_mapping(provider) == {
        **provider.as_mapping(),
        "api_key": "test-openai-key",
    }

    monkeypatch.delenv("OPENAI_API_KEY")
    with pytest.raises(EnvironmentError, match="OPENAI_API_KEY"):
        _copilot_provider_mapping(provider)