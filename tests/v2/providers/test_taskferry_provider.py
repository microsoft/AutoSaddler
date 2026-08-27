from __future__ import annotations

import asyncio
import json
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

from autosaddler.v2.config.registry import _taskferry_provider
from autosaddler.v2.providers.taskferry import (
    TaskferryAgentProvider,
    TaskferryCliTransport,
    TaskferryProviderConfig,
    _decode_toon,
    _export_taskferry_session_state,
    _taskferry_usage,
)

REAL_OPCODE_TOKENS = {
    "input": 12,
    "cacheRead": 4,
    "cacheWrite": 2,
    "output": 5,
    "totalTokens": 17,
    "cost": {"total": 0.42},
}

REAL_PI_TOKENS = {
    "total": 26,
    "input": 20,
    "output": 6,
    "reasoning": 3,
    "cache": {"write": 9, "read": 15},
}


def test_decode_toon_object_nested_scalars_and_literals() -> None:
    text = (
        "id: oc_test123\n"
        "status: queued\n"
        "model: openai/gpt-5.6\n"
        "exitCode: null\n"
        "incomplete: false\n"
        "startedAt: 2026-08-27T19:00:00Z\n"
        "tokens:\n"
        "  input: 12\n"
        "  cacheRead: 4\n"
    )
    assert _decode_toon(text) == {
        "id": "oc_test123",
        "status": "queued",
        "model": "openai/gpt-5.6",
        "exitCode": None,
        "incomplete": False,
        "startedAt": "2026-08-27T19:00:00Z",
        "tokens": {"input": 12, "cacheRead": 4},
    }


def test_decode_toon_quoted_strings_numbers_and_arrays() -> None:
    text = (
        "message: \"multi\\nline, with 'quote' \\\" and \\\\\"\n"
        "cost: 0.0015\n"
        "negative: -4\n"
        'names[3]: a,b c,"d: e","x\\"y",1,"2"\n'
        'empty: ""\n'
    )
    decoded = _decode_toon(text)
    assert decoded["message"] == 'multi\nline, with \'quote\' " and \\'
    assert decoded["cost"] == 0.0015
    assert decoded["negative"] == -4
    assert decoded["names"] == ["a", "b c", "d: e", 'x"y', 1, "2"]
    assert decoded["empty"] == ""


def test_decode_toon_bare_strings_that_look_like_numbers_stay_strings() -> None:
    decoded = _decode_toon("message: 0.5.6 weird number-ish\n")
    assert decoded == {"message": "0.5.6 weird number-ish"}


def test_decode_toon_rejects_scalar_or_garbage_documents() -> None:
    with pytest.raises(ValueError, match="single TOON object|unsupported line"):
        _decode_toon("hello world\n")
    with pytest.raises(ValueError, match="unexpected indentation"):
        _decode_toon("  status: done\n")


def test_taskferry_usage_opencode_shape_normalizes_tokens_and_cost() -> None:
    usage = _taskferry_usage(
        REAL_OPCODE_TOKENS,
        TaskferryProviderConfig(model="openai/gpt-5.6", variant="default"),
        duration_seconds=3.5,
    )

    assert len(usage) == 1
    item = usage[0]
    assert item.role == "optimizer"
    assert item.model == "openai/gpt-5.6"
    assert item.cached_input_tokens == 4
    assert item.uncached_input_tokens == 14
    assert item.input_tokens == 18
    assert item.output_tokens == 5
    assert item.total_tokens == 23
    assert item.provider_reported_input_tokens == 12
    assert item.provider_reported_total_tokens == 17
    assert item.total_tokens_is_inferred is False
    assert item.input_token_semantics == "excludes_cached_tokens"
    assert item.provider_cost == 0.42
    assert item.provider_cost_unit == "usd"
    assert item.duration_seconds == 3.5
    assert item.configured_settings == {"model": "openai/gpt-5.6", "variant": "default"}
    assert item.status == "success"


def test_taskferry_usage_pi_shape_reads_nested_cache() -> None:
    item = _taskferry_usage(REAL_PI_TOKENS, TaskferryProviderConfig(model="openai/gpt-5.6"))[0]

    assert item.cached_input_tokens == 15
    assert item.uncached_input_tokens == 29
    assert item.input_tokens == 44
    assert item.output_tokens == 6
    assert item.reasoning_tokens == 3
    assert item.total_tokens == 50
    assert item.provider_reported_total_tokens == 26
    assert item.provider_cost is None
    assert item.provider_cost_unit is None


def test_taskferry_usage_without_tokens_or_cost_yields_zero_usage() -> None:
    assert _taskferry_usage(None, TaskferryProviderConfig(model="openai/gpt-5.6")) == ()
    assert _taskferry_usage("unexpected", TaskferryProviderConfig(model="openai/gpt-5.6")) == ()

    item = _taskferry_usage({}, TaskferryProviderConfig(model="openai/gpt-5.6"))[0]
    assert item.input_tokens == 0
    assert item.output_tokens == 0
    assert item.provider_reported_total_tokens is None
    assert item.total_tokens_is_inferred is True
    assert item.provider_cost is None
    assert item.provider_cost_unit is None


def test_taskferry_usage_clamps_reasoning_and_marks_incomplete() -> None:
    item = _taskferry_usage(
        {"input": 10, "output": 2, "reasoning": 9},
        TaskferryProviderConfig(model="openai/gpt-5.6"),
        incomplete=True,
    )[0]

    assert item.reasoning_tokens == 2
    assert item.status == "failed"
    assert item.usage_incomplete is True
    assert item.provider_metadata == {"incomplete": True}


def test_taskferry_config_validation() -> None:
    with pytest.raises(ValueError, match="executor"):
        TaskferryProviderConfig(model="openai/gpt-5.6", executor="fish")
    assert TaskferryProviderConfig(model="openai/gpt-5.6", executor="pi").executor == "pi"
    provider = TaskferryAgentProvider(transport=object())
    assert provider._renderer.provider == "taskferry"


def test_taskferry_provider_settings_validation() -> None:
    with pytest.raises(ValueError, match="missing"):
        _taskferry_provider({})
    with pytest.raises(ValueError, match="extra"):
        _taskferry_provider({"model": "openai/gpt-5.6", "bogus": 1})
    with pytest.raises(TypeError, match="boolean"):
        _taskferry_provider({"model": "openai/gpt-5.6", "sandboxed": "yes"})

    provider = _taskferry_provider(
        {"model": "openai/gpt-5.6", "variant": "ideas", "executor": "pi", "sandboxed": False}
    )
    transport = provider._transport
    assert isinstance(transport, TaskferryCliTransport)
    assert transport.config.variant == "ideas"
    assert transport.config.executor == "pi"
    assert transport.config.sandboxed is False


def test_taskferry_export_manifest_writes_success_and_failure_paths(tmp_path: Path) -> None:
    destination = tmp_path / "export"
    _export_taskferry_session_state(
        task_id="oc_fake123",
        destination_root=destination,
        config=TaskferryProviderConfig(model="openai/gpt-5.6", variant="default"),
        result_payload={
            "status": "done",
            "sessionId": "sdk-session-1",
            "exitCode": 0,
            "incomplete": False,
            "tokens": {"input": 12},
            "message": '{"change":"done"}',
        },
    )
    manifest = json.loads((destination / "export-manifest.json").read_text())
    assert manifest["schema_version"] == "autosaddler-taskferry-trace-export/v1"
    assert manifest["taskferry_task_id"] == "oc_fake123"
    assert manifest["status"] == "done"
    assert manifest["session_id"] == "sdk-session-1"
    assert manifest["exit_code"] == 0
    assert manifest["tokens"] == {"input": 12}
    assert manifest["sensitive"] is True

    failure_destination = tmp_path / "export-failed"
    _export_taskferry_session_state(
        task_id="oc_fake123",
        destination_root=failure_destination,
        config=TaskferryProviderConfig(model="openai/gpt-5.6"),
        error_summary={"error_type": "TimeoutError", "error": "timed out"},
    )
    failed = json.loads((failure_destination / "export-manifest.json").read_text())
    assert failed["status"] == "unsettled"
    assert failed["error_type"] == "TimeoutError"


FAKE_TASKSFERRY_CLI = textwrap.dedent(
    r"""
    #!/usr/bin/env python3
    import json
    import os
    import sys
    from pathlib import Path

    state = Path(os.environ["FAKE_TF_STATE"])
    command = sys.argv[1]
    if command == "dispatch":
        args = sys.argv[2:]
        (state / "args.json").write_text(json.dumps(args))
        prompt_index = args.index("--prompt")
        assert args[prompt_index + 1] == "-"
        (state / "prompt.txt").write_text(sys.stdin.read())
        (state / "directory.txt").write_text(args[args.index("--directory") + 1])
        (state / "model.txt").write_text(args[args.index("--model") + 1])
        if (state / "fail-dispatch").exists():
            print("dispatch exploded", file=sys.stderr)
            sys.exit(2)
        print("id: oc_fake123")
        print("status: queued")
        print("model: openai/gpt-5.6")
    elif command == "wait":
        if (state / "stall").exists():
            print("id: oc_fake123")
            print("status: running")
        else:
            print("id: oc_fake123")
            print("status: done")
    elif command == "result":
        result_status = "done"
        status_file = state / "result-status.txt"
        if status_file.exists():
            result_status = status_file.read_text().strip()
        print("status: " + result_status)
        print("sessionId: sdk-session-1")
        print("message: " + json.dumps('{"change":"done"}'))
        if result_status == "done":
            print("tokens:")
            print("  input: 12")
            print("  cacheRead: 4")
            print("  cacheWrite: 2")
            print("  output: 5")
            print("  totalTokens: 17")
            print("  cost:")
            print("    total: 0.42")
        else:
            print("failureReason: exploded")
            print("exitCode: 1")
    elif command == "cancel":
        (state / "cancelled.txt").write_text(" ".join(sys.argv[2:]))
        print("cancelled")
    """
)


@pytest.fixture
def fake_taskferry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Path]:
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    script = tmp_path / "fake-taskferry"
    script.write_text(FAKE_TASKSFERRY_CLI.lstrip("\n"))
    script.chmod(0o755)
    monkeypatch.setenv("FAKE_TF_STATE", str(state_dir))
    return script, state_dir


def session_state(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        workspace=tmp_path / "workspace",
        system_context="Optimize only the declared candidate surface.",
        task_prompt="Diagnose the training failures and emit one mutation.",
        trace_dir=tmp_path / "traces",
    )


def test_taskferry_transport_dispatches_and_parses_result(
    tmp_path: Path, fake_taskferry: tuple[Path, Path]
) -> None:
    script, state_dir = fake_taskferry
    config = TaskferryProviderConfig(
        model="openai/gpt-5.6",
        variant="ideas",
        executor="opencode",
        executable=str(script),
    )
    session = session_state(tmp_path)
    session.workspace.mkdir(parents=True)

    outcome = asyncio.run(TaskferryCliTransport(config).run(session, 30))

    assert outcome.raw_response == '{"change":"done"}'
    assert outcome.usage_streamed is True
    usage = outcome.usage[0]
    assert usage.input_tokens == 18
    assert usage.cached_input_tokens == 4
    assert usage.uncached_input_tokens == 14
    assert usage.output_tokens == 5
    assert usage.total_tokens == 23
    assert usage.provider_cost == 0.42
    assert usage.provider_cost_unit == "usd"
    assert usage.provider_correlation_id == "sdk-session-1"

    dispatch_args = json.loads((state_dir / "args.json").read_text())
    assert "--no-overlay" in dispatch_args
    assert dispatch_args[dispatch_args.index("--class") + 1] == "autosaddler-v2"
    assert dispatch_args[dispatch_args.index("--variant") + 1] == "ideas"
    assert dispatch_args[dispatch_args.index("--executor") + 1] == "opencode"
    assert (state_dir / "directory.txt").read_text() == str(session.workspace)
    prompt = (state_dir / "prompt.txt").read_text()
    assert session.system_context in prompt
    assert session.task_prompt in prompt
    assert "session_output_schema.json" in prompt
    assert not (state_dir / "cancelled.txt").exists()

    manifest = json.loads(
        (session.trace_dir / "taskferry-session-state" / "export-manifest.json").read_text()
    )
    assert manifest["taskferry_task_id"] == "oc_fake123"
    assert manifest["status"] == "done"
    assert manifest["tokens"]["input"] == 12
    assert "Diagnose" not in json.dumps(manifest)


def test_taskferry_transport_without_sandbox_passes_no_sandbox_flag(
    tmp_path: Path, fake_taskferry: tuple[Path, Path]
) -> None:
    script, state_dir = fake_taskferry
    config = TaskferryProviderConfig(
        model="openai/gpt-5.6",
        sandboxed=False,
        executable=str(script),
    )
    session = session_state(tmp_path)
    session.workspace.mkdir(parents=True)

    asyncio.run(TaskferryCliTransport(config).run(session, 30))

    assert "--no-sandbox" in json.loads((state_dir / "args.json").read_text())
    assert "--no-overlay" in json.loads((state_dir / "args.json").read_text())


def test_taskferry_transport_fails_closed_on_crashed_task_and_cancels(
    tmp_path: Path, fake_taskferry: tuple[Path, Path]
) -> None:
    script, state_dir = fake_taskferry
    (state_dir / "result-status.txt").write_text("crashed")
    config = TaskferryProviderConfig(model="openai/gpt-5.6", executable=str(script))
    session = session_state(tmp_path)
    session.workspace.mkdir(parents=True)

    with pytest.raises(RuntimeError, match="crashed"):
        asyncio.run(TaskferryCliTransport(config).run(session, 30))

    assert (state_dir / "cancelled.txt").read_text() == "oc_fake123"
    manifest = json.loads(
        (session.trace_dir / "taskferry-session-state" / "export-manifest.json").read_text()
    )
    assert manifest["status"] == "crashed"
    assert manifest["failureReason"] == "exploded"


def test_taskferry_transport_times_out_when_wait_stalls_and_cancels(
    tmp_path: Path, fake_taskferry: tuple[Path, Path]
) -> None:
    script, state_dir = fake_taskferry
    (state_dir / "stall").write_text("")
    config = TaskferryProviderConfig(model="openai/gpt-5.6", executable=str(script))
    session = session_state(tmp_path)
    session.workspace.mkdir(parents=True)

    with pytest.raises(TimeoutError, match="did not settle"):
        asyncio.run(TaskferryCliTransport(config).run(session, 30))

    assert (state_dir / "cancelled.txt").read_text() == "oc_fake123"
    manifest = json.loads(
        (session.trace_dir / "taskferry-session-state" / "export-manifest.json").read_text()
    )
    assert manifest["status"] == "unsettled"
    assert manifest["error_type"] == "TimeoutError"


def test_taskferry_transport_raises_on_dispatch_failure(
    tmp_path: Path, fake_taskferry: tuple[Path, Path]
) -> None:
    script, state_dir = fake_taskferry
    (state_dir / "fail-dispatch").write_text("")
    config = TaskferryProviderConfig(model="openai/gpt-5.6", executable=str(script))
    session = session_state(tmp_path)
    session.workspace.mkdir(parents=True)

    with pytest.raises(RuntimeError, match="exit 2"):
        asyncio.run(TaskferryCliTransport(config).run(session, 30))

    assert not (state_dir / "cancelled.txt").exists()