from __future__ import annotations

import asyncio
import json
import os
import tomllib
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from autosaddler.v2.config.registry import build_runtime
from autosaddler.v2.prompting.models import SessionRequest, SessionSpec
from autosaddler.v2.providers.codex import CodexAgentProvider, CodexProviderConfig, _codex_usage


@pytest.fixture
def codex_cli(tmp_path: Path) -> Path:
    path = tmp_path / "codex"
    path.write_text("""#!/usr/bin/env python3
import json
import os
from pathlib import Path
import subprocess
import sys
import time

if sys.argv[1:] == ["--version"]:
    print("codex-cli 0.153.0")
    sys.exit(0)
root = Path.cwd()
prompt = sys.stdin.read()
with (Path(__file__).parent / "calls.jsonl").open("a") as calls:
    calls.write(json.dumps(sys.argv[1:]) + "\\n")
(root / "invocation.json").write_text(json.dumps({"args": sys.argv[1:], "prompt": prompt}))
mode = (root / "mode").read_text() if (root / "mode").exists() else "success"
def emit(value):
    print(json.dumps(value), flush=True)
emit({"type": "thread.started", "thread_id": "codex-thread"})
emit({"type": "turn.started"})
if mode in ("timeout", "cancel"):
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    (root / "child.pid").write_text(str(child.pid))
    (root / "parent.pid").write_text(str(os.getpid()))
    time.sleep(60)
if mode == "malformed":
    print("not JSON", flush=True)
    time.sleep(60)
if mode == "turn-failed":
    emit({"type": "turn.failed", "error": {"message": "provider rejected request"}})
    sys.exit(0)
if mode == "truncated":
    sys.exit(0)
emit({"type": "item.completed", "item": {
    "id": "command", "type": "command_execution", "command": "cat candidate.json",
    "aggregated_output": "baseline", "exit_code": 0, "status": "completed"}})
emit({"type": "item.completed", "item": {
    "id": "patch", "type": "file_change", "changes": [{"path": "candidate.json", "kind": "update"}],
    "status": "completed"}})
response = {"change": "improved"} if mode != "schema" else {"wrong": True}
if (root / ".autosaddler/fake_response.json").exists():
    response = json.loads((root / ".autosaddler/fake_response.json").read_text())
if mode != "missing":
    (root / ".autosaddler/session_output.json").write_text(json.dumps(response))
emit({"type": "item.completed", "item": {"id": "message", "type": "agent_message", "text": "Done"}})
emit({"type": "turn.completed", "usage": {
    "input_tokens": 10, "cached_input_tokens": 4, "output_tokens": 3, "reasoning_output_tokens": 1}})
if mode == "exit":
    print("provider failure", file=sys.stderr)
    sys.exit(2)
""")
    path.chmod(0o755)
    return path


def request(tmp_path: Path, *, timeout: float = 10) -> SessionRequest:
    return SessionRequest(
        session_id="session-1",
        operation_id="operation-1",
        spec=SessionSpec(
            kind="diagnose_patch",
            system_context='Only use the staged training evidence. Preserve Unicode 🤖 and "quotes".',
            task_prompt="Improve the candidate; quotes and $(shell text) are literal.",
            skills={},
            workspace_files={},
            output_schema={
                "type": "object",
                "required": ["change"],
                "properties": {"change": {"type": "string"}},
                "additionalProperties": False,
            },
            capabilities=frozenset({"read_workspace", "edit_workspace", "run_commands"}),
        ),
        workspace=tmp_path / "workspace",
        trace_dir=tmp_path / "trace",
        timeout_seconds=timeout,
    )


def provider(codex_cli: Path) -> CodexAgentProvider:
    return CodexAgentProvider(
        CodexProviderConfig(model="test-model", reasoning_effort="low", executable=str(codex_cli))
    )


def test_cli_transport_returns_output_usage_tools_and_trace(tmp_path: Path, codex_cli: Path) -> None:
    observed = []
    req = replace(request(tmp_path), usage_observer=observed.append)
    result = asyncio.run(provider(codex_cli).run(req))

    assert result.status == "completed"
    assert result.structured_output == {"change": "improved"}
    assert result.cost.input_tokens == 10
    assert result.cost.output_tokens == 3
    assert len(observed) == 1
    assert observed[0] == result.usage[0]
    assert result.usage[0].cached_input_tokens == 4
    assert result.usage[0].uncached_input_tokens == 6
    assert result.usage[0].reasoning_tokens == 1
    assert result.usage[0].provider_cost is None
    assert result.usage[0].provider_correlation_id == "codex-thread"
    assert [tool.tool for tool in result.tool_calls] == ["command_execution", "file_change"]
    invocation = json.loads((req.workspace / "invocation.json").read_text())
    assert invocation["prompt"] == req.spec.task_prompt
    args = invocation["args"]
    assert args[-1] == "-"
    assert args[args.index("--sandbox") + 1] == "workspace-write"
    assert 'web_search="disabled"' in args
    assert "sandbox_workspace_write.network_access=false" in args
    assert "--ignore-user-config" in args
    assert "project_doc_max_bytes=0" in args
    instructions = next(arg for arg in args if arg.startswith("developer_instructions="))
    assert req.spec.system_context in tomllib.loads(instructions)["developer_instructions"]
    assert not any("dangerously" in arg for arg in args)
    trace = req.trace_dir / "codex-session-state"
    manifest = json.loads((trace / "export-manifest.json").read_text())
    assert manifest["status"] == "completed"
    assert manifest["codex_thread_id"] == "codex-thread"
    assert len((trace / "events.jsonl").read_text().splitlines()) == 6


@pytest.mark.parametrize("mode", ["turn-failed", "truncated", "malformed", "exit", "missing", "schema"])
def test_failures_cannot_become_success(tmp_path: Path, codex_cli: Path, mode: str) -> None:
    req = request(tmp_path)
    req.workspace.mkdir()
    (req.workspace / "mode").write_text(mode)
    result = asyncio.run(provider(codex_cli).run(req))
    assert result.status == "failed"
    assert result.error
    assert (req.trace_dir / "codex-session-state/events.jsonl").exists()
    if mode == "exit":
        assert "status 2" in result.error


def test_read_only_and_network_capabilities(tmp_path: Path, codex_cli: Path) -> None:
    req = request(tmp_path)
    req = replace(req, spec=replace(req.spec, capabilities=frozenset({"read_workspace", "network"})))
    result = asyncio.run(provider(codex_cli).run(req))
    assert result.status == "completed"
    args = json.loads((req.workspace / "invocation.json").read_text())["args"]
    assert args[args.index("--sandbox") + 1] == "read-only"
    assert 'web_search="live"' in args


@pytest.mark.skipif(os.name != "posix", reason="POSIX process group cleanup")
@pytest.mark.parametrize("cancel", [False, True])
def test_timeout_and_cancellation_reap_cli_and_child(tmp_path: Path, codex_cli: Path, cancel: bool) -> None:
    req = request(tmp_path, timeout=1)
    req.workspace.mkdir()
    (req.workspace / "mode").write_text("cancel" if cancel else "timeout")

    async def run():
        task = asyncio.create_task(provider(codex_cli).run(req))
        if cancel:
            for _ in range(100):
                if (req.workspace / "child.pid").exists():
                    break
                await asyncio.sleep(0.01)
            assert (req.workspace / "child.pid").exists()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        else:
            result = await task
            assert result.status == "timeout"

    asyncio.run(run())
    parent = int((req.workspace / "parent.pid").read_text())
    with pytest.raises(ProcessLookupError):
        os.kill(parent, 0)
    child = int((req.workspace / "child.pid").read_text())
    # An orphan may briefly be a zombie until the operating system reaps it.
    import subprocess

    state = subprocess.run(["ps", "-o", "stat=", "-p", str(child)], capture_output=True, text=True).stdout.strip()
    assert not state or state.startswith("Z")
    manifest = json.loads((req.trace_dir / "codex-session-state/export-manifest.json").read_text())
    assert manifest["status"] in {"timeout", "interrupted"}


@pytest.mark.parametrize("counter", [-1, True, "10", 1.5, None])
def test_invalid_usage_is_rejected(counter) -> None:
    with pytest.raises(ValueError, match="usage counter"):
        _codex_usage({"input_tokens": counter, "output_tokens": 1}, CodexProviderConfig("test"), None)


def config_path(tmp_path: Path, codex_cli: Path) -> Path:
    source = Path(__file__).resolve().parents[3] / "configs/v2/local_template.yaml"
    config = yaml.safe_load(source.read_text())
    config["provider"] = {
        "type": "codex",
        "capabilities": ["read_workspace", "edit_workspace", "load_skills"],
        "settings": {"model": "test-model", "reasoning_effort": None, "executable": str(codex_cli)},
    }
    config["storage"]["run_root"] = str(tmp_path / "runs")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config))
    return path


def test_codex_optimization_and_resume_preserve_usage_and_cli_provenance(tmp_path: Path, codex_cli: Path) -> None:
    path = config_path(tmp_path, codex_cli)
    runtime = build_runtime(path, run_id="codex-run")
    result = runtime.engine.run()
    assert result.development_score == 1.0
    assert (runtime.store.run_dir / "result.json").is_file()
    manifest = json.loads((runtime.store.run_dir / "manifest.json").read_text())
    assert manifest["provider_runtime"]["cli"] == {
        "executable": str(codex_cli),
        "version": "codex-cli 0.153.0",
    }
    calls = (tmp_path / "calls.jsonl").read_text()
    resumed = build_runtime(path, run_id="codex-run").engine.run()
    assert resumed.selected_candidate_id == result.selected_candidate_id
    assert (tmp_path / "calls.jsonl").read_text() == calls

    codex_cli.write_text(codex_cli.read_text().replace("codex-cli 0.153.0", "codex-cli 0.154.0"))
    with pytest.raises(ValueError, match="Resolved run input changed"):
        build_runtime(path, run_id="codex-run")


@pytest.mark.parametrize("change", [{"unknown": True}, {"model": ""}, {"executable": ""}])
def test_codex_registry_rejects_invalid_settings(tmp_path: Path, codex_cli: Path, change: dict) -> None:
    path = config_path(tmp_path, codex_cli)
    config = yaml.safe_load(path.read_text())
    config["provider"]["settings"].update(change)
    path.write_text(yaml.safe_dump(config))
    with pytest.raises((ValueError, TypeError)):
        build_runtime(path, run_id="invalid-codex")
