from __future__ import annotations

import os
import subprocess
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from autosaddler.v2.core.domain import ArtifactRef, Candidate, sha256_digest

if TYPE_CHECKING:
    from autosaddler.v2.harness.git import GitVerificationContext


def test_verifier_rejects_invalid_python_hook_and_steering_code(tmp_path: Path) -> None:
    from autosaddler.v2.plugins.meta_are.verification import MetaAREVerifier

    workspace = tmp_path / "workspace"
    agent_root = workspace / "are/simulation/agents/default_agent"
    agent_root.mkdir(parents=True)
    verifier = MetaAREVerifier(
        import_check="import candidate_agent",
        verification_timeout_seconds=30,
        train_case_ids=("secret-train-case",),
    )

    broken = agent_root / "broken.py"
    broken.write_text("def broken(:\n", encoding="utf-8")
    verdict = verifier(_context(workspace, (PurePosixPath("are/simulation/agents/default_agent/broken.py"),)))
    assert not verdict.accepted
    assert verdict.check == "python_compile"

    broken.unlink()
    (workspace / "hook.json").write_text('{"hooks": [}', encoding="utf-8")
    verdict = verifier(_context(workspace, (PurePosixPath("hook.json"),)))
    assert not verdict.accepted
    assert verdict.check == "hook_config"

    (workspace / "hook.json").write_text(
        '{"hooks":{"PreToolUse":[{"matcher":"[","hooks":[]}]}}',
        encoding="utf-8",
    )
    verdict = verifier(_context(workspace, (PurePosixPath("hook.json"),)))
    assert not verdict.accepted
    assert verdict.check == "hook_config"
    assert "matcher is invalid" in verdict.summary

    (workspace / "hook.json").write_text(
        '{"hooks":{"PreToolUse":[{"matcher":".*","hooks":[{"type":"reminder","reminder":""}]}]}}',
        encoding="utf-8",
    )
    verdict = verifier(_context(workspace, (PurePosixPath("hook.json"),)))
    assert not verdict.accepted
    assert verdict.check == "hook_config"
    assert "reminder must be a non-empty string" in verdict.summary

    (workspace / "hook.json").unlink()
    code = agent_root / "agent.py"
    code.write_text("VALUE = 1\n", encoding="utf-8")
    verdict = verifier(
        _context(
            workspace,
            (PurePosixPath("are/simulation/agents/default_agent/agent.py"),),
            patch_label="steering",
        )
    )
    assert not verdict.accepted
    assert verdict.check == "steering_scope"


def test_verifier_rejects_case_specific_lookup_and_accepts_valid_capability_edit(tmp_path: Path) -> None:
    from autosaddler.v2.plugins.meta_are.verification import MetaAREVerifier

    workspace = tmp_path / "workspace"
    agent_root = workspace / "are/simulation/agents/default_agent"
    agent_root.mkdir(parents=True)
    code = agent_root / "agent.py"
    verifier = MetaAREVerifier(
        import_check="",
        verification_timeout_seconds=30,
        train_case_ids=("scenario_universe_29_secret",),
    )
    code.write_text('TARGET = "scenario_universe_29_secret"\n', encoding="utf-8")
    verdict = verifier(_context(workspace, (PurePosixPath("are/simulation/agents/default_agent/agent.py"),)))
    assert not verdict.accepted
    assert verdict.check == "anti_cheating"

    code.write_text("def choose_tool(name: str) -> str:\n    return name\n", encoding="utf-8")
    verdict = verifier(_context(workspace, (PurePosixPath("are/simulation/agents/default_agent/agent.py"),)))
    assert verdict.accepted
    assert verdict.check == "all"


def test_verifier_import_check_uses_candidate_workspace_project(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from autosaddler.v2.plugins.meta_are.verification import MetaAREVerifier

    workspace = tmp_path / "workspace"
    code = workspace / "are/simulation/agents/default_agent/agent.py"
    code.parent.mkdir(parents=True)
    code.write_text("VALUE = 1\n", encoding="utf-8")

    def fake_run(argv, **kwargs):
        assert argv[:4] == ["uv", "run", "--project", str(workspace)]
        assert kwargs["cwd"] == workspace
        assert kwargs["env"]["PYTHONPATH"].split(os.pathsep)[0] == str(workspace)
        assert kwargs["timeout"] == 30
        return subprocess.CompletedProcess(argv, 0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    verifier = MetaAREVerifier(
        import_check="import candidate_agent",
        verification_timeout_seconds=30,
        train_case_ids=(),
    )

    verdict = verifier(
        _context(
            workspace,
            (PurePosixPath("are/simulation/agents/default_agent/agent.py"),),
        )
    )

    assert verdict.accepted


def test_verifier_rejects_import_check_timeout(tmp_path: Path, monkeypatch) -> None:
    from autosaddler.v2.plugins.meta_are.verification import MetaAREVerifier

    workspace = tmp_path / "workspace"
    code = workspace / "are/simulation/agents/default_agent/agent.py"
    code.parent.mkdir(parents=True)
    code.write_text("VALUE = 1\n", encoding="utf-8")

    def timeout(argv, **_kwargs):
        raise subprocess.TimeoutExpired(argv, 5)

    monkeypatch.setattr(subprocess, "run", timeout)
    verifier = MetaAREVerifier(
        import_check="import candidate_agent",
        verification_timeout_seconds=5,
        train_case_ids=(),
    )

    verdict = verifier(
        _context(
            workspace,
            (PurePosixPath("are/simulation/agents/default_agent/agent.py"),),
        )
    )

    assert not verdict.accepted
    assert verdict.check == "import_check_timeout"
    assert "5 seconds" in verdict.summary


def _context(
    workspace: Path,
    changed_paths: tuple[PurePosixPath, ...],
    *,
    patch_label: str = "capability",
) -> GitVerificationContext:
    from autosaddler.v2.harness.git import GitVerificationContext

    parent = Candidate(
        candidate_id=sha256_digest("parent"),
        parent_ids=(),
        space="git",
        artifact=ArtifactRef(uri="candidates/parent.json", kind="git-tree"),
    )
    return GitVerificationContext(
        parent=parent,
        workspace=workspace,
        changed_paths=changed_paths,
        patch_label=patch_label,
    )
