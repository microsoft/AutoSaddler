from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path, PurePosixPath

import pytest

from autosaddler.v2.core.ports import CompositionPlan, MutationContext
from autosaddler.v2.harness.git import GitHarnessSpace, GitVerificationVerdict, _git_path_exists


def git(path: Path, *arguments: str) -> str:
    return subprocess.run(["git", *arguments], cwd=path, text=True, capture_output=True, check=True).stdout.strip()


def initialized_space(tmp_path: Path) -> GitHarnessSpace:
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    candidate = repo / "candidate"
    candidate.mkdir()
    (candidate / "instructions.txt").write_text("baseline\n")
    (repo / "benchmark.py").write_text("fixed\n")
    (repo / ".gitignore").write_text(".venv/\n.cache/\n*.log\n", encoding="utf-8")
    git(repo, "add", ".")
    git(repo, "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", "baseline")
    return GitHarnessSpace(
        source_repo=repo,
        base_revision="HEAD",
        store_root=tmp_path / "candidates",
        worktree_root=tmp_path / "worktrees",
        writable_paths=(PurePosixPath("candidate"),),
        forbidden_paths=(PurePosixPath("benchmark.py"),),
    )


def context(tmp_path: Path, iteration: int) -> MutationContext:
    return MutationContext(
        iteration=iteration, patch_label="steering", evidence=None, workspace_root=tmp_path / "sessions"
    )


def test_git_space_finalizes_allowlisted_content_addressed_candidate(tmp_path: Path) -> None:
    harness = initialized_space(tmp_path)
    seed = harness.seed()
    session = harness.begin_mutation(seed, context(tmp_path, 0))
    harness.apply_updates(session, {"candidate/instructions.txt": "improved\n"})
    child = harness.finalize(session)

    assert seed.candidate_id.startswith("sha256:")
    assert child.parent_ids == (seed.candidate_id,)
    assert child.change.changed_units == ("candidate/instructions.txt",)
    materialized = harness.materialize(child, "evaluate")
    try:
        assert (materialized.root / "candidate/instructions.txt").read_text() == "improved\n"
        assert git(materialized.root, "status", "--porcelain") == ""
    finally:
        materialized.release()


def test_git_space_replays_finalized_mutation(tmp_path: Path) -> None:
    harness = initialized_space(tmp_path)
    seed = harness.seed()
    session = harness.begin_mutation(seed, context(tmp_path, 0))
    harness.apply_updates(session, {"candidate/instructions.txt": "improved\n"})

    first = harness.finalize(session)
    replayed = harness.finalize(session)

    assert replayed == first


def test_git_space_captures_and_replays_provider_attempt_delta(tmp_path: Path) -> None:
    harness = initialized_space(tmp_path)
    seed = harness.seed()
    session = harness.begin_mutation(seed, context(tmp_path, 0))
    attempt_workspace = tmp_path / "attempt"
    shutil.copytree(session.workspace, attempt_workspace)
    (attempt_workspace / "candidate/instructions.txt").write_text("provider edit\n")

    delta = harness.capture_attempt_delta(
        session,
        attempt_workspace,
        attempt_operation_id="diagnose:attempt:1",
    )

    assert delta.attempt_operation_id == "diagnose:attempt:1"
    assert delta.changed_paths == ("candidate/instructions.txt",)
    assert delta.artifact.sha256.startswith("sha256:")
    harness.apply_attempt_delta(session, delta)
    child = harness.finalize(session)
    materialized = harness.materialize(child, "evaluate")
    try:
        assert (materialized.root / "candidate/instructions.txt").read_text() == "provider edit\n"
    finally:
        materialized.release()


def test_git_space_attempt_delta_ignores_runtime_artifacts(tmp_path: Path) -> None:
    harness = initialized_space(tmp_path)
    seed = harness.seed()
    session = harness.begin_mutation(seed, context(tmp_path, 0))
    attempt_workspace = tmp_path / "attempt"
    shutil.copytree(session.workspace, attempt_workspace)
    (attempt_workspace / ".venv").mkdir()
    (attempt_workspace / ".venv/generated.py").write_text("runtime\n", encoding="utf-8")
    (attempt_workspace / ".cache").mkdir()
    (attempt_workspace / ".cache/state.json").write_text("{}\n", encoding="utf-8")
    (attempt_workspace / "provider.log").write_text("runtime\n", encoding="utf-8")
    (attempt_workspace / "candidate/instructions.txt").write_text(
        "provider edit\n",
        encoding="utf-8",
    )

    delta = harness.capture_attempt_delta(
        session,
        attempt_workspace,
        attempt_operation_id="diagnose:attempt:1",
    )

    assert delta.changed_paths == ("candidate/instructions.txt",)


def test_git_space_attempt_delta_ignores_provider_namespace_only(tmp_path: Path) -> None:
    harness = initialized_space(tmp_path)
    seed = harness.seed()
    session = harness.begin_mutation(seed, context(tmp_path, 0))
    attempt_workspace = tmp_path / "attempt"
    shutil.copytree(session.workspace, attempt_workspace)
    (attempt_workspace / ".autosaddler/session_context.json").write_text("{}\n", encoding="utf-8")
    (attempt_workspace / ".autosaddler/training_evidence.json").write_text("{}\n", encoding="utf-8")
    (attempt_workspace / "candidate/instructions.txt").write_text(
        "provider edit\n",
        encoding="utf-8",
    )

    delta = harness.capture_attempt_delta(
        session,
        attempt_workspace,
        attempt_operation_id="diagnose:attempt:1",
    )

    assert delta.changed_paths == ("candidate/instructions.txt",)

    (attempt_workspace / "session_context.json").write_text("{}\n", encoding="utf-8")
    protected_delta = harness.capture_attempt_delta(
        session,
        attempt_workspace,
        attempt_operation_id="diagnose:attempt:2",
    )
    with pytest.raises(ValueError, match="session_context.json"):
        harness.apply_attempt_delta(session, protected_delta)


def test_git_space_attempt_delta_ignores_sanitized_source_deletions_only(tmp_path: Path) -> None:
    harness = initialized_space(tmp_path)
    source_instructions = harness.source_repo / ".github/copilot-instructions.md"
    source_instructions.parent.mkdir()
    source_instructions.write_text("source instructions\n", encoding="utf-8")
    git(harness.source_repo, "add", ".github/copilot-instructions.md")
    git(
        harness.source_repo,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.com",
        "commit",
        "-qm",
        "add source instructions",
    )
    harness = GitHarnessSpace(
        source_repo=harness.source_repo,
        base_revision="HEAD",
        store_root=tmp_path / "candidates-with-instructions",
        worktree_root=tmp_path / "worktrees-with-instructions",
        writable_paths=(PurePosixPath("candidate"),),
        forbidden_paths=(PurePosixPath("benchmark.py"),),
    )
    seed = harness.seed()
    session = harness.begin_mutation(seed, context(tmp_path, 0))
    attempt_workspace = tmp_path / "sanitized-attempt"
    shutil.copytree(session.workspace, attempt_workspace, ignore=shutil.ignore_patterns(".github"))
    (attempt_workspace / "candidate/instructions.txt").write_text("provider edit\n", encoding="utf-8")

    delta = harness.capture_attempt_delta(
        session,
        attempt_workspace,
        attempt_operation_id="diagnose:attempt:1",
    )

    assert delta.changed_paths == ("candidate/instructions.txt",)

    (attempt_workspace / ".github").mkdir()
    (attempt_workspace / ".github/provider-created.md").write_text("unexpected\n", encoding="utf-8")
    protected_delta = harness.capture_attempt_delta(
        session,
        attempt_workspace,
        attempt_operation_id="diagnose:attempt:2",
    )
    assert ".github/provider-created.md" in protected_delta.changed_paths
    with pytest.raises(ValueError, match=r"\.github/provider-created\.md"):
        harness.apply_attempt_delta(session, protected_delta)


def test_git_space_rejects_protected_path_even_when_edited_directly(tmp_path: Path) -> None:
    harness = initialized_space(tmp_path)
    seed = harness.seed()
    session = harness.begin_mutation(seed, context(tmp_path, 0))
    (session.workspace / "benchmark.py").write_text("tampered\n")

    with pytest.raises(ValueError, match="protected paths"):
        harness.finalize(session)


def test_git_space_rejects_workspace_changed_by_verifier(tmp_path: Path) -> None:
    harness = initialized_space(tmp_path)
    seed = harness.seed()
    session = harness.begin_mutation(seed, context(tmp_path, 0))
    harness.apply_updates(session, {"candidate/instructions.txt": "checked edit\n"})

    def mutating_verifier(verification_context):
        (verification_context.workspace / "candidate/instructions.txt").write_text(
            "post-verification edit\n",
            encoding="utf-8",
        )
        (verification_context.workspace / "candidate/verifier-created.txt").write_text(
            "must not survive\n",
            encoding="utf-8",
        )
        return GitVerificationVerdict(
            accepted=True,
            check="test",
            summary="accepted before mutation",
            artifacts=(),
        )

    harness.verifier = mutating_verifier

    with pytest.raises(RuntimeError, match="Mutation workspace changed during verification"):
        harness.finalize(session)

    assert (session.workspace / "candidate/instructions.txt").read_text(encoding="utf-8") == "checked edit\n"
    assert not (session.workspace / "candidate/verifier-created.txt").exists()


def test_git_space_restores_workspace_changed_by_rejecting_verifier(tmp_path: Path) -> None:
    harness = initialized_space(tmp_path)
    seed = harness.seed()
    session = harness.begin_mutation(seed, context(tmp_path, 0))
    harness.apply_updates(session, {"candidate/instructions.txt": "provider edit\n"})

    def rejecting_verifier(verification_context):
        (verification_context.workspace / "candidate/verifier-created.txt").write_text(
            "must not survive\n",
            encoding="utf-8",
        )
        return GitVerificationVerdict(
            accepted=False,
            check="test_rejection",
            summary="reject provider edit",
            artifacts=(),
        )

    harness.verifier = rejecting_verifier
    with pytest.raises(RuntimeError, match="Mutation workspace changed during verification"):
        harness.finalize(session)

    assert (session.workspace / "candidate/instructions.txt").read_text(encoding="utf-8") == "provider edit\n"
    assert not (session.workspace / "candidate/verifier-created.txt").exists()

    harness.verifier = lambda _verification: GitVerificationVerdict(
        accepted=True,
        check="clean_resume",
        summary="provider edit is valid",
        artifacts=(),
    )
    child = harness.finalize(session)
    assert child.change is not None
    assert child.change.changed_units == ("candidate/instructions.txt",)


def test_git_space_restores_workspace_when_verifier_raises(tmp_path: Path) -> None:
    harness = initialized_space(tmp_path)
    seed = harness.seed()
    session = harness.begin_mutation(seed, context(tmp_path, 0))
    harness.apply_updates(session, {"candidate/instructions.txt": "provider edit\n"})

    def failing_verifier(verification_context):
        (verification_context.workspace / "candidate/instructions.txt").write_text(
            "exception-path edit\n",
            encoding="utf-8",
        )
        raise RuntimeError("verifier failed")

    harness.verifier = failing_verifier
    with pytest.raises(RuntimeError, match="verifier failed"):
        harness.finalize(session)

    assert (session.workspace / "candidate/instructions.txt").read_text(encoding="utf-8") == "provider edit\n"


def test_git_candidate_identity_ignores_commit_metadata(tmp_path: Path) -> None:
    harness = initialized_space(tmp_path)
    seed = harness.seed()
    first_session = harness.begin_mutation(seed, context(tmp_path, 1))
    harness.apply_updates(first_session, {"candidate/instructions.txt": "same child\n"})
    first = harness.finalize(first_session)
    second_session = harness.begin_mutation(seed, context(tmp_path, 2))
    harness.apply_updates(second_session, {"candidate/instructions.txt": "same child\n"})
    second = harness.finalize(second_session)

    assert first.candidate_id == second.candidate_id


def test_git_space_replays_composition_and_removes_workspace(tmp_path: Path) -> None:
    harness = initialized_space(tmp_path)
    seed = harness.seed()
    first_session = harness.begin_mutation(seed, context(tmp_path, 1))
    harness.apply_updates(first_session, {"candidate/instructions.txt": "first\n"})
    first = harness.finalize(first_session)
    second_session = harness.begin_mutation(seed, context(tmp_path, 2))
    harness.apply_updates(second_session, {"candidate/instructions.txt": "second\n"})
    second = harness.finalize(second_session)
    plan = CompositionPlan(
        parents=(first.candidate_id, second.candidate_id),
        selections={"candidate/instructions.txt": second.candidate_id},
        overrides={},
        rationale="Use the second candidate's instruction.",
    )

    composed = harness.compose(plan)
    replayed = harness.compose(plan)

    assert replayed == composed
    assert not any((tmp_path / "worktrees").glob("compose-*"))


def test_git_space_composes_a_source_file_deletion(tmp_path: Path) -> None:
    harness = initialized_space(tmp_path)
    seed = harness.seed()
    keep_session = harness.begin_mutation(seed, context(tmp_path, 1))
    harness.apply_updates(keep_session, {"candidate/instructions.txt": "retained base\n"})
    keep = harness.finalize(keep_session)
    delete_session = harness.begin_mutation(seed, context(tmp_path, 2))
    (delete_session.workspace / "candidate/instructions.txt").unlink()
    deleted = harness.finalize(delete_session)

    composed = harness.compose(
        CompositionPlan(
            parents=(keep.candidate_id, deleted.candidate_id),
            selections={"candidate/instructions.txt": deleted.candidate_id},
            overrides={},
            rationale="Compose the measured deletion.",
        )
    )
    materialized = harness.materialize(composed, "inspect")
    try:
        assert not (materialized.root / "candidate/instructions.txt").exists()
    finally:
        materialized.release()


def test_git_path_lookup_fails_closed_for_invalid_revision(tmp_path: Path) -> None:
    harness = initialized_space(tmp_path)

    with pytest.raises(subprocess.CalledProcessError):
        _git_path_exists(
            harness.source_repo,
            "invalid-revision",
            PurePosixPath("candidate/instructions.txt"),
        )


def test_git_candidate_identity_excludes_provider_assets(tmp_path: Path) -> None:
    harness = initialized_space(tmp_path)
    seed = harness.seed()
    session = harness.begin_mutation(seed, context(tmp_path, 0))
    (session.workspace / "AGENTS.md").write_text("provider instructions\n")
    (session.workspace / ".copilot").mkdir()
    (session.workspace / ".copilot/instructions.md").write_text("provider instructions\n")
    (session.workspace / "candidate/instructions.txt").write_text("candidate edit\n")

    child = harness.finalize(session)

    assert child.change is not None
    assert child.change.changed_units == ("candidate/instructions.txt",)
    materialized = harness.materialize(child, "inspect")
    try:
        assert not (materialized.root / "AGENTS.md").exists()
        assert not (materialized.root / ".copilot").exists()
    finally:
        materialized.release()


def test_git_space_runs_structured_verifier_before_publishing(tmp_path: Path) -> None:
    from autosaddler.v2.harness.git import GitVerificationContext, GitVerificationVerdict

    seen: list[GitVerificationContext] = []

    def verifier(verification: GitVerificationContext) -> GitVerificationVerdict:
        seen.append(verification)
        return GitVerificationVerdict(
            accepted=False,
            check="python_compile",
            summary="invalid Python",
            artifacts=(),
        )

    base = initialized_space(tmp_path)
    harness = GitHarnessSpace(
        source_repo=base.source_repo,
        base_revision=base.base_revision,
        store_root=tmp_path / "verified-candidates",
        worktree_root=tmp_path / "verified-worktrees",
        writable_paths=(PurePosixPath("candidate"),),
        verifier=verifier,
    )
    seed = harness.seed()
    session = harness.begin_mutation(seed, context(tmp_path, 0))
    (session.workspace / "candidate/broken.py").write_text("def broken(:\n")

    with pytest.raises(ValueError, match="invalid Python"):
        harness.finalize(session)

    assert seen[0].parent.candidate_id == seed.candidate_id
    assert seen[0].changed_paths == (PurePosixPath("candidate/broken.py"),)
    assert seen[0].patch_label == "steering"


def test_git_space_verifies_provider_committed_workspace(tmp_path: Path) -> None:
    from autosaddler.v2.harness.git import GitVerificationVerdict

    def verifier(_verification) -> GitVerificationVerdict:
        return GitVerificationVerdict(
            accepted=False,
            check="steering_scope",
            summary="provider commit is still subject to verification",
            artifacts=(),
        )

    base = initialized_space(tmp_path)
    harness = GitHarnessSpace(
        source_repo=base.source_repo,
        base_revision=base.base_revision,
        store_root=tmp_path / "verified-candidates",
        worktree_root=tmp_path / "verified-worktrees",
        writable_paths=(PurePosixPath("candidate"),),
        verifier=verifier,
    )
    session = harness.begin_mutation(harness.seed(), context(tmp_path, 0))
    (session.workspace / "candidate/instructions.txt").write_text("committed edit\n", encoding="utf-8")
    git(session.workspace, "add", "candidate/instructions.txt")
    git(
        session.workspace,
        "-c",
        "user.name=Provider",
        "-c",
        "user.email=provider@example.com",
        "commit",
        "-qm",
        "provider commit",
    )

    with pytest.raises(ValueError, match="provider commit is still subject to verification"):
        harness.finalize(session)


def test_git_space_rejects_stale_materialization_identity(tmp_path: Path) -> None:
    harness = initialized_space(tmp_path)
    seed = harness.seed()
    stale_root = tmp_path / "worktrees" / f"evaluate-{seed.candidate_id.removeprefix('sha256:')}"
    stale_root.mkdir(parents=True)
    (stale_root / ".autosaddler").mkdir()
    (stale_root / ".autosaddler/materialization.json").write_text(
        '{"candidate_id":"sha256:stale","purpose":"evaluate"}\n'
    )

    with pytest.raises(RuntimeError, match="materialization metadata mismatch"):
        harness.materialize(seed, "evaluate")


def test_git_space_replaces_materialization_without_metadata(tmp_path: Path) -> None:
    harness = initialized_space(tmp_path)
    seed = harness.seed()
    stale_root = tmp_path / "worktrees" / f"evaluate-{seed.candidate_id.removeprefix('sha256:')}"
    stale_root.mkdir(parents=True)
    (stale_root / "orphaned.txt").write_text("incomplete materialization\n")

    materialized = harness.materialize(seed, "evaluate")
    try:
        assert not (materialized.root / "orphaned.txt").exists()
        assert json.loads((materialized.root / ".autosaddler/materialization.json").read_text()) == {
            "candidate_id": seed.candidate_id,
            "purpose": "evaluate",
        }
    finally:
        materialized.release()


def test_git_space_rejected_attempt_delta_does_not_contaminate_next_attempt(tmp_path: Path) -> None:
    harness = initialized_space(tmp_path)
    seed = harness.seed()
    session = harness.begin_mutation(seed, context(tmp_path, 0))
    rejected = tmp_path / "rejected-attempt"
    shutil.copytree(session.workspace, rejected)
    (rejected / "candidate/instructions.txt").write_text("rejected edit\n")
    (rejected / "benchmark.py").write_text("tampered\n")

    rejected_delta = harness.capture_attempt_delta(
        session,
        rejected,
        attempt_operation_id="attempt:rejected",
    )
    with pytest.raises(ValueError, match="protected paths"):
        harness.apply_attempt_delta(session, rejected_delta)

    clean_attempt = tmp_path / "clean-attempt"
    shutil.copytree(session.workspace, clean_attempt)
    assert (clean_attempt / "candidate/instructions.txt").read_text() == "baseline\n"


def test_git_space_attempt_delta_round_trips_binary_content(tmp_path: Path) -> None:
    harness = initialized_space(tmp_path)
    seed = harness.seed()
    session = harness.begin_mutation(seed, context(tmp_path, 0))
    attempt = tmp_path / "binary-attempt"
    shutil.copytree(session.workspace, attempt)
    payload = bytes(range(256))
    (attempt / "candidate/blob.bin").write_bytes(payload)

    delta = harness.capture_attempt_delta(session, attempt, attempt_operation_id="attempt:binary")
    harness.apply_attempt_delta(session, delta)
    child = harness.finalize(session)
    materialized = harness.materialize(child, "inspect")
    try:
        assert (materialized.root / "candidate/blob.bin").read_bytes() == payload
    finally:
        materialized.release()
