from __future__ import annotations

import base64
import json
import os
import shutil
import stat
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Callable

from autosaddler.v2.core.domain import ArtifactRef, Candidate, ChangeSummary, JsonValue, canonical_json, sha256_digest
from autosaddler.v2.core.ports import (
    CompositionPlan,
    MaterializedHarness,
    MutationContext,
    MutationOutcome,
    MutationSession,
    WorkspaceDelta,
)

PROVIDER_ASSETS = (
    PurePosixPath("CLAUDE.md"),
    PurePosixPath("AGENTS.md"),
    PurePosixPath(".claude"),
    PurePosixPath(".copilot"),
    PurePosixPath(".autosaddler"),
)
SANITIZED_SOURCE_ASSETS = (PurePosixPath(".github"),)


@dataclass(frozen=True, slots=True)
class GitVerificationContext:
    parent: Candidate
    workspace: Path
    changed_paths: tuple[PurePosixPath, ...]
    patch_label: str


@dataclass(frozen=True, slots=True)
class GitVerificationVerdict:
    accepted: bool
    check: str
    summary: str
    artifacts: tuple[ArtifactRef, ...]


GitVerifier = Callable[[GitVerificationContext], GitVerificationVerdict]


class GitHarnessSpace:
    """A detached-worktree candidate space with immutable tree-derived identity."""

    def __init__(
        self,
        *,
        source_repo: Path,
        base_revision: str,
        store_root: Path,
        worktree_root: Path,
        writable_paths: Sequence[PurePosixPath],
        forbidden_paths: Sequence[PurePosixPath] = (),
        verifier: GitVerifier | None = None,
    ) -> None:
        self.source_repo = source_repo.resolve()
        self.base_revision = _git(self.source_repo, "rev-parse", base_revision)
        self.store_root = store_root.resolve()
        self.worktree_root = worktree_root.resolve()
        self.writable_paths = _validated_roots(writable_paths, "writable")
        self.forbidden_paths = _validated_roots(forbidden_paths, "forbidden", allow_empty=True)
        overlap = [path for path in self.writable_paths if _within(path, self.forbidden_paths)]
        if overlap:
            raise ValueError(f"Writable paths overlap forbidden paths: {overlap}")
        self.verifier = verifier
        self._sessions: dict[str, tuple[str, ...]] = {}

    def seed(self) -> Candidate:
        return self._publish(self.base_revision, parent_ids=(), change=None)

    def begin_mutation(self, parent: Candidate, context: MutationContext) -> MutationSession:
        revision = self._revision(parent.candidate_id)
        session_id = sha256_digest(
            canonical_json(
                {
                    "parent_id": parent.candidate_id,
                    "iteration": context.iteration,
                    "patch_label": context.patch_label,
                }
            )
        )
        workspace = context.workspace_root.resolve() / session_id.removeprefix("sha256:")
        metadata_path = workspace / ".autosaddler/session.json"
        if workspace.exists():
            metadata = _read_json(metadata_path)
            if metadata.get("parent_id") != parent.candidate_id:
                raise RuntimeError(f"Mutation workspace parent drifted: {workspace}")
        else:
            _git_worktree_add(self.source_repo, workspace, revision)
            _write_json(
                metadata_path,
                {
                    "session_id": session_id,
                    "parent_id": parent.candidate_id,
                    "patch_label": context.patch_label,
                },
            )
        self._sessions[session_id] = (parent.candidate_id,)
        return MutationSession(
            session_id=session_id,
            parent=parent,
            workspace=workspace,
            context=context,
        )

    def capture_attempt_delta(
        self,
        session: MutationSession,
        attempt_workspace: Path,
        *,
        attempt_operation_id: str,
    ) -> WorkspaceDelta:
        parent_entries = _workspace_entries(session.workspace, session.workspace)
        attempt_entries = _workspace_entries(attempt_workspace, session.workspace)
        changed_paths = tuple(
            sorted(
                path
                for path in parent_entries.keys() | attempt_entries.keys()
                if parent_entries.get(path) != attempt_entries.get(path)
                and not (path not in attempt_entries and _within(PurePosixPath(path), SANITIZED_SOURCE_ASSETS))
            )
        )
        payload = {
            "schema_version": "autosaddler-workspace-delta/v1",
            "attempt_operation_id": attempt_operation_id,
            "parent_candidate_id": session.parent.candidate_id,
            "entries": [
                {
                    "path": path,
                    "content_base64": attempt_entries[path][0] if path in attempt_entries else None,
                    "executable": attempt_entries[path][1] if path in attempt_entries else None,
                }
                for path in changed_paths
            ],
        }
        encoded = (canonical_json(payload) + "\n").encode("utf-8")
        digest = sha256_digest(encoded)
        relative_uri = f"mutation-deltas/{digest.removeprefix('sha256:')}.json"
        artifact_path = self.store_root.parent / relative_uri
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        if artifact_path.exists():
            if artifact_path.read_bytes() != encoded:
                raise RuntimeError(f"Workspace delta content collision: {digest}")
        else:
            _atomic_write_bytes(artifact_path, encoded)
            artifact_path.chmod(0o444)
        return WorkspaceDelta(
            attempt_operation_id=attempt_operation_id,
            changed_paths=changed_paths,
            artifact=ArtifactRef(
                uri=relative_uri,
                kind="workspace-delta",
                sha256=digest,
                bytes=len(encoded),
            ),
        )

    def apply_mutation(self, session: MutationSession, outcome: MutationOutcome) -> None:
        delta = outcome.workspace_delta
        if delta is None:
            raise ValueError("Git mutation requires a durable workspace delta")
        self.apply_attempt_delta(session, delta)

    def apply_attempt_delta(self, session: MutationSession, delta: WorkspaceDelta) -> None:
        artifact_path = self.store_root.parent / delta.artifact.uri
        encoded = artifact_path.read_bytes()
        if delta.artifact.sha256 != sha256_digest(encoded):
            raise ValueError("Workspace delta digest mismatch")
        value = json.loads(encoded)
        if not isinstance(value, dict) or value.get("schema_version") != "autosaddler-workspace-delta/v1":
            raise TypeError("Workspace delta artifact is malformed")
        if value.get("attempt_operation_id") != delta.attempt_operation_id:
            raise ValueError("Workspace delta attempt identity mismatch")
        if value.get("parent_candidate_id") != session.parent.candidate_id:
            raise ValueError("Workspace delta parent identity mismatch")
        entries = value.get("entries")
        if not isinstance(entries, list):
            raise TypeError("Workspace delta entries must be a list")
        if not all(isinstance(entry, dict) for entry in entries):
            raise TypeError("Workspace delta entries must be objects")
        paths = tuple(_relative_path(str(entry.get("path", ""))) for entry in entries)
        if tuple(str(path) for path in paths) != delta.changed_paths:
            raise ValueError("Workspace delta changed paths mismatch")
        self._require_writable(paths)
        for path, entry in zip(paths, entries, strict=True):
            destination = session.workspace.joinpath(*path.parts)
            content = entry.get("content_base64")
            if content is None:
                if destination.is_dir():
                    shutil.rmtree(destination)
                else:
                    destination.unlink(missing_ok=True)
                continue
            if not isinstance(content, str) or not isinstance(entry.get("executable"), bool):
                raise TypeError(f"Malformed workspace delta entry: {path}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(base64.b64decode(content, validate=True))
            destination.chmod(0o755 if entry["executable"] else 0o644)

    def apply_updates(self, session: MutationSession, updates: Mapping[str, JsonValue]) -> None:
        for raw_path, content in updates.items():
            path = _relative_path(raw_path)
            self._require_writable((path,))
            if not isinstance(content, str):
                raise TypeError(f"Git update content must be text: {raw_path}")
            destination = session.workspace.joinpath(*path.parts)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_text(content, encoding="utf-8")

    def finalize(self, session: MutationSession) -> Candidate:
        parent_ids = self._sessions.get(session.session_id)
        if parent_ids is None:
            raise RuntimeError(f"Unknown mutation session: {session.session_id}")
        parent_revision = self._revision(session.parent.candidate_id)
        workspace_revision = _git(session.workspace, "rev-parse", "HEAD")
        changed_paths = self._changed_paths(session.workspace, parent_revision)
        self._require_writable(changed_paths)
        if not changed_paths:
            raise ValueError("Candidate mutation must change at least one allowlisted file")
        if self.verifier is not None:
            pre_verification_entries = _workspace_entries(session.workspace, session.workspace)
            try:
                verdict = self.verifier(
                    GitVerificationContext(
                        parent=session.parent,
                        workspace=session.workspace,
                        changed_paths=changed_paths,
                        patch_label=session.context.patch_label,
                    )
                )
            except Exception:
                _restore_workspace_entries(
                    session.workspace,
                    session.workspace,
                    pre_verification_entries,
                )
                raise
            if _workspace_entries(session.workspace, session.workspace) != pre_verification_entries:
                _restore_workspace_entries(
                    session.workspace,
                    session.workspace,
                    pre_verification_entries,
                )
                raise RuntimeError(f"Mutation workspace changed during verification: {session.workspace}")
            if not verdict.accepted:
                raise ValueError(f"{verdict.check}: {verdict.summary}")
        if workspace_revision == parent_revision:
            _git(session.workspace, "add", "--", *(str(path) for path in changed_paths))
            _git(
                session.workspace,
                "-c",
                "user.name=AutoSaddler",
                "-c",
                "user.email=autosaddler@local",
                "commit",
                "-m",
                f"AutoSaddler candidate {session.session_id[:20]}",
            )
            revision = _git(session.workspace, "rev-parse", "HEAD")
        else:
            if _git(session.workspace, "rev-parse", "HEAD^") != parent_revision:
                raise RuntimeError(f"Mutation workspace revision drifted: {session.workspace}")
            dirty_paths = self._changed_paths(session.workspace, workspace_revision)
            self._require_writable(dirty_paths)
            if dirty_paths:
                raise RuntimeError(f"Finalized mutation workspace changed during replay: {session.workspace}")
            revision = workspace_revision
        change = self._change_summary(
            parent_revision,
            revision,
            labels=(session.context.patch_label,),
        )
        return self._publish(revision, parent_ids=parent_ids, change=change)

    def compose(self, plan: CompositionPlan) -> Candidate:
        if not plan.parents:
            raise ValueError("Composition requires at least one parent")
        if len(set(plan.parents)) != len(plan.parents):
            raise ValueError("Composition parents must be unique")
        parent_candidates = {candidate_id: self._revision(candidate_id) for candidate_id in plan.parents}
        session_key = sha256_digest(canonical_json({"plan": plan, "kind": "git-composition"}))
        workspace = self.worktree_root / f"compose-{session_key.removeprefix('sha256:')}"
        if workspace.exists():
            self._remove_worktree(workspace)
        _git_worktree_add(self.source_repo, workspace, parent_candidates[plan.parents[0]])
        seed = Candidate(
            candidate_id=plan.parents[0],
            parent_ids=(),
            space="git",
            artifact=self._artifact(plan.parents[0]),
        )
        session = MutationSession(
            session_id=session_key,
            parent=seed,
            workspace=workspace,
            context=MutationContext(
                iteration=-1,
                patch_label="composition",
                evidence=None,
                workspace_root=self.worktree_root,
            ),
        )
        self._sessions[session_key] = plan.parents
        try:
            for raw_path, source_id in plan.selections.items():
                path = _relative_path(raw_path)
                self._require_writable((path,))
                if source_id not in parent_candidates:
                    raise ValueError(f"Composition source is not a declared parent: {source_id}")
                destination = workspace.joinpath(*path.parts)
                source_revision = parent_candidates[source_id]
                if _git_path_exists(self.source_repo, source_revision, path):
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(
                        _git_bytes(self.source_repo, "show", f"{source_revision}:{path.as_posix()}")
                    )
                else:
                    destination.unlink(missing_ok=True)
            self.apply_updates(session, plan.overrides)
            candidate = self.finalize(session)
            return Candidate(
                candidate_id=candidate.candidate_id,
                parent_ids=candidate.parent_ids,
                space=candidate.space,
                artifact=candidate.artifact,
                change=ChangeSummary(
                    changed_units=candidate.change.changed_units,
                    added=candidate.change.added,
                    removed=candidate.change.removed,
                    labels=("composition",),
                    diff=candidate.change.diff,
                )
                if candidate.change
                else None,
            )
        finally:
            self._remove_worktree(workspace)

    def materialize(self, candidate: Candidate, purpose: str) -> MaterializedHarness:
        if purpose not in {"evaluate", "inspect", "mutate"}:
            raise ValueError(f"Unknown materialization purpose: {purpose}")
        revision = self._revision(candidate.candidate_id)
        root = self.worktree_root / f"{purpose}-{candidate.candidate_id.removeprefix('sha256:')}"
        if root.exists():
            metadata_path = root / ".autosaddler/materialization.json"
            if metadata_path.is_file():
                metadata = _read_json(metadata_path)
                expected = {"candidate_id": candidate.candidate_id, "purpose": purpose}
                if metadata != expected:
                    raise RuntimeError(f"Git materialization metadata mismatch: {root}")
            self._remove_worktree(root)
        _git_worktree_add(self.source_repo, root, revision)
        _write_json(
            root / ".autosaddler/materialization.json",
            {"candidate_id": candidate.candidate_id, "purpose": purpose},
        )
        _exclude_provider_assets(root)
        return MaterializedHarness(
            root=root,
            candidate_id=candidate.candidate_id,
            release=lambda: self._remove_worktree(root),
        )

    def diff(self, parent: Candidate, child: Candidate) -> ChangeSummary:
        return self._change_summary(self._revision(parent.candidate_id), self._revision(child.candidate_id), labels=())

    def _publish(
        self,
        revision: str,
        *,
        parent_ids: tuple[str, ...],
        change: ChangeSummary | None,
    ) -> Candidate:
        candidate_id = _tree_identity(self.source_repo, revision)
        digest = candidate_id.removeprefix("sha256:")
        candidate_dir = self.store_root / digest
        manifest_path = candidate_dir / "candidate.json"
        manifest = {"candidate_id": candidate_id, "revision": revision}
        if manifest_path.exists():
            existing = _read_json(manifest_path)
            if existing.get("candidate_id") != candidate_id:
                raise RuntimeError(f"Git candidate content collision: {candidate_id}")
            revision = str(existing["revision"])
        else:
            candidate_dir.mkdir(parents=True, exist_ok=True)
            _write_json(manifest_path, manifest)
            manifest_path.chmod(0o444)
            _git(self.source_repo, "update-ref", f"refs/autosaddler/candidates/{digest}", revision)
        return Candidate(
            candidate_id=candidate_id,
            parent_ids=parent_ids,
            space="git",
            artifact=self._artifact(candidate_id),
            change=change,
        )

    def _artifact(self, candidate_id: str) -> ArtifactRef:
        digest = candidate_id.removeprefix("sha256:")
        manifest_path = self.store_root / digest / "candidate.json"
        payload = manifest_path.read_bytes()
        return ArtifactRef(
            uri=f"candidates/{digest}/candidate.json",
            kind="git-tree",
            sha256=sha256_digest(payload),
            bytes=len(payload),
        )

    def _revision(self, candidate_id: str) -> str:
        manifest = _read_json(self.store_root / candidate_id.removeprefix("sha256:") / "candidate.json")
        revision = str(manifest.get("revision", ""))
        if manifest.get("candidate_id") != candidate_id or _tree_identity(self.source_repo, revision) != candidate_id:
            raise RuntimeError(f"Git candidate manifest drifted: {candidate_id}")
        return revision

    def _changed_paths(self, workspace: Path, parent_revision: str) -> tuple[PurePosixPath, ...]:
        tracked = _git(workspace, "diff", "--name-only", parent_revision, "--").splitlines()
        untracked = _git(workspace, "ls-files", "--others", "--exclude-standard").splitlines()
        paths = {_relative_path(item) for item in (*tracked, *untracked) if item}
        return tuple(sorted((path for path in paths if not _within(path, PROVIDER_ASSETS)), key=str))

    def _require_writable(self, paths: Sequence[PurePosixPath]) -> None:
        protected = [
            str(path) for path in paths if not _within(path, self.writable_paths) or _within(path, self.forbidden_paths)
        ]
        if protected:
            raise ValueError(f"Candidate mutation modifies protected paths: {', '.join(sorted(protected))}")

    def _change_summary(
        self,
        parent_revision: str,
        child_revision: str,
        *,
        labels: tuple[str, ...],
    ) -> ChangeSummary:
        paths = tuple(
            line
            for line in _git(
                self.source_repo, "diff", "--name-only", parent_revision, child_revision, "--"
            ).splitlines()
            if line
        )
        added = 0
        removed = 0
        for row in _git(self.source_repo, "diff", "--numstat", parent_revision, child_revision, "--").splitlines():
            add_text, remove_text, _ = row.split("\t", 2)
            if add_text.isdigit():
                added += int(add_text)
            if remove_text.isdigit():
                removed += int(remove_text)
        child_id = _tree_identity(self.source_repo, child_revision)
        digest = child_id.removeprefix("sha256:")
        patch = _git_bytes(self.source_repo, "diff", "--binary", parent_revision, child_revision, "--")
        patch_path = self.store_root / digest / "change.patch"
        patch_path.parent.mkdir(parents=True, exist_ok=True)
        if not patch_path.exists():
            patch_path.write_bytes(patch)
            patch_path.chmod(0o444)
        return ChangeSummary(
            changed_units=paths,
            added=added,
            removed=removed,
            labels=labels,
            diff=ArtifactRef(
                uri=f"candidates/{digest}/change.patch",
                kind="git-diff",
                sha256=sha256_digest(patch),
                bytes=len(patch),
            ),
        )

    def _remove_worktree(self, path: Path) -> None:
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(path)],
            cwd=self.source_repo,
            text=True,
            capture_output=True,
            check=False,
        )
        if path.exists():
            shutil.rmtree(path)


def _validated_roots(
    paths: Sequence[PurePosixPath],
    label: str,
    *,
    allow_empty: bool = False,
) -> tuple[PurePosixPath, ...]:
    roots = tuple(paths)
    if not roots and not allow_empty:
        raise ValueError(f"Git {label} paths cannot be empty")
    for path in roots:
        _relative_path(str(path))
    return roots


def _relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if not value or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Candidate path must be repository-relative: {value}")
    return path


def _within(path: PurePosixPath, roots: Sequence[PurePosixPath]) -> bool:
    return any(path == root or root in path.parents for root in roots)


def _workspace_entries(root: Path, git_reference: Path) -> dict[str, tuple[str, bool]]:
    entries: dict[str, tuple[str, bool]] = {}
    git_dir = _git(git_reference, "rev-parse", "--absolute-git-dir")
    listed = subprocess.run(
        [
            "git",
            f"--git-dir={git_dir}",
            f"--work-tree={root}",
            "ls-files",
            "--cached",
            "--others",
            "--exclude-standard",
            "-z",
        ],
        cwd=git_reference,
        capture_output=True,
        check=True,
    ).stdout
    for raw_path in listed.split(b"\0"):
        if not raw_path:
            continue
        relative = _relative_path(raw_path.decode("utf-8"))
        if _within(relative, PROVIDER_ASSETS):
            continue
        path = root.joinpath(*relative.parts)
        if path.is_symlink():
            raise ValueError(f"Candidate mutation may not create symlinks: {relative}")
        if not path.is_file():
            continue
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        executable = bool(path.stat().st_mode & stat.S_IXUSR)
        entries[relative.as_posix()] = (encoded, executable)
    return entries


def _restore_workspace_entries(
    root: Path,
    git_reference: Path,
    expected: Mapping[str, tuple[str, bool]],
) -> None:
    current = _workspace_entries(root, git_reference)
    changed = {path for path in current.keys() | expected.keys() if current.get(path) != expected.get(path)}
    for raw_path in sorted(changed, key=lambda value: (len(PurePosixPath(value).parts), value), reverse=True):
        path = root.joinpath(*PurePosixPath(raw_path).parts)
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        else:
            path.unlink(missing_ok=True)
    for raw_path in sorted(expected):
        if raw_path not in changed:
            continue
        encoded, executable = expected[raw_path]
        path = root.joinpath(*PurePosixPath(raw_path).parts)
        if path.is_dir() and not path.is_symlink():
            shutil.rmtree(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(base64.b64decode(encoded, validate=True))
        path.chmod(0o755 if executable else 0o644)


def _exclude_provider_assets(worktree: Path) -> None:
    exclude_path = Path(_git(worktree, "rev-parse", "--git-path", "info/exclude"))
    if not exclude_path.is_absolute():
        exclude_path = worktree / exclude_path
    existing = exclude_path.read_text(encoding="utf-8") if exclude_path.exists() else ""
    pattern = ".autosaddler/"
    if pattern not in existing.splitlines():
        exclude_path.parent.mkdir(parents=True, exist_ok=True)
        with exclude_path.open("a", encoding="utf-8") as destination:
            if existing and not existing.endswith("\n"):
                destination.write("\n")
            destination.write(pattern + "\n")


def _atomic_write_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as destination:
            destination.write(content)
            destination.flush()
            os.fsync(destination.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _tree_identity(repo: Path, revision: str) -> str:
    return sha256_digest(_git_bytes(repo, "ls-tree", "-r", "--full-tree", "-z", revision))


def _git(path: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments],
        cwd=path,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()


def _git_bytes(path: Path, *arguments: str) -> bytes:
    return subprocess.run(
        ["git", *arguments],
        cwd=path,
        capture_output=True,
        check=True,
    ).stdout


def _git_path_exists(repo: Path, revision: str, path: PurePosixPath) -> bool:
    listed = _git_bytes(repo, "ls-tree", "-z", "--name-only", revision, "--", path.as_posix())
    return listed == path.as_posix().encode("utf-8") + b"\0"


def _git_worktree_add(repo: Path, destination: Path, revision: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "worktree", "add", "--detach", str(destination), revision],
        cwd=repo,
        text=True,
        capture_output=True,
        check=True,
    )


def _read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")
