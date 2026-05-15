from __future__ import annotations

import difflib
import json
import shutil
from collections.abc import Callable, Mapping
from pathlib import Path

from autosaddler.v2.core.domain import (
    ArtifactRef,
    Candidate,
    ChangeSummary,
    JsonValue,
    canonical_json,
    sha256_digest,
)
from autosaddler.v2.core.ports import (
    CompositionPlan,
    MaterializedHarness,
    MutationContext,
    MutationOutcome,
    MutationSession,
    WorkspaceDelta,
)

ComponentValidator = Callable[[Mapping[str, str]], None]


class ComponentMapHarnessSpace:
    """A frozen-key text candidate space backed by immutable canonical JSON."""

    def __init__(
        self,
        *,
        baseline: Mapping[str, str],
        store_root: Path,
        materialization_root: Path | None = None,
        validator: ComponentValidator | None = None,
    ) -> None:
        if not baseline:
            raise ValueError("Component-map baseline cannot be empty")
        self._baseline = _validated_components(baseline)
        self._schema = frozenset(self._baseline)
        self.store_root = store_root.resolve()
        self.materialization_root = (materialization_root or store_root.parent / "materialized").resolve()
        self.validator = validator
        self._sessions: dict[str, tuple[str, ...]] = {}
        self._validate(self._baseline)

    def seed(self) -> Candidate:
        return self._publish(self._baseline, parent_ids=(), change=None)

    def begin_mutation(self, parent: Candidate, context: MutationContext) -> MutationSession:
        parent_components = self._load(parent.candidate_id)
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
        workspace.mkdir(parents=True, exist_ok=True)
        candidate_path = workspace / "candidate.json"
        if candidate_path.exists():
            existing = json.loads(candidate_path.read_text(encoding="utf-8"))
            if existing != parent_components:
                raise RuntimeError(f"Mutation workspace parent drifted: {workspace}")
        else:
            _write_json(candidate_path, parent_components)
        self._sessions[session_id] = (parent.candidate_id,)
        return MutationSession(
            session_id=session_id,
            parent=parent,
            workspace=workspace,
            context=context,
            output_contract=ArtifactRef(
                uri=f"sessions/{session_id.removeprefix('sha256:')}/candidate_updates.json",
                kind="component-updates",
            ),
        )

    def capture_attempt_delta(
        self,
        session: MutationSession,
        attempt_workspace: Path,
        *,
        attempt_operation_id: str,
    ) -> WorkspaceDelta | None:
        del session, attempt_workspace, attempt_operation_id
        return None

    def apply_mutation(self, session: MutationSession, outcome: MutationOutcome) -> None:
        output = outcome.result.structured_output
        if not isinstance(output, Mapping):
            raise TypeError("Component mutation output must be an object")
        updates = output.get("updates")
        if not isinstance(updates, Mapping):
            raise TypeError("Component mutation output 'updates' must be an object")
        self.apply_updates(session, updates)

    def apply_updates(self, session: MutationSession, updates: Mapping[str, JsonValue]) -> None:
        normalized = _validated_updates(updates, self._schema)
        _write_json(session.workspace / "candidate_updates.json", normalized)

    def finalize(self, session: MutationSession) -> Candidate:
        updates_path = session.workspace / "candidate_updates.json"
        if not updates_path.is_file():
            raise FileNotFoundError(f"Mutation output is missing: {updates_path}")
        updates_value = json.loads(updates_path.read_text(encoding="utf-8"))
        if not isinstance(updates_value, dict):
            raise TypeError("Component updates must be a JSON object")
        updates = _validated_updates(updates_value, self._schema)
        parent_components = self._load(session.parent.candidate_id)
        child_components = {**parent_components, **updates}
        self._validate(child_components)
        if child_components == parent_components:
            raise ValueError("Candidate mutation must change at least one component")
        change = _component_diff(parent_components, child_components)
        parent_ids = self._sessions.get(session.session_id)
        if parent_ids is None:
            raise RuntimeError(f"Unknown mutation session: {session.session_id}")
        return self._publish(child_components, parent_ids=parent_ids, change=change)

    def compose(self, plan: CompositionPlan) -> Candidate:
        if not plan.parents:
            raise ValueError("Composition requires at least one parent")
        if len(set(plan.parents)) != len(plan.parents):
            raise ValueError("Composition parents must be unique")
        parent_values = {candidate_id: self._load(candidate_id) for candidate_id in plan.parents}
        first = parent_values[plan.parents[0]]
        child = dict(first)
        for component, source_id in plan.selections.items():
            if component not in self._schema:
                raise ValueError(f"Unknown component in composition: {component}")
            if source_id not in parent_values:
                raise ValueError(f"Composition source is not a declared parent: {source_id}")
            child[component] = parent_values[source_id][component]
        child.update(_validated_updates(plan.overrides, self._schema))
        self._validate(child)
        if child == first:
            raise ValueError("Composition must change at least one component")
        change = _component_diff(first, child, labels=("composition",))
        return self._publish(child, parent_ids=plan.parents, change=change)

    def materialize(self, candidate: Candidate, purpose: str) -> MaterializedHarness:
        if purpose not in {"evaluate", "inspect", "mutate"}:
            raise ValueError(f"Unknown materialization purpose: {purpose}")
        components = self._load(candidate.candidate_id)
        root = self.materialization_root / f"{purpose}-{candidate.candidate_id.removeprefix('sha256:')}"
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True)
        _write_json(root / "candidate.json", components)
        (root / "candidate.json").chmod(0o444)
        return MaterializedHarness(
            root=root,
            candidate_id=candidate.candidate_id,
            release=lambda: shutil.rmtree(root, ignore_errors=True),
        )

    def diff(self, parent: Candidate, child: Candidate) -> ChangeSummary:
        return _component_diff(self._load(parent.candidate_id), self._load(child.candidate_id))

    def _validate(self, value: Mapping[str, str]) -> None:
        if frozenset(value) != self._schema:
            missing = sorted(self._schema - set(value))
            extra = sorted(set(value) - self._schema)
            raise ValueError(f"Candidate component schema changed; missing={missing}, extra={extra}")
        _validated_components(value)
        if self.validator is not None:
            self.validator(value)

    def _publish(
        self,
        components: Mapping[str, str],
        *,
        parent_ids: tuple[str, ...],
        change: ChangeSummary | None,
    ) -> Candidate:
        payload = canonical_json(components) + "\n"
        candidate_id = sha256_digest(payload.rstrip("\n"))
        digest = candidate_id.removeprefix("sha256:")
        candidate_dir = self.store_root / digest
        candidate_path = candidate_dir / "candidate.json"
        if candidate_path.exists():
            if candidate_path.read_text(encoding="utf-8") != payload:
                raise RuntimeError(f"Candidate content collision: {candidate_id}")
        else:
            candidate_dir.mkdir(parents=True, exist_ok=False)
            candidate_path.write_text(payload, encoding="utf-8")
            candidate_path.chmod(0o444)
        return Candidate(
            candidate_id=candidate_id,
            parent_ids=parent_ids,
            space="component-map",
            artifact=ArtifactRef(
                uri=f"candidates/{digest}/candidate.json",
                kind="component-map",
                sha256=candidate_id,
                bytes=len(payload.encode("utf-8")),
            ),
            change=change,
        )

    def _load(self, candidate_id: str) -> dict[str, str]:
        path = self.store_root / candidate_id.removeprefix("sha256:") / "candidate.json"
        if not path.is_file():
            raise FileNotFoundError(f"Unknown component candidate: {candidate_id}")
        payload = path.read_text(encoding="utf-8")
        value = json.loads(payload)
        if not isinstance(value, dict):
            raise TypeError(f"Component candidate is not a JSON object: {path}")
        components = _validated_components(value)
        self._validate(components)
        if sha256_digest(canonical_json(components)) != candidate_id:
            raise RuntimeError(f"Component candidate digest mismatch: {candidate_id}")
        return components


def _validated_components(value: Mapping[str, object]) -> dict[str, str]:
    components: dict[str, str] = {}
    for component, text in value.items():
        if not isinstance(component, str) or not component:
            raise ValueError("Component IDs must be non-empty strings")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Component text must be non-empty: {component}")
        components[component] = text
    return components


def _validated_updates(value: Mapping[str, object], schema: frozenset[str]) -> dict[str, str]:
    unknown = sorted(set(value) - schema)
    if unknown:
        raise ValueError(f"Mutation contains components outside the frozen schema: {unknown}")
    return _validated_components(value)


def _component_diff(
    parent: Mapping[str, str],
    child: Mapping[str, str],
    *,
    labels: tuple[str, ...] = (),
) -> ChangeSummary:
    changed = tuple(sorted(component for component in parent if parent[component] != child[component]))
    added = 0
    removed = 0
    for component in changed:
        for line in difflib.ndiff(parent[component].splitlines(), child[component].splitlines()):
            added += line.startswith("+ ")
            removed += line.startswith("- ")
    return ChangeSummary(changed_units=changed, added=added, removed=removed, labels=labels)


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(value) + "\n", encoding="utf-8")