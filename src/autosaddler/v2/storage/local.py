from __future__ import annotations

import json
import os
import shutil
from collections.abc import Callable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from autosaddler.v2.core.domain import ArtifactRef, JsonValue, canonical_json, sha256_digest, to_json_value
from autosaddler.v2.core.events import RunEvent, operation_id
from autosaddler.v2.core.metrics import metrics_records, summarize_metrics

TransitionHook = Callable[[RunEvent], None]


class LocalRunStore:
    """Portable append-only event storage with disposable JSON projections."""

    def __init__(self, *, run_dir: Path, run_id: str, transition_hook: TransitionHook | None = None) -> None:
        if not run_id:
            raise ValueError("run_id must be non-empty")
        self.run_dir = run_dir.resolve()
        self.run_id = run_id
        self.transition_hook = transition_hook
        self.transition_hook_error: BaseException | None = None
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.events_path = self.run_dir / "events.jsonl"

    def initialize(
        self,
        *,
        resolved_config: Mapping[str, JsonValue],
        resolved_entities: Mapping[str, str | Mapping[str, JsonValue]],
    ) -> None:
        for directory in (
            "resolved/sources",
            "resolved/prompts",
            "resolved/schemas",
            "evaluations",
            "sessions",
            "evidence",
            "artifacts/train",
            "quarantine/dev",
            "strategy",
        ):
            (self.run_dir / directory).mkdir(parents=True, exist_ok=True)
        config_text = yaml.safe_dump(to_json_value(resolved_config), sort_keys=True)
        self._write_once_or_verify(Path("resolved_config.yaml"), config_text)
        for relative_path, value in resolved_entities.items():
            text = value if isinstance(value, str) else canonical_json(value) + "\n"
            self._write_once_or_verify(Path(relative_path), text)
        self.refresh_projections()

    def fork_from(self, source: "LocalRunStore", *, through_sequence: int) -> RunEvent:
        if self.run_id == source.run_id:
            raise ValueError("Fork target must use a new run_id")
        if self.events():
            raise ValueError("Fork target must not contain events")
        source.validate_integrity()
        source_events = source.events()
        if through_sequence < 1 or through_sequence > len(source_events):
            raise ValueError("Fork sequence must identify an existing source event")
        prefix = source_events[:through_sequence]
        if prefix[0].event_type != "RunStarted":
            raise ValueError("Fork source prefix must begin with RunStarted")
        terminal_types = {"OptimizationCompleted", "RunCompleted", "RunFailed"}
        if any(event.event_type in terminal_types for event in prefix):
            raise ValueError("Cannot fork from a terminal run prefix")
        self._validate_fork_compatibility(source)

        for uri in sorted(_artifact_uris(prefix)):
            _copy_artifact(source._path(uri), self._path(uri))

        source_prefix = "".join(canonical_json(event) + "\n" for event in prefix)
        for event in prefix:
            self.append(
                event.event_type,
                _rebase_run_string(event.operation_id, source.run_id, self.run_id),
                _rebase_run_value(event.payload, source.run_id, self.run_id),
            )
        return self.append(
            "RunForked",
            operation_id(self.run_id, "fork", source.run_id, through_sequence),
            {
                "source_run_id": source.run_id,
                "source_sequence": through_sequence,
                "source_prefix_sha256": sha256_digest(source_prefix.encode("utf-8")),
            },
        )

    def append(
        self,
        event_type: str,
        operation_id: str,
        payload: Mapping[str, JsonValue],
    ) -> RunEvent:
        self.transition_hook_error = None
        events = self.events()
        normalized = to_json_value(payload)
        assert isinstance(normalized, dict)
        for event in events:
            if event.event_type == event_type and event.operation_id == operation_id:
                if dict(event.payload) != normalized:
                    raise RuntimeError(
                        f"Idempotency conflict for {event_type} operation {operation_id}: payload changed"
                    )
                return event
        event = RunEvent.create(
            sequence=len(events) + 1,
            run_id=self.run_id,
            event_type=event_type,
            operation_id=operation_id,
            payload=normalized,
        )
        with self.events_path.open("a", encoding="utf-8") as destination:
            destination.write(canonical_json(event) + "\n")
            destination.flush()
            os.fsync(destination.fileno())
        self.refresh_projections()
        if self.transition_hook is not None:
            try:
                self.transition_hook(event)
            except BaseException as error:
                self.transition_hook_error = error
                raise
        return event

    def events(self) -> tuple[RunEvent, ...]:
        if not self.events_path.exists():
            return ()
        events: list[RunEvent] = []
        for line_number, line in enumerate(self.events_path.read_text(encoding="utf-8").splitlines(), start=1):
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid event JSON on line {line_number}") from exc
            if not isinstance(raw, dict):
                raise TypeError(f"Event line {line_number} must be an object")
            event = RunEvent.from_mapping(raw)
            if event.sequence != line_number:
                raise ValueError(f"Event sequence gap at line {line_number}: found {event.sequence}")
            if event.run_id != self.run_id:
                raise ValueError(f"Event run ID mismatch at line {line_number}")
            events.append(event)
        return tuple(events)

    def find(self, event_type: str, operation_id: str) -> RunEvent | None:
        return next(
            (event for event in self.events() if event.event_type == event_type and event.operation_id == operation_id),
            None,
        )

    def events_of_type(self, event_type: str) -> tuple[RunEvent, ...]:
        return tuple(event for event in self.events() if event.event_type == event_type)

    def write_json(self, relative_path: str, value: Any, *, kind: str) -> ArtifactRef:
        payload = canonical_json(value) + "\n"
        path = self._path(relative_path)
        _atomic_write(path, payload)
        return ArtifactRef(
            uri=PurePosixPath(relative_path).as_posix(),
            kind=kind,
            sha256=sha256_digest(payload.encode("utf-8")),
            bytes=len(payload.encode("utf-8")),
        )

    def write_text(self, relative_path: str, text: str, *, kind: str) -> ArtifactRef:
        path = self._path(relative_path)
        _atomic_write(path, text)
        return ArtifactRef(
            uri=PurePosixPath(relative_path).as_posix(),
            kind=kind,
            sha256=sha256_digest(text.encode("utf-8")),
            bytes=len(text.encode("utf-8")),
        )

    def read_json(self, relative_path: str) -> object:
        return json.loads(self._path(relative_path).read_text(encoding="utf-8"))

    def refresh_projections(self) -> None:
        events = self.events()
        status = _run_status(events)
        snapshot = {
            "schema_version": "autosaddler-run-snapshot/v1",
            "run_id": self.run_id,
            "status": status,
            "last_sequence": events[-1].sequence if events else 0,
            "completed_operations": sorted(
                {
                    event.operation_id
                    for event in events
                    if event.event_type
                    in {
                        "SessionCompleted",
                        "EvaluationCompleted",
                        "EvaluationAttemptCompleted",
                        "DeferredWorkCompleted",
                        "IterationCompleted",
                    }
                }
            ),
        }
        _atomic_write(self.run_dir / "snapshot.json", canonical_json(snapshot) + "\n")
        _atomic_write(self.run_dir / "evolution_dag.json", canonical_json(_evolution_dag(events)) + "\n")
        _atomic_write(
            self.run_dir / "strategy" / "lessons.json",
            canonical_json(_extension_projection(events, "autosaddler.lessons")) + "\n",
        )
        metric_rows = metrics_records(events)
        _atomic_write(
            self.run_dir / "metrics.jsonl",
            "".join(canonical_json(row) + "\n" for row in metric_rows),
        )
        _atomic_write(
            self.run_dir / "metrics-summary.json",
            canonical_json(summarize_metrics(events)) + "\n",
        )
        resolved = _resolved_entities(self.run_dir)
        manifest = {
            "schema_version": "autosaddler-run-manifest/v1",
            "run_id": self.run_id,
            "status": status,
            "autosaddler_version": "0.2.0",
            "resolved_entities": resolved,
            "event_log": {
                "path": "events.jsonl",
                "last_sequence": events[-1].sequence if events else 0,
                "sha256": sha256_digest(self.events_path.read_bytes())
                if self.events_path.exists()
                else sha256_digest(b""),
            },
        }
        provider_runtime_path = self.run_dir / "resolved" / "provider_runtime.json"
        if provider_runtime_path.is_file():
            manifest["provider_runtime"] = json.loads(provider_runtime_path.read_text(encoding="utf-8"))
        _atomic_write(self.run_dir / "manifest.json", canonical_json(manifest) + "\n")

    def validate_integrity(self) -> None:
        manifest = self.read_json("manifest.json")
        if not isinstance(manifest, dict):
            raise TypeError("Run manifest must be an object")
        entities = manifest.get("resolved_entities")
        if not isinstance(entities, dict):
            raise TypeError("Run manifest resolved_entities must be an object")
        for name, raw in entities.items():
            if not isinstance(raw, dict) or not isinstance(raw.get("path"), str):
                raise TypeError(f"Malformed resolved entity: {name}")
            path = self._path(raw["path"])
            actual = sha256_digest(path.read_bytes())
            if actual != raw.get("sha256"):
                raise ValueError(f"Resolved entity checksum mismatch: {name}")
        provider_runtime_path = self.run_dir / "resolved" / "provider_runtime.json"
        if provider_runtime_path.is_file():
            provider_runtime = json.loads(provider_runtime_path.read_text(encoding="utf-8"))
            if manifest.get("provider_runtime") != provider_runtime:
                raise ValueError("Run manifest provider_runtime differs from its resolved entity")
        event_log = manifest.get("event_log")
        if not isinstance(event_log, dict):
            raise TypeError("Run manifest event_log must be an object")
        actual_events = (
            sha256_digest(self.events_path.read_bytes()) if self.events_path.exists() else sha256_digest(b"")
        )
        if actual_events != event_log.get("sha256"):
            raise ValueError("Event log checksum mismatch")
        self.events()

    def _validate_fork_compatibility(self, source: "LocalRunStore") -> None:
        source_config = yaml.safe_load(source._path("resolved_config.yaml").read_text(encoding="utf-8"))
        target_config = yaml.safe_load(self._path("resolved_config.yaml").read_text(encoding="utf-8"))
        if not isinstance(source_config, dict) or not isinstance(target_config, dict):
            raise TypeError("Resolved fork configs must be mappings")
        source_budget = source_config.get("optimization", {}).get("budget", {})
        target_budget = target_config.get("optimization", {}).get("budget", {})
        if not isinstance(source_budget, dict) or not isinstance(target_budget, dict):
            raise TypeError("Resolved fork budgets must be mappings")
        source_budget["max_iterations"] = target_budget.get("max_iterations")
        if source_config != target_config:
            raise ValueError("Fork target config may differ only in optimization.budget.max_iterations")

        source_manifest = source.read_json("manifest.json")
        target_manifest = self.read_json("manifest.json")
        assert isinstance(source_manifest, dict) and isinstance(target_manifest, dict)
        source_entities = source_manifest.get("resolved_entities")
        target_entities = target_manifest.get("resolved_entities")
        if not isinstance(source_entities, dict) or not isinstance(target_entities, dict):
            raise TypeError("Fork manifests must contain resolved entities")
        if source_entities.keys() != target_entities.keys():
            raise ValueError("Fork target resolved entities differ from source")
        for name in source_entities:
            if name in {"resolved_config.yaml", "resolved/policies.json"}:
                continue
            if source_entities[name] != target_entities[name]:
                raise ValueError(f"Fork target resolved entity differs from source: {name}")

    def _write_once_or_verify(self, relative_path: Path, text: str) -> None:
        path = self._path(relative_path.as_posix())
        if path.exists():
            if path.read_text(encoding="utf-8") != text:
                raise ValueError(f"Resolved run input changed on resume: {relative_path.as_posix()}")
            return
        _atomic_write(path, text)

    def _path(self, relative_path: str) -> Path:
        path = PurePosixPath(relative_path)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"Run artifact path must be relative: {relative_path}")
        resolved = self.run_dir.joinpath(*path.parts).resolve()
        if not resolved.is_relative_to(self.run_dir):
            raise ValueError(f"Run artifact escapes its root: {relative_path}")
        return resolved


def _run_status(events: tuple[RunEvent, ...]) -> str:
    active_events = events
    for index in range(len(events) - 1, -1, -1):
        if events[index].event_type == "RunResumed":
            active_events = events[index:]
            break
    types = {event.event_type for event in active_events}
    for event in reversed(active_events):
        if event.event_type == "RunCompleted":
            return "completed"
        if event.event_type == "RunFailed":
            return "failed"
        if event.event_type == "RunInterrupted":
            return "optimization_interrupted"
    if "TestEvaluationStarted" in types and "TestEvaluationCompleted" not in types:
        return "test_interrupted"
    if "OptimizationCompleted" in types:
        return "optimization_completed"
    return "optimizing"


def _artifact_uris(events: tuple[RunEvent, ...]) -> set[str]:
    uris: set[str] = set()

    def visit(value: object) -> None:
        if isinstance(value, Mapping):
            uri = value.get("uri")
            if (
                isinstance(uri, str)
                and isinstance(value.get("kind"), str)
                and isinstance(value.get("sha256"), str)
                and isinstance(value.get("bytes"), int)
            ):
                path = PurePosixPath(uri)
                if path.is_absolute() or ".." in path.parts:
                    raise ValueError(f"Fork artifact path must be relative: {uri}")
                uris.add(path.as_posix())
            for child in value.values():
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)

    for event in events:
        visit(event.payload)
    return uris


def _copy_artifact(source: Path, target: Path) -> None:
    if source.is_symlink():
        raise ValueError(f"Fork artifact cannot be a symlink: {source}")
    if source.is_file():
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists():
            if not target.is_file() or target.read_bytes() != source.read_bytes():
                raise ValueError(f"Fork target artifact conflicts with source: {target}")
            return
        shutil.copy2(source, target)
        return
    if source.is_dir():
        for child in source.rglob("*"):
            if child.is_symlink():
                raise ValueError(f"Fork artifact cannot contain a symlink: {child}")
            if child.is_file():
                _copy_artifact(child, target / child.relative_to(source))
        return
    raise FileNotFoundError(f"Fork source artifact does not exist: {source}")


def _rebase_run_string(value: str, source_run_id: str, target_run_id: str) -> str:
    prefix = f"{source_run_id}:"
    if not value.startswith(prefix):
        return value
    return f"{target_run_id}:{value[len(prefix) :]}"


def _rebase_run_value(value: object, source_run_id: str, target_run_id: str) -> JsonValue:
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _rebase_run_string(value, source_run_id, target_run_id)
    if isinstance(value, Mapping):
        return {str(key): _rebase_run_value(child, source_run_id, target_run_id) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_rebase_run_value(child, source_run_id, target_run_id) for child in value]
    raise TypeError(f"Unsupported fork payload value: {type(value).__name__}")


def _evolution_dag(events: tuple[RunEvent, ...]) -> dict[str, JsonValue]:
    nodes: dict[str, dict[str, JsonValue]] = {}
    selected: str | None = None
    for event in events:
        if event.event_type == "RunStarted":
            seed = event.payload.get("seed_candidate")
            if isinstance(seed, dict) and isinstance(seed.get("candidate_id"), str):
                nodes[seed["candidate_id"]] = {
                    "candidate_id": seed["candidate_id"],
                    "parent_ids": seed.get("parent_ids", []),
                    "status": "evaluated",
                }
        elif event.event_type == "CandidateFinalized":
            candidate = event.payload.get("candidate")
            if isinstance(candidate, dict) and isinstance(candidate.get("candidate_id"), str):
                nodes[candidate["candidate_id"]] = {
                    "candidate_id": candidate["candidate_id"],
                    "parent_ids": candidate.get("parent_ids", []),
                    "status": "proposed",
                    "change": candidate.get("change"),
                }
        elif event.event_type == "EvolutionRecorded":
            candidate_id = event.payload.get("candidate_id")
            if isinstance(candidate_id, str) and candidate_id in nodes:
                nodes[candidate_id]["status"] = "accepted" if event.payload.get("accepted") else "declined"
                nodes[candidate_id]["development_aggregate"] = event.payload.get("development_aggregate")
                nodes[candidate_id]["iteration"] = event.payload.get("iteration")
        elif event.event_type == "BestCandidateSelected":
            value = event.payload.get("candidate_id")
            if isinstance(value, str):
                selected = value
    if selected in nodes:
        nodes[selected]["status"] = "selected"
    edges: list[JsonValue] = []
    for node in nodes.values():
        for parent_id in node.get("parent_ids", []):
            edges.append({"parent_id": parent_id, "child_id": node["candidate_id"]})
    return {
        "schema_version": "autosaddler-evolution-dag/v1",
        "nodes": [nodes[key] for key in sorted(nodes)],
        "edges": edges,
    }


def _resolved_entities(run_dir: Path) -> dict[str, JsonValue]:
    paths = [run_dir / "resolved_config.yaml"]
    resolved_root = run_dir / "resolved"
    if resolved_root.exists():
        paths.extend(sorted(path for path in resolved_root.rglob("*") if path.is_file()))
    entities: dict[str, JsonValue] = {}
    for path in paths:
        if not path.is_file():
            continue
        relative = path.relative_to(run_dir).as_posix()
        entities[relative] = {"path": relative, "sha256": sha256_digest(path.read_bytes())}
    return entities


def _extension_projection(events: tuple[RunEvent, ...], namespace: str) -> dict[str, JsonValue]:
    changes = [
        dict(event.payload)
        for event in events
        if event.event_type == "ExtensionStateChanged" and event.payload.get("namespace") == namespace
    ]
    return {
        "schema_version": "autosaddler-extension-projection/v1",
        "namespace": namespace,
        "changes": changes,
    }


def _atomic_write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as destination:
        destination.write(text)
        destination.flush()
        os.fsync(destination.fileno())
    temporary.replace(path)
