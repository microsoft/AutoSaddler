from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_SCENARIO_REPO_ID = "meta-agents-research-environments/gaia2"
SCENARIO_PROVISIONING_SCHEMA_VERSION = "autosaddler-meta-are-scenarios/v1"
_FULL_GIT_COMMIT = re.compile(r"[0-9a-fA-F]{40}")
_SAFE_NAME = re.compile(r"[A-Za-z0-9_-]+")
_SOURCE_DESCRIPTOR = ".autosaddler-gaia2-source.json"


class ScenarioProvisioningError(RuntimeError):
    pass


DatasetLoader = Callable[..., Iterable[Mapping[str, Any]]]


@dataclass(frozen=True, slots=True)
class ScenarioProvisioningRequest:
    destination_root: Path
    source_revision: str
    manifests: tuple[Path, ...]
    repo_id: str = DEFAULT_SCENARIO_REPO_ID

    def __post_init__(self) -> None:
        destination_root = Path(self.destination_root).expanduser()
        if not destination_root.is_absolute():
            raise ValueError("destination_root must be absolute")
        if destination_root.is_symlink():
            raise ValueError("destination_root must not be a symlink")
        if _FULL_GIT_COMMIT.fullmatch(self.source_revision) is None:
            raise ValueError("source_revision must be a full 40-character Git commit")
        if not self.repo_id.strip():
            raise ValueError("repo_id must be non-empty")
        if not self.manifests:
            raise ValueError("at least one split manifest is required")
        manifests = tuple(Path(path).expanduser().resolve(strict=True) for path in self.manifests)
        if any(not path.is_file() for path in manifests):
            raise ValueError("every split manifest must be a file")
        object.__setattr__(self, "destination_root", destination_root.resolve(strict=False))
        object.__setattr__(self, "source_revision", self.source_revision.lower())
        object.__setattr__(self, "manifests", manifests)


@dataclass(frozen=True, slots=True)
class ScenarioProvisioningResult:
    destination_root: Path
    source_descriptor: Path
    source_revision: str
    file_count: int
    reused_count: int


def provision_gaia2_scenarios(
    request: ScenarioProvisioningRequest,
    *,
    loader: DatasetLoader | None = None,
) -> ScenarioProvisioningResult:
    if request.destination_root.exists():
        if not request.destination_root.is_dir():
            raise ScenarioProvisioningError("destination_root must be a directory")
        _reject_destination_symlinks(request.destination_root)
    requested = _load_requested_cases(request.manifests)
    existing_descriptor = request.destination_root / _SOURCE_DESCRIPTOR
    if existing_descriptor.exists():
        descriptor = _read_json_object(existing_descriptor)
        if descriptor.get("repo_id") != request.repo_id or descriptor.get("source_revision") != request.source_revision:
            raise ScenarioProvisioningError(
                f"existing source descriptor does not match requested source: {existing_descriptor}"
            )

    if loader is None:
        from datasets import load_dataset

        loader = load_dataset

    selected: dict[str, tuple[Path, bytes]] = {}
    for capability, case_ids in sorted(requested.items()):
        rows = loader(
            request.repo_id,
            name=capability,
            split="validation",
            revision=request.source_revision,
        )
        wanted = set(case_ids)
        for index, row in enumerate(rows):
            scenario_id = row.get("scenario_id")
            if scenario_id not in wanted:
                continue
            if scenario_id in selected:
                raise ScenarioProvisioningError(f"dataset contains duplicate scenario ID: {scenario_id}")
            data = row.get("data")
            if not isinstance(data, str):
                raise ScenarioProvisioningError(f"scenario {scenario_id} has a non-text data payload")
            relative_path = Path(capability) / "validation" / f"{index:04d}_{scenario_id}.json"
            selected[str(scenario_id)] = (relative_path, data.encode("utf-8"))

    expected_ids = {case_id for case_ids in requested.values() for case_id in case_ids}
    missing = sorted(expected_ids - selected.keys())
    if missing:
        raise ScenarioProvisioningError(f"requested scenarios were not found at the pinned revision: {missing}")

    request.destination_root.mkdir(parents=True, exist_ok=True)
    _reject_destination_symlinks(request.destination_root)
    reused_count = 0
    records = []
    for scenario_id in sorted(selected):
        relative_path, payload = selected[scenario_id]
        destination = request.destination_root / relative_path
        matches = tuple(
            path
            for path in request.destination_root.rglob("*.json")
            if path.stem == scenario_id or path.stem.endswith(f"_{scenario_id}")
        )
        unexpected_matches = [path for path in matches if path != destination]
        if unexpected_matches:
            raise ScenarioProvisioningError(
                f"scenario {scenario_id} already exists at an unexpected path: {unexpected_matches}"
            )
        if destination.exists():
            if not destination.is_file() or destination.read_bytes() != payload:
                raise ScenarioProvisioningError(f"existing scenario payload differs from the pinned source: {destination}")
            reused_count += 1
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            _atomic_write(destination, payload)
        records.append(
            {
                "scenario_id": scenario_id,
                "path": relative_path.as_posix(),
                "bytes": len(payload),
                "sha256": _sha256(payload),
            }
        )

    descriptor = {
        "schema_version": SCENARIO_PROVISIONING_SCHEMA_VERSION,
        "repo_id": request.repo_id,
        "source_revision": request.source_revision,
        "split": "validation",
        "manifests": [
            {"path": str(path), "sha256": _sha256(path.read_bytes())}
            for path in request.manifests
        ],
        "files": records,
    }
    _atomic_write(existing_descriptor, _canonical_json(descriptor))
    return ScenarioProvisioningResult(
        destination_root=request.destination_root,
        source_descriptor=existing_descriptor,
        source_revision=request.source_revision,
        file_count=len(records),
        reused_count=reused_count,
    )


def _load_requested_cases(manifests: Sequence[Path]) -> dict[str, tuple[str, ...]]:
    grouped: dict[str, list[str]] = {}
    seen: set[str] = set()
    for manifest_path in manifests:
        manifest = _read_json_object(manifest_path)
        for capability, raw_case_ids in manifest.items():
            if _SAFE_NAME.fullmatch(capability) is None:
                raise ScenarioProvisioningError(f"unsafe capability name in {manifest_path}: {capability!r}")
            if not isinstance(raw_case_ids, list) or not raw_case_ids:
                raise ScenarioProvisioningError(f"capability {capability!r} must contain scenario IDs")
            case_ids = grouped.setdefault(capability, [])
            for raw_case_id in raw_case_ids:
                if not isinstance(raw_case_id, str) or _SAFE_NAME.fullmatch(raw_case_id) is None:
                    raise ScenarioProvisioningError(f"invalid scenario ID in {manifest_path}: {raw_case_id!r}")
                if raw_case_id in seen:
                    raise ScenarioProvisioningError(f"duplicate scenario ID across manifests: {raw_case_id}")
                seen.add(raw_case_id)
                case_ids.append(raw_case_id)
    if not grouped:
        raise ScenarioProvisioningError("split manifests contain no scenarios")
    return {capability: tuple(case_ids) for capability, case_ids in grouped.items()}


def _reject_destination_symlinks(destination_root: Path) -> None:
    if destination_root.is_symlink():
        raise ScenarioProvisioningError("destination_root must not be a symlink")
    symlinks = sorted(path.relative_to(destination_root) for path in destination_root.rglob("*") if path.is_symlink())
    if symlinks:
        raise ScenarioProvisioningError(f"destination tree contains symlinks: {[path.as_posix() for path in symlinks]}")


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ScenarioProvisioningError(f"cannot read JSON object: {path}") from error
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ScenarioProvisioningError(f"expected a JSON object with string keys: {path}")
    return value


def _atomic_write(path: Path, payload: bytes) -> None:
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _canonical_json(value: Mapping[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _sha256(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()