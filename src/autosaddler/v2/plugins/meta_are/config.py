from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from autosaddler.v2.core.domain import Case, JsonValue, canonical_json, sha256_digest
from autosaddler.v2.plugins.meta_are.responses_runtime import RESPONSES_RUNTIME_PATH
from autosaddler.v2.plugins.meta_are.scenario_provisioning import (
    DEFAULT_SCENARIO_REPO_ID,
    SCENARIO_PROVISIONING_SCHEMA_VERSION,
)

_FULL_COMMIT = re.compile(r"[0-9a-fA-F]{40}")
APPROVED_WRITABLE_PATHS = (
    PurePosixPath("are/simulation/agents/default_agent"),
    PurePosixPath("are/simulation/agents/are_simulation_agent_config.py"),
    PurePosixPath("are/simulation/agents/agent_config_builder.py"),
    PurePosixPath("are/simulation/apps/agent_user_interface.py"),
    PurePosixPath("hook.json"),
)
APPROVED_FORBIDDEN_PATHS = (
    PurePosixPath("are/simulation/validation"),
    PurePosixPath("are/simulation/scenarios"),
    PurePosixPath("are/simulation/benchmark"),
    PurePosixPath("are/simulation/benchmark.py"),
    PurePosixPath("are/simulation/tests"),
    PurePosixPath("are/simulation/data"),
    PurePosixPath("are/simulation/data_handler"),
    PurePosixPath("are/simulation/checkpoint"),
    PurePosixPath("are/simulation/tutorials"),
    PurePosixPath("are/simulation/gui"),
)
_EXPECTED_KEYS = {
    "source_repo",
    "base_revision",
    "data_mode",
    "dataset_root",
    "dataset_source_revision",
    "dataset_source_descriptor",
    "train_manifest",
    "development_manifest",
    "test_manifest",
    "writable_paths",
    "forbidden_paths",
    "demo_filesystem_root",
    "demo_filesystem_source_revision",
    "demo_filesystem_source_descriptor",
    "demo_filesystem_manifest",
    "benchmark_config",
    "benchmark_split",
    "agent",
    "model",
    "model_provider",
    "model_wire_api",
    "model_endpoint",
    "reasoning_effort",
    "judge_model",
    "judge_provider",
    "judge_endpoint",
    "scenario_timeout_seconds",
    "process_completion_grace_seconds",
    "verification_timeout_seconds",
    "repetitions",
    "max_concurrent",
    "infrastructure_retries",
    "import_check",
    "capability_phase_iterations",
}


@dataclass(frozen=True, slots=True)
class MetaARESettings:
    source_repo: Path
    base_revision: str
    data_mode: str
    dataset_root: Path
    dataset_source_revision: str
    dataset_source_descriptor: Path
    dataset_digest: str
    train_manifest: Path
    development_manifest: Path
    test_manifest: Path
    test_case_count: int
    writable_paths: tuple[PurePosixPath, ...]
    forbidden_paths: tuple[PurePosixPath, ...]
    demo_filesystem_root: Path
    demo_filesystem_source_revision: str
    demo_filesystem_source_descriptor: Path
    demo_filesystem_manifest: Path
    benchmark_config: str
    benchmark_split: str
    agent: str
    model: str
    model_provider: str
    model_wire_api: str
    model_endpoint: str | None
    reasoning_effort: str | None
    judge_model: str
    judge_provider: str
    judge_endpoint: str | None
    scenario_timeout_seconds: int
    process_completion_grace_seconds: int
    verification_timeout_seconds: int
    repetitions: int
    max_concurrent: int
    infrastructure_retries: int
    import_check: str
    capability_phase_iterations: int
    pyproject_sha256: str
    uv_lock_sha256: str
    responses_runtime_sha256: str | None
    demo_filesystem_digest: str
    execution_fingerprint: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, JsonValue], *, base_dir: Path) -> "MetaARESettings":
        missing = sorted(_EXPECTED_KEYS - value.keys())
        extra = sorted(value.keys() - _EXPECTED_KEYS)
        if missing or extra:
            raise ValueError(f"Invalid Meta-ARE settings keys: missing={missing}, unexpected={extra}")

        source_repo = _directory(value["source_repo"], base_dir, "source_repo")
        base_revision = _full_commit(value["base_revision"], "base_revision")
        _require_git_commit(source_repo, base_revision)
        pyproject = _git_file(source_repo, base_revision, "pyproject.toml")
        uv_lock = _git_file(source_repo, base_revision, "uv.lock")

        data_mode = _string(value["data_mode"], "data_mode")
        if data_mode != "local_only":
            raise ValueError("Meta-ARE data_mode must be local_only")
        dataset_root = _directory(value["dataset_root"], base_dir, "dataset_root")
        dataset_revision = _full_commit(value["dataset_source_revision"], "dataset_source_revision")
        dataset_descriptor = _file(
            value["dataset_source_descriptor"],
            base_dir,
            "dataset_source_descriptor",
        )
        train_manifest = _file(value["train_manifest"], base_dir, "train_manifest")
        development_manifest = _file(value["development_manifest"], base_dir, "development_manifest")
        test_manifest = _file(value["test_manifest"], base_dir, "test_manifest")
        train_ids = _manifest_case_ids(train_manifest)
        development_ids = _manifest_case_ids(development_manifest)
        test_ids = _manifest_case_ids(test_manifest)
        _reject_split_overlap(train_ids, development_ids, test_ids)
        test_case_count = len(test_ids)
        dataset_digest = _verify_dataset_source(
            root=dataset_root,
            revision=dataset_revision,
            descriptor_path=dataset_descriptor,
            manifests=(train_manifest, development_manifest),
        )
        writable_paths = _approved_paths(
            value["writable_paths"],
            "writable_paths",
            APPROVED_WRITABLE_PATHS,
        )
        forbidden_paths = _approved_paths(
            value["forbidden_paths"],
            "forbidden_paths",
            APPROVED_FORBIDDEN_PATHS,
        )
        demo_root = _directory(value["demo_filesystem_root"], base_dir, "demo_filesystem_root")
        demo_revision = _full_commit(
            value["demo_filesystem_source_revision"],
            "demo_filesystem_source_revision",
        )
        demo_descriptor = _file(
            value["demo_filesystem_source_descriptor"],
            base_dir,
            "demo_filesystem_source_descriptor",
        )
        demo_manifest = _file(
            value["demo_filesystem_manifest"],
            base_dir,
            "demo_filesystem_manifest",
        )
        demo_digest = _verify_demo_filesystem(
            root=demo_root,
            revision=demo_revision,
            descriptor_path=demo_descriptor,
            manifest_path=demo_manifest,
        )
        model_provider = _string(value["model_provider"], "model_provider")
        model_wire_api = _string(value["model_wire_api"], "model_wire_api")
        if model_wire_api not in {"chat_completions", "responses"}:
            raise ValueError(
                "Meta-ARE model_wire_api must be chat_completions or responses"
            )
        if model_wire_api == "responses" and model_provider != "copilot":
            raise ValueError(
                "Meta-ARE Responses task transport is supported only for the copilot provider"
            )
        reasoning_effort = _optional_string(
            value["reasoning_effort"], "reasoning_effort"
        )
        if model_wire_api == "responses" and reasoning_effort is not None:
            raise ValueError(
                "Meta-ARE Responses task transport requires provider-default reasoning"
            )
        responses_runtime_sha256 = (
            sha256_digest(RESPONSES_RUNTIME_PATH.read_bytes())
            if model_wire_api == "responses"
            else None
        )

        resolved: dict[str, JsonValue] = {
            "base_revision": base_revision.lower(),
            "pyproject_sha256": sha256_digest(pyproject),
            "uv_lock_sha256": sha256_digest(uv_lock),
            "data_mode": data_mode,
            "dataset_root": str(dataset_root),
            "dataset_source_revision": dataset_revision.lower(),
            "dataset_digest": dataset_digest,
            "train_manifest_sha256": sha256_digest(train_manifest.read_bytes()),
            "development_manifest_sha256": sha256_digest(development_manifest.read_bytes()),
            "test_manifest_sha256": sha256_digest(test_manifest.read_bytes()),
            "test_case_count": test_case_count,
            "writable_paths": [path.as_posix() for path in writable_paths],
            "forbidden_paths": [path.as_posix() for path in forbidden_paths],
            "demo_filesystem_source_revision": demo_revision.lower(),
            "demo_filesystem_digest": demo_digest,
            "benchmark_config": _string(value["benchmark_config"], "benchmark_config"),
            "benchmark_split": _string(value["benchmark_split"], "benchmark_split"),
            "agent": _string(value["agent"], "agent"),
            "model": _string(value["model"], "model"),
            "model_provider": model_provider,
            "model_wire_api": model_wire_api,
            "model_endpoint": _optional_string(value["model_endpoint"], "model_endpoint"),
            "reasoning_effort": reasoning_effort,
            "responses_runtime_sha256": responses_runtime_sha256,
            "judge_model": _string(value["judge_model"], "judge_model"),
            "judge_provider": _string(value["judge_provider"], "judge_provider"),
            "judge_endpoint": _optional_string(value["judge_endpoint"], "judge_endpoint"),
            "scenario_timeout_seconds": _positive_int(
                value["scenario_timeout_seconds"], "scenario_timeout_seconds"
            ),
            "process_completion_grace_seconds": _nonnegative_int(
                value["process_completion_grace_seconds"],
                "process_completion_grace_seconds",
            ),
            "verification_timeout_seconds": _positive_int(
                value["verification_timeout_seconds"],
                "verification_timeout_seconds",
            ),
            "repetitions": _positive_int(value["repetitions"], "repetitions"),
            "max_concurrent": _positive_int(value["max_concurrent"], "max_concurrent"),
            "infrastructure_retries": _nonnegative_int(
                value["infrastructure_retries"], "infrastructure_retries"
            ),
            "import_check": _string(value["import_check"], "import_check"),
            "capability_phase_iterations": _nonnegative_int(
                value["capability_phase_iterations"], "capability_phase_iterations"
            ),
        }
        fingerprint = sha256_digest(canonical_json(resolved))
        return cls(
            source_repo=source_repo,
            base_revision=base_revision.lower(),
            data_mode=data_mode,
            dataset_root=dataset_root,
            dataset_source_revision=dataset_revision.lower(),
            dataset_source_descriptor=dataset_descriptor,
            dataset_digest=dataset_digest,
            train_manifest=train_manifest,
            development_manifest=development_manifest,
            test_manifest=test_manifest,
            test_case_count=test_case_count,
            writable_paths=writable_paths,
            forbidden_paths=forbidden_paths,
            demo_filesystem_root=demo_root,
            demo_filesystem_source_revision=demo_revision.lower(),
            demo_filesystem_source_descriptor=demo_descriptor,
            demo_filesystem_manifest=demo_manifest,
            benchmark_config=str(resolved["benchmark_config"]),
            benchmark_split=str(resolved["benchmark_split"]),
            agent=str(resolved["agent"]),
            model=str(resolved["model"]),
            model_provider=str(resolved["model_provider"]),
            model_wire_api=str(resolved["model_wire_api"]),
            model_endpoint=_as_optional_string(resolved["model_endpoint"]),
            reasoning_effort=_as_optional_string(resolved["reasoning_effort"]),
            judge_model=str(resolved["judge_model"]),
            judge_provider=str(resolved["judge_provider"]),
            judge_endpoint=_as_optional_string(resolved["judge_endpoint"]),
            scenario_timeout_seconds=int(resolved["scenario_timeout_seconds"]),
            process_completion_grace_seconds=int(resolved["process_completion_grace_seconds"]),
            verification_timeout_seconds=int(resolved["verification_timeout_seconds"]),
            repetitions=int(resolved["repetitions"]),
            max_concurrent=int(resolved["max_concurrent"]),
            infrastructure_retries=int(resolved["infrastructure_retries"]),
            import_check=str(resolved["import_check"]),
            capability_phase_iterations=int(resolved["capability_phase_iterations"]),
            pyproject_sha256=str(resolved["pyproject_sha256"]),
            uv_lock_sha256=str(resolved["uv_lock_sha256"]),
            responses_runtime_sha256=_as_optional_string(
                resolved["responses_runtime_sha256"]
            ),
            demo_filesystem_digest=demo_digest,
            execution_fingerprint=fingerprint,
        )

    def load_cases(self) -> tuple[tuple[Case, ...], tuple[Case, ...]]:
        train_ids = _manifest_case_ids(self.train_manifest)
        development_ids = _manifest_case_ids(self.development_manifest)
        test_ids = _manifest_case_ids(self.test_manifest)
        _reject_split_overlap(train_ids, development_ids, test_ids)
        return (
            self._cases(train_ids, "train"),
            self._cases(development_ids, "development"),
        )

    def _cases(self, case_ids: Sequence[str], split: str) -> tuple[Case, ...]:
        all_files = tuple(sorted(self.dataset_root.rglob("*.json")))
        cases = []
        for case_id in case_ids:
            matches = [
                path
                for path in all_files
                if path.stem == case_id or path.stem.endswith(f"_{case_id}")
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Meta-ARE case {case_id!r} must resolve to exactly one dataset file; found {len(matches)}"
                )
            source = matches[0]
            payload = source.read_bytes()
            cases.append(
                Case(
                    case_id=case_id,
                    split=split,  # type: ignore[arg-type]
                    payload={
                        "source_path": source.relative_to(self.dataset_root).as_posix(),
                        "source_sha256": sha256_digest(payload),
                        "source_bytes": len(payload),
                    },
                )
            )
        return tuple(cases)


def _manifest_case_ids(path: Path) -> tuple[str, ...]:
    value = _json_object_or_list(path)
    if isinstance(value, list):
        raw_ids = value
    else:
        raw_ids = []
        for group, entries in value.items():
            if not isinstance(group, str) or not isinstance(entries, list):
                raise TypeError(f"Meta-ARE split manifest groups must contain lists: {path}")
            raw_ids.extend(entries)
    if not raw_ids or any(not isinstance(case_id, str) or not case_id for case_id in raw_ids):
        raise ValueError(f"Meta-ARE split manifest must contain non-empty case IDs: {path}")
    case_ids = tuple(raw_ids)
    if len(set(case_ids)) != len(case_ids):
        raise ValueError(f"Meta-ARE split manifest contains duplicate case IDs: {path}")
    return case_ids


def _reject_split_overlap(
    train_ids: Sequence[str],
    development_ids: Sequence[str],
    test_ids: Sequence[str],
) -> None:
    overlap = sorted(
        (set(train_ids) & set(development_ids))
        | (set(train_ids) & set(test_ids))
        | (set(development_ids) & set(test_ids))
    )
    if overlap:
        raise ValueError(f"Meta-ARE train, development, and test splits must be disjoint; overlap={overlap}")


def _verify_dataset_source(
    *,
    root: Path,
    revision: str,
    descriptor_path: Path,
    manifests: tuple[Path, Path],
) -> str:
    descriptor = _json_object(descriptor_path)
    if descriptor.get("schema_version") != SCENARIO_PROVISIONING_SCHEMA_VERSION:
        raise ValueError("Meta-ARE dataset source descriptor has an unsupported schema version")
    if descriptor.get("repo_id") != DEFAULT_SCENARIO_REPO_ID:
        raise ValueError("Meta-ARE dataset source descriptor has an unexpected repository ID")
    if descriptor.get("source_revision") != revision:
        raise ValueError("Meta-ARE dataset source revision does not match its descriptor")
    if descriptor.get("split") != "validation":
        raise ValueError("Meta-ARE dataset source descriptor must select the validation split")

    raw_manifests = descriptor.get("manifests")
    if not isinstance(raw_manifests, list):
        raise TypeError("Meta-ARE dataset source descriptor manifests are malformed")
    manifest_digests = []
    for raw_manifest in raw_manifests:
        if not isinstance(raw_manifest, dict) or not isinstance(raw_manifest.get("sha256"), str):
            raise TypeError("Meta-ARE dataset source descriptor manifest entry is malformed")
        manifest_digests.append(raw_manifest["sha256"])
    expected_manifest_digests = [sha256_digest(path.read_bytes()) for path in manifests]
    if sorted(manifest_digests) != sorted(expected_manifest_digests):
        raise ValueError("Meta-ARE dataset source descriptor does not match the configured manifests")

    expected_case_ids = {
        case_id
        for manifest in manifests
        for case_id in _manifest_case_ids(manifest)
    }
    raw_files = descriptor.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise ValueError("Meta-ARE dataset source descriptor has no files")
    records = []
    seen_paths: set[str] = set()
    seen_case_ids: set[str] = set()
    for raw_file in raw_files:
        if not isinstance(raw_file, dict):
            raise TypeError("Meta-ARE dataset source descriptor file entry is malformed")
        scenario_id = raw_file.get("scenario_id")
        relative = raw_file.get("path")
        expected_bytes = raw_file.get("bytes")
        expected_digest = raw_file.get("sha256")
        if (
            not isinstance(scenario_id, str)
            or not isinstance(relative, str)
            or isinstance(expected_bytes, bool)
            or not isinstance(expected_bytes, int)
            or expected_bytes < 0
            or not isinstance(expected_digest, str)
        ):
            raise TypeError("Meta-ARE dataset source descriptor file entry is malformed")
        relative_path = Path(relative)
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or relative in seen_paths
            or scenario_id in seen_case_ids
        ):
            raise ValueError("Meta-ARE dataset source descriptor contains an unsafe or duplicate file")
        path = root / relative_path
        cursor = root
        for part in relative_path.parts:
            cursor /= part
            if cursor.is_symlink():
                raise ValueError(f"Meta-ARE dataset source contains a symlink: {relative}")
        if not path.is_file():
            raise ValueError(f"Meta-ARE dataset source file is missing: {relative}")
        payload = path.read_bytes()
        actual_digest = sha256_digest(payload)
        if len(payload) != expected_bytes or actual_digest != expected_digest:
            raise ValueError(f"Meta-ARE dataset source file digest drift: {relative}")
        seen_paths.add(relative)
        seen_case_ids.add(scenario_id)
        records.append(
            {
                "scenario_id": scenario_id,
                "path": relative,
                "bytes": expected_bytes,
                "sha256": actual_digest,
            }
        )
    if seen_case_ids != expected_case_ids:
        raise ValueError("Meta-ARE dataset source files do not match the configured case manifests")
    return sha256_digest(canonical_json(sorted(records, key=lambda record: record["scenario_id"])))


def _approved_paths(
    value: JsonValue,
    label: str,
    approved: tuple[PurePosixPath, ...],
) -> tuple[PurePosixPath, ...]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise TypeError(f"Meta-ARE {label} must be a list of relative paths")
    paths = tuple(PurePosixPath(item) for item in value)
    if any(path.is_absolute() or ".." in path.parts or path.as_posix() in {"", "."} for path in paths):
        raise ValueError(f"Meta-ARE {label} contains an unsafe path")
    if paths != approved:
        raise ValueError(
            f"Meta-ARE {label} must exactly match the approved mutation scope; "
            f"expected={[path.as_posix() for path in approved]}"
        )
    return paths


def _verify_demo_filesystem(
    *,
    root: Path,
    revision: str,
    descriptor_path: Path,
    manifest_path: Path,
) -> str:
    descriptor = _json_object(descriptor_path)
    recorded_revision = descriptor.get("source_revision", descriptor.get("revision"))
    if recorded_revision != revision:
        raise ValueError("Meta-ARE demo filesystem source revision does not match its descriptor")
    manifest = _json_object(manifest_path)
    raw_entries = manifest.get("entries", manifest.get("files"))
    if not isinstance(raw_entries, list) or not raw_entries:
        raise ValueError("Meta-ARE demo filesystem manifest has no files")
    expected: dict[str, str | None] = {}
    for raw_entry in raw_entries:
        if isinstance(raw_entry, str):
            relative = raw_entry
            digest = None
        elif isinstance(raw_entry, dict) and isinstance(raw_entry.get("path"), str):
            relative = raw_entry["path"]
            raw_digest = raw_entry.get("sha256")
            digest = raw_digest if isinstance(raw_digest, str) else None
        else:
            raise TypeError("Meta-ARE demo filesystem manifest entries are malformed")
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or relative in expected:
            raise ValueError("Meta-ARE demo filesystem manifest contains an unsafe or duplicate path")
        expected[relative] = digest
    actual_paths = {
        path.relative_to(root).as_posix(): path
        for path in root.rglob("*")
        if path.is_file() and not path.is_symlink()
    }
    if set(expected) != set(actual_paths):
        raise ValueError("Meta-ARE demo filesystem content does not match its manifest")
    records = []
    for relative in sorted(expected):
        path = actual_paths[relative]
        actual_digest = sha256_digest(path.read_bytes())
        if expected[relative] is not None and expected[relative] != actual_digest:
            raise ValueError(f"Meta-ARE demo filesystem manifest digest drift: {relative}")
        records.append({"path": relative, "size": path.stat().st_size, "sha256": actual_digest})
    recorded_content_digest = manifest.get("content_digest")
    if recorded_content_digest is None:
        content_digest = sha256_digest(canonical_json(records))
    else:
        body = {
            "schema_version": manifest.get("schema_version"),
            "file_count": len(records),
            "total_bytes": sum(record["size"] for record in records),
            "entries": records,
        }
        content_digest = sha256_digest(canonical_json(body))
        if recorded_content_digest != content_digest:
            raise ValueError("Meta-ARE demo filesystem aggregate manifest digest drift")
    return content_digest


def _require_git_commit(repo: Path, revision: str) -> None:
    completed = subprocess.run(
        ["git", "cat-file", "-e", f"{revision}^{{commit}}"],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ValueError(f"Meta-ARE base revision is not an available commit: {revision}")


def _git_file(repo: Path, revision: str, relative_path: str) -> bytes:
    completed = subprocess.run(
        ["git", "show", f"{revision}:{relative_path}"],
        cwd=repo,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise ValueError(f"Meta-ARE revision is missing required runtime file: {relative_path}")
    return completed.stdout


def _full_commit(value: JsonValue, label: str) -> str:
    text = _string(value, label)
    if _FULL_COMMIT.fullmatch(text) is None:
        raise ValueError(f"Meta-ARE {label} must be a full commit; symbolic revisions are forbidden")
    return text


def _directory(value: JsonValue, base_dir: Path, label: str) -> Path:
    path = _path(value, base_dir, label)
    if not path.is_dir():
        raise ValueError(f"Meta-ARE {label} must be an existing local directory: {path}")
    return path


def _file(value: JsonValue, base_dir: Path, label: str) -> Path:
    path = _path(value, base_dir, label)
    if not path.is_file():
        raise ValueError(f"Meta-ARE {label} must be an existing local file: {path}")
    return path


def _path(value: JsonValue, base_dir: Path, label: str) -> Path:
    text = _string(value, label)
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def _json_object_or_list(path: Path) -> dict[str, object] | list[object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"Malformed Meta-ARE JSON file: {path}") from error
    if not isinstance(value, (dict, list)):
        raise TypeError(f"Meta-ARE JSON file must contain an object or list: {path}")
    return value


def _json_object(path: Path) -> dict[str, object]:
    value = _json_object_or_list(path)
    if not isinstance(value, dict):
        raise TypeError(f"Meta-ARE JSON file must contain an object: {path}")
    return value


def _string(value: JsonValue, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"Meta-ARE {label} must be a non-empty string")
    return value


def _optional_string(value: JsonValue, label: str) -> str | None:
    if value is None:
        return None
    return _string(value, label)


def _as_optional_string(value: JsonValue) -> str | None:
    return value if isinstance(value, str) else None


def _positive_number(value: JsonValue, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
        raise ValueError(f"Meta-ARE {label} must be positive")
    return float(value)


def _positive_int(value: JsonValue, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"Meta-ARE {label} must be a positive integer")
    return value


def _nonnegative_int(value: JsonValue, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"Meta-ARE {label} must be a nonnegative integer")
    return value
