from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Iterator

DEFAULT_FILESYSTEM_REPO_ID = "meta-agents-research-environments/gaia2_filesystem"
EXPECTED_TOP_LEVEL_DIRECTORIES = (
    "Documents",
    "Downloads",
    "Pictures",
)
PROVISIONING_SCHEMA_VERSION = "autosaddler-meta-are-filesystem/v1"
_OWNERSHIP_MARKER = ".autosaddler-provisioning.json"
_FULL_GIT_COMMIT = re.compile(r"[0-9a-fA-F]{40}")
_PROJECT_ROOT = Path(__file__).resolve().parents[5]
_PROBE_MARKER = "AUTOSADDLER_META_ARE_PROBE="
_META_ARE_PROBE_CODE = f"""
import hashlib
import json
import sys
import tempfile

from are.simulation.apps.sandbox_file_system import SandboxLocalFileSystem

filesystem_root = sys.argv[1]
relative_path = sys.argv[2]
with tempfile.TemporaryDirectory(prefix="autosaddler-meta-are-probe-") as sandbox_root:
    filesystem = SandboxLocalFileSystem(
        sandbox_dir=sandbox_root,
        state_directory=filesystem_root,
    )
    filesystem.local_fs.set_fallback_root(filesystem_root, {{"/" + relative_path}})
    payload = filesystem.cat("/" + relative_path)
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    print(
        {_PROBE_MARKER!r}
        + json.dumps(
            {{
                "bytes_read": len(payload),
                "sha256": "sha256:" + hashlib.sha256(payload).hexdigest(),
            }},
            sort_keys=True,
        )
    )
"""


class ProvisioningError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ProvisioningRequest:
    destination_root: Path
    source_revision: str
    repo_id: str = DEFAULT_FILESYSTEM_REPO_ID

    def __post_init__(self) -> None:
        if _FULL_GIT_COMMIT.fullmatch(self.source_revision) is None:
            raise ValueError("source_revision must be a full 40-character Git commit")
        if not self.repo_id.strip():
            raise ValueError("repo_id must be non-empty")
        destination_root = Path(self.destination_root).expanduser()
        if not destination_root.is_absolute():
            raise ValueError("destination_root must be absolute")
        if destination_root.is_symlink():
            raise ValueError("destination_root must not be a symlink")
        destination_root = destination_root.resolve(strict=False)
        if _is_within(destination_root, _PROJECT_ROOT):
            raise ValueError("destination_root must be outside the AutoSaddler repository")
        existing_ancestor = _nearest_existing_ancestor(destination_root)
        if existing_ancestor.stat().st_uid != os.getuid():
            raise ValueError("destination_root must have a user-owned existing ancestor")
        object.__setattr__(self, "destination_root", destination_root)
        object.__setattr__(self, "source_revision", self.source_revision.lower())

    @property
    def revision_root(self) -> Path:
        return self.destination_root / self.source_revision


@dataclass(frozen=True, slots=True)
class ProvisioningResult:
    revision_root: Path
    filesystem_root: Path
    source_descriptor: Path
    content_manifest: Path
    content_digest: str
    file_count: int
    total_bytes: int
    reused: bool


Acquirer = Callable[[ProvisioningRequest, Path], Path]
Probe = Callable[[Path, Path], Mapping[str, Any]]


def provision_demo_filesystem(
    request: ProvisioningRequest,
    *,
    acquire: Acquirer | None = None,
    probe: Probe | None = None,
) -> ProvisioningResult:
    request.destination_root.mkdir(parents=True, exist_ok=True)
    if request.destination_root.is_symlink():
        raise ProvisioningError("destination_root became a symlink")

    with _destination_lock(request.destination_root):
        if request.revision_root.exists():
            return _verify_published(request, reused=True, probe=probe)

        _remove_owned_stale_directories(request)
        temporary_root = Path(
            tempfile.mkdtemp(
                prefix=f".autosaddler-meta-are-{request.source_revision[:12]}-",
                dir=request.destination_root,
            )
        )
        try:
            _write_json(
                temporary_root / _OWNERSHIP_MARKER,
                {
                    "schema_version": PROVISIONING_SCHEMA_VERSION,
                    "repo_id": request.repo_id,
                    "source_revision": request.source_revision,
                },
            )
            snapshot_root = temporary_root / "snapshot"
            snapshot_root.mkdir()
            acquisition = acquire or _snapshot_download
            acquired_root = Path(acquisition(request, snapshot_root)).resolve(strict=True)
            if not _is_within(acquired_root, temporary_root):
                raise ProvisioningError("acquisition returned a path outside the owned temporary directory")
            acquired_filesystem = acquired_root / "demo_filesystem"
            _validate_filesystem_tree(acquired_filesystem)

            publication_root = temporary_root / "publication"
            publication_root.mkdir()
            filesystem_root = publication_root / "demo_filesystem"
            shutil.move(str(acquired_filesystem), filesystem_root)
            manifest = _build_manifest(filesystem_root)
            representative_path = _representative_payload(manifest)
            read_probe: dict[str, Any] = {"status": "not_run"}
            if probe is not None:
                read_probe = dict(probe(filesystem_root, representative_path))
                if read_probe.get("status") != "passed":
                    raise ProvisioningError("Meta-ARE read probe did not return status 'passed'")
                manifest_after_probe = _build_manifest(filesystem_root)
                if _canonical_json(manifest_after_probe) != _canonical_json(manifest):
                    raise ProvisioningError("Meta-ARE read probe modified the demo filesystem")
            source = _build_source_descriptor(
                request,
                manifest,
                injected=acquire is not None,
                read_probe=read_probe,
            )
            _write_json(publication_root / "manifest.json", manifest)
            _write_json(publication_root / "source.json", source)
            _fsync_tree(publication_root)

            if request.revision_root.exists():
                return _verify_published(request, reused=True, probe=probe)
            publication_root.rename(request.revision_root)
            _fsync_directory(request.destination_root)
            return _verify_published(request, reused=False)
        finally:
            if temporary_root.exists():
                shutil.rmtree(temporary_root)


def _snapshot_download(request: ProvisioningRequest, snapshot_root: Path) -> Path:
    from huggingface_hub import snapshot_download

    downloaded = snapshot_download(
        repo_id=request.repo_id,
        repo_type="dataset",
        revision=request.source_revision,
        allow_patterns="demo_filesystem/**",
        local_dir=snapshot_root,
    )
    return Path(downloaded)


def run_meta_are_filesystem_probe(
    meta_are_project: Path,
    filesystem_root: Path,
    representative_path: Path,
) -> dict[str, Any]:
    meta_are_project = Path(meta_are_project).resolve(strict=True)
    if not (meta_are_project / "pyproject.toml").is_file():
        raise ProvisioningError("Meta-ARE project must contain pyproject.toml")
    filesystem_root = Path(filesystem_root).resolve(strict=True)
    representative_path = Path(representative_path)
    if representative_path.is_absolute() or ".." in representative_path.parts:
        raise ProvisioningError("representative probe path must be relative and confined")
    payload_path = (filesystem_root / representative_path).resolve(strict=True)
    if not _is_within(payload_path, filesystem_root) or not payload_path.is_file():
        raise ProvisioningError("representative probe path escapes the demo filesystem")

    with tempfile.TemporaryDirectory(prefix="autosaddler-meta-are-probe-runtime-") as runtime_directory:
        runtime_root = Path(runtime_directory)
        cache_paths = {
            "HF_HOME": runtime_root / "hf",
            "HF_DATASETS_CACHE": runtime_root / "datasets",
            "TRANSFORMERS_CACHE": runtime_root / "transformers",
        }
        for cache_path in cache_paths.values():
            cache_path.mkdir()
        environment = os.environ.copy()
        environment.update({name: str(path) for name, path in cache_paths.items()})
        environment.update(
            {
                "HF_HUB_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            }
        )
        command = [
            "uv",
            "run",
            "--project",
            str(meta_are_project),
            "python",
            "-c",
            _META_ARE_PROBE_CODE,
            str(filesystem_root),
            representative_path.as_posix(),
        ]
        completed = subprocess.run(
            command,
            cwd=meta_are_project,
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
    if completed.returncode != 0:
        stderr = completed.stderr[-4000:]
        raise ProvisioningError(f"Meta-ARE filesystem probe failed with exit {completed.returncode}: {stderr}")
    probe_lines = [line for line in completed.stdout.splitlines() if line.startswith(_PROBE_MARKER)]
    if len(probe_lines) != 1:
        raise ProvisioningError("Meta-ARE filesystem probe did not emit exactly one result record")
    try:
        probe_result = json.loads(probe_lines[0].removeprefix(_PROBE_MARKER))
    except json.JSONDecodeError as error:
        raise ProvisioningError("Meta-ARE filesystem probe emitted malformed JSON") from error
    expected_size = payload_path.stat().st_size
    expected_digest = _sha256_file(payload_path)
    if not isinstance(probe_result, dict):
        raise ProvisioningError("Meta-ARE filesystem probe result must be a JSON object")
    if probe_result.get("bytes_read") != expected_size or probe_result.get("sha256") != expected_digest:
        raise ProvisioningError("Meta-ARE filesystem probe payload does not match the local source")
    return {
        "status": "passed",
        "implementation": "Meta-ARE SandboxLocalFileSystem",
        "meta_are_project": str(meta_are_project),
        "representative_path": representative_path.as_posix(),
        "bytes_read": expected_size,
        "sha256": expected_digest,
        "offline_data_clients": True,
    }


def _validate_filesystem_tree(filesystem_root: Path) -> None:
    if not filesystem_root.is_dir():
        raise ProvisioningError("download does not contain a demo_filesystem directory")
    missing = [name for name in EXPECTED_TOP_LEVEL_DIRECTORIES if not (filesystem_root / name).is_dir()]
    if missing:
        raise ProvisioningError(f"demo filesystem is missing expected top-level directories: {missing}")

    regular_files = []
    for path in sorted(filesystem_root.rglob("*")):
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise ProvisioningError(f"demo filesystem contains a symlink: {path.relative_to(filesystem_root)}")
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise ProvisioningError(f"demo filesystem contains a special file: {path.relative_to(filesystem_root)}")
        if _is_pointer_payload(path):
            raise ProvisioningError(f"demo filesystem contains an unresolved LFS/Xet-backed pointer: {path.name}")
        regular_files.append(path)

    if not regular_files:
        raise ProvisioningError("demo filesystem contains no regular file payloads")
    documents = [path for path in regular_files if _is_within(path, filesystem_root / "Documents")]
    if not documents or not any(path.stat().st_size > 0 for path in documents):
        raise ProvisioningError("demo filesystem Documents directory has no representative nonempty payload")


def _is_pointer_payload(path: Path) -> bool:
    if path.stat().st_size > 4096:
        return False
    prefix = path.read_bytes()[:256]
    pointer_headers = (
        b"version https://git-lfs.github.com/spec/v1",
        b"version https://xetdata.com/spec/v1",
        b"# xet version ",
    )
    return prefix.startswith(pointer_headers)


def _build_manifest(filesystem_root: Path) -> dict[str, Any]:
    _validate_filesystem_tree(filesystem_root)
    entries = []
    total_bytes = 0
    for path in sorted(item for item in filesystem_root.rglob("*") if item.is_file()):
        relative_path = path.relative_to(filesystem_root).as_posix()
        size = path.stat().st_size
        total_bytes += size
        entries.append(
            {
                "path": relative_path,
                "size": size,
                "sha256": _sha256_file(path),
            }
        )
    body = {
        "schema_version": PROVISIONING_SCHEMA_VERSION,
        "file_count": len(entries),
        "total_bytes": total_bytes,
        "entries": entries,
    }
    return {**body, "content_digest": _sha256_bytes(_canonical_json(body).encode("utf-8"))}


def _build_source_descriptor(
    request: ProvisioningRequest,
    manifest: dict[str, Any],
    *,
    injected: bool,
    read_probe: dict[str, Any],
) -> dict[str, Any]:
    try:
        hub_version = version("huggingface-hub")
    except PackageNotFoundError:
        hub_version = "not-installed"
    try:
        xet_version = version("hf-xet")
    except PackageNotFoundError:
        xet_version = "not-installed"
    return {
        "schema_version": PROVISIONING_SCHEMA_VERSION,
        "repo_id": request.repo_id,
        "repo_type": "dataset",
        "source_revision": request.source_revision,
        "filesystem_path": "demo_filesystem",
        "content_manifest": "manifest.json",
        "content_digest": manifest["content_digest"],
        "file_count": manifest["file_count"],
        "total_bytes": manifest["total_bytes"],
        "read_probe": read_probe,
        "acquisition": {
            "implementation": "injected-test-acquirer" if injected else "huggingface_hub.snapshot_download",
            "huggingface_hub_version": hub_version,
            "hf_xet_version": xet_version,
            "allow_patterns": "demo_filesystem/**",
        },
    }


def _representative_payload(manifest: dict[str, Any]) -> Path:
    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise ProvisioningError("content manifest entries are missing")
    for entry in entries:
        if (
            isinstance(entry, dict)
            and isinstance(entry.get("path"), str)
            and entry["path"].startswith("Documents/")
            and isinstance(entry.get("size"), int)
            and entry["size"] > 0
        ):
            return Path(entry["path"])
    raise ProvisioningError("content manifest has no representative Documents payload")


def _verify_published(
    request: ProvisioningRequest,
    *,
    reused: bool,
    probe: Probe | None = None,
) -> ProvisioningResult:
    revision_root = request.revision_root
    filesystem_root = revision_root / "demo_filesystem"
    source_path = revision_root / "source.json"
    manifest_path = revision_root / "manifest.json"
    if not source_path.is_file() or not manifest_path.is_file():
        raise ProvisioningError("published revision is missing source or content manifest")
    source = _read_json_object(source_path)
    expected_manifest = _read_json_object(manifest_path)
    if source.get("repo_id") != request.repo_id or source.get("source_revision") != request.source_revision:
        raise ProvisioningError("published source provenance does not match the provisioning request")
    actual_manifest = _build_manifest(filesystem_root)
    if _canonical_json(expected_manifest) != _canonical_json(actual_manifest):
        raise ProvisioningError("published demo filesystem content or manifest digest has drifted")
    if source.get("content_digest") != actual_manifest["content_digest"]:
        raise ProvisioningError("published source descriptor content digest has drifted")
    if probe is not None:
        if not isinstance(source.get("read_probe"), dict) or source["read_probe"].get("status") != "passed":
            raise ProvisioningError("published source descriptor has no successful Meta-ARE read probe")
        representative_path = _representative_payload(actual_manifest)
        current_probe = dict(probe(filesystem_root, representative_path))
        if current_probe.get("status") != "passed":
            raise ProvisioningError("Meta-ARE read probe did not return status 'passed' during reuse")
        manifest_after_probe = _build_manifest(filesystem_root)
        if _canonical_json(manifest_after_probe) != _canonical_json(actual_manifest):
            raise ProvisioningError("Meta-ARE read probe modified the published demo filesystem")
    return ProvisioningResult(
        revision_root=revision_root,
        filesystem_root=filesystem_root,
        source_descriptor=source_path,
        content_manifest=manifest_path,
        content_digest=str(actual_manifest["content_digest"]),
        file_count=int(actual_manifest["file_count"]),
        total_bytes=int(actual_manifest["total_bytes"]),
        reused=reused,
    )


def _remove_owned_stale_directories(request: ProvisioningRequest) -> None:
    prefix = f".autosaddler-meta-are-{request.source_revision[:12]}-"
    for path in request.destination_root.glob(f"{prefix}*"):
        marker_path = path / _OWNERSHIP_MARKER
        if not path.is_dir() or not marker_path.is_file():
            continue
        marker = _read_json_object(marker_path)
        if (
            marker.get("schema_version") == PROVISIONING_SCHEMA_VERSION
            and marker.get("repo_id") == request.repo_id
            and marker.get("source_revision") == request.source_revision
        ):
            shutil.rmtree(path)


@contextmanager
def _destination_lock(destination_root: Path) -> Iterator[None]:
    lock_path = destination_root / ".autosaddler-meta-are.lock"
    with lock_path.open("a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(_canonical_json(value) + "\n")


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ProvisioningError(f"cannot read provisioning record {path}: {error}") from error
    if not isinstance(value, dict):
        raise ProvisioningError(f"provisioning record must be a JSON object: {path}")
    return value


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return f"sha256:{digest.hexdigest()}"


def _sha256_bytes(value: bytes) -> str:
    return f"sha256:{hashlib.sha256(value).hexdigest()}"


def _is_within(path: Path, root: Path) -> bool:
    path = path.resolve(strict=False)
    root = root.resolve(strict=False)
    return path == root or root in path.parents


def _nearest_existing_ancestor(path: Path) -> Path:
    candidate = path
    while not candidate.exists():
        if candidate.parent == candidate:
            raise ValueError("destination_root has no existing ancestor")
        candidate = candidate.parent
    return candidate


def _fsync_tree(root: Path) -> None:
    for path in sorted((item for item in root.rglob("*") if item.is_file()), key=lambda item: item.as_posix()):
        descriptor = os.open(path, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    directories = [root, *(item for item in root.rglob("*") if item.is_dir())]
    for directory in sorted(directories, key=lambda item: len(item.parts), reverse=True):
        _fsync_directory(directory)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)