from __future__ import annotations

import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, Lock

import huggingface_hub
import pytest

from autosaddler.v2.plugins.meta_are import provisioning
from autosaddler.v2.plugins.meta_are.provisioning import (
    EXPECTED_TOP_LEVEL_DIRECTORIES,
    PROVISIONING_SCHEMA_VERSION,
    ProvisioningError,
    ProvisioningRequest,
    provision_demo_filesystem,
)

REVISION = "a" * 40


def write_demo_filesystem(snapshot_root: Path, *, content: str = '{"text_content":"example"}\n') -> Path:
    filesystem_root = snapshot_root / "demo_filesystem"
    for directory in EXPECTED_TOP_LEVEL_DIRECTORIES:
        (filesystem_root / directory).mkdir(parents=True)
    (filesystem_root / "Documents" / "example.json").write_text(content)
    return filesystem_root


def test_provisioning_request_rejects_symbolic_revision(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="full 40-character Git commit"):
        ProvisioningRequest(
            destination_root=tmp_path / "datasets",
            source_revision="main",
        )


def test_provisioning_request_rejects_relative_and_repository_destinations() -> None:
    with pytest.raises(ValueError, match="absolute"):
        ProvisioningRequest(destination_root=Path("datasets"), source_revision=REVISION)

    repository_root = Path(provisioning.__file__).resolve().parents[5]
    with pytest.raises(ValueError, match="outside the AutoSaddler repository"):
        ProvisioningRequest(destination_root=repository_root / "datasets", source_revision=REVISION)


def test_provisioning_request_requires_an_owned_existing_ancestor(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    real_stat = Path.stat

    def root_owned_stat(path: Path, *args: object, **kwargs: object) -> os.stat_result:
        result = real_stat(path, *args, **kwargs)
        if path == tmp_path:
            values = list(result)
            values[4] = os.getuid() + 1
            return os.stat_result(values)
        return result

    monkeypatch.setattr(Path, "stat", root_owned_stat)
    with pytest.raises(ValueError, match="user-owned"):
        ProvisioningRequest(destination_root=tmp_path / "datasets", source_revision=REVISION)


def test_provisioner_publishes_data_and_provenance_together(tmp_path: Path) -> None:
    request = ProvisioningRequest(destination_root=tmp_path / "datasets", source_revision=REVISION.upper())

    def acquire(_request: ProvisioningRequest, snapshot_root: Path) -> Path:
        write_demo_filesystem(snapshot_root)
        return snapshot_root

    def probe(filesystem_root: Path, representative_path: Path) -> dict[str, object]:
        payload = (filesystem_root / representative_path).read_bytes()
        return {
            "status": "passed",
            "implementation": "fake-meta-are-probe",
            "representative_path": representative_path.as_posix(),
            "bytes_read": len(payload),
        }

    result = provision_demo_filesystem(request, acquire=acquire, probe=probe)

    assert request.source_revision == REVISION
    assert result.reused is False
    assert result.filesystem_root == request.destination_root / REVISION / "demo_filesystem"
    assert (result.filesystem_root / "Documents" / "example.json").read_text() == '{"text_content":"example"}\n'
    source = json.loads(result.source_descriptor.read_text())
    manifest = json.loads(result.content_manifest.read_text())
    assert source["source_revision"] == REVISION
    assert source["content_digest"] == manifest["content_digest"] == result.content_digest
    assert source["read_probe"]["status"] == "passed"
    assert source["read_probe"]["representative_path"] == "Documents/example.json"
    assert manifest["file_count"] == 1
    assert manifest["total_bytes"] > 0


def test_provisioner_does_not_publish_when_meta_are_probe_fails(tmp_path: Path) -> None:
    request = ProvisioningRequest(destination_root=tmp_path / "datasets", source_revision=REVISION)

    def acquire(_request: ProvisioningRequest, snapshot_root: Path) -> Path:
        write_demo_filesystem(snapshot_root)
        return snapshot_root

    def probe(_filesystem_root: Path, _representative_path: Path) -> dict[str, object]:
        raise ProvisioningError("Meta-ARE probe failed")

    with pytest.raises(ProvisioningError, match="probe failed"):
        provision_demo_filesystem(request, acquire=acquire, probe=probe)

    assert not request.revision_root.exists()
    assert not list(request.destination_root.glob(".autosaddler-meta-are-*"))


def test_snapshot_download_uses_pinned_dataset_revision_and_demo_pattern(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = ProvisioningRequest(destination_root=tmp_path / "datasets", source_revision=REVISION)
    snapshot_root = tmp_path / "snapshot"
    calls: list[dict[str, object]] = []

    def snapshot_download(**kwargs: object) -> str:
        calls.append(kwargs)
        return str(snapshot_root)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", snapshot_download)

    assert provisioning._snapshot_download(request, snapshot_root) == snapshot_root
    assert calls == [
        {
            "repo_id": request.repo_id,
            "repo_type": "dataset",
            "revision": REVISION,
            "allow_patterns": "demo_filesystem/**",
            "local_dir": snapshot_root,
        }
    ]


def test_meta_are_probe_uses_external_uv_runtime_and_offline_data_clients(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    meta_are_project = tmp_path / "Meta-ARE"
    meta_are_project.mkdir()
    (meta_are_project / "pyproject.toml").write_text("[project]\nname = 'are'\n")
    filesystem_root = write_demo_filesystem(tmp_path / "snapshot")
    representative_path = Path("Documents/example.json")
    calls: list[tuple[list[str], dict[str, object]]] = []

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append((command, kwargs))
        payload = (filesystem_root / representative_path).read_bytes()
        record = {
            "bytes_read": len(payload),
            "sha256": provisioning._sha256_bytes(payload),
        }
        return subprocess.CompletedProcess(
            command,
            returncode=0,
            stdout="AUTOSADDLER_META_ARE_PROBE=" + json.dumps(record) + "\n",
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", run)

    result = provisioning.run_meta_are_filesystem_probe(
        meta_are_project,
        filesystem_root,
        representative_path,
    )

    assert result["status"] == "passed"
    assert result["representative_path"] == representative_path.as_posix()
    command, kwargs = calls[0]
    assert command[:4] == ["uv", "run", "--project", str(meta_are_project)]
    assert command[-2:] == [str(filesystem_root), representative_path.as_posix()]
    environment = kwargs["env"]
    assert isinstance(environment, dict)
    assert environment["HF_HUB_OFFLINE"] == "1"
    assert environment["HF_DATASETS_OFFLINE"] == "1"
    assert environment["TRANSFORMERS_OFFLINE"] == "1"


def test_provisioner_rejects_pointer_payload_without_publishing(tmp_path: Path) -> None:
    request = ProvisioningRequest(destination_root=tmp_path / "datasets", source_revision=REVISION)

    def acquire(_request: ProvisioningRequest, snapshot_root: Path) -> Path:
        write_demo_filesystem(
            snapshot_root,
            content="version https://git-lfs.github.com/spec/v1\noid sha256:" + "b" * 64 + "\nsize 100\n",
        )
        return snapshot_root

    with pytest.raises(ProvisioningError, match="pointer"):
        provision_demo_filesystem(request, acquire=acquire)

    assert not (request.destination_root / REVISION).exists()


def test_provisioner_rejects_xet_stub_without_publishing(tmp_path: Path) -> None:
    request = ProvisioningRequest(destination_root=tmp_path / "datasets", source_revision=REVISION)

    def acquire(_request: ProvisioningRequest, snapshot_root: Path) -> Path:
        write_demo_filesystem(
            snapshot_root,
            content="# xet version 0\ngit-hash = " + "b" * 64 + "\nfilesize = 100\n",
        )
        return snapshot_root

    with pytest.raises(ProvisioningError, match="pointer"):
        provision_demo_filesystem(request, acquire=acquire)

    assert not (request.destination_root / REVISION).exists()


def test_provisioner_rejects_missing_structure_without_publishing(tmp_path: Path) -> None:
    request = ProvisioningRequest(destination_root=tmp_path / "datasets", source_revision=REVISION)

    def acquire(_request: ProvisioningRequest, snapshot_root: Path) -> Path:
        (snapshot_root / "demo_filesystem" / "Documents").mkdir(parents=True)
        (snapshot_root / "demo_filesystem" / "Documents" / "example.json").write_text("payload")
        return snapshot_root

    with pytest.raises(ProvisioningError, match="top-level"):
        provision_demo_filesystem(request, acquire=acquire)

    assert not (request.destination_root / REVISION).exists()


def test_provisioner_cleans_owned_temporary_directory_after_failure(tmp_path: Path) -> None:
    request = ProvisioningRequest(destination_root=tmp_path / "datasets", source_revision=REVISION)

    def acquire(_request: ProvisioningRequest, snapshot_root: Path) -> Path:
        raise RuntimeError("download failed")

    with pytest.raises(RuntimeError, match="download failed"):
        provision_demo_filesystem(request, acquire=acquire)

    assert not list(request.destination_root.glob(".autosaddler-meta-are-*"))
    assert not (request.destination_root / REVISION).exists()


def test_provisioner_removes_only_matching_owned_stale_directories(tmp_path: Path) -> None:
    request = ProvisioningRequest(destination_root=tmp_path / "datasets", source_revision=REVISION)
    request.destination_root.mkdir()
    matching = request.destination_root / f".autosaddler-meta-are-{REVISION[:12]}-matching"
    mismatched = request.destination_root / f".autosaddler-meta-are-{REVISION[:12]}-mismatched"
    unmarked = request.destination_root / f".autosaddler-meta-are-{REVISION[:12]}-unmarked"
    for path in (matching, mismatched, unmarked):
        path.mkdir()
    (matching / ".autosaddler-provisioning.json").write_text(
        json.dumps(
            {
                "schema_version": PROVISIONING_SCHEMA_VERSION,
                "repo_id": request.repo_id,
                "source_revision": REVISION,
            }
        )
    )
    (mismatched / ".autosaddler-provisioning.json").write_text(
        json.dumps(
            {
                "schema_version": PROVISIONING_SCHEMA_VERSION,
                "repo_id": "somebody/else",
                "source_revision": REVISION,
            }
        )
    )

    def acquire(_request: ProvisioningRequest, snapshot_root: Path) -> Path:
        write_demo_filesystem(snapshot_root)
        return snapshot_root

    provision_demo_filesystem(request, acquire=acquire)

    assert not matching.exists()
    assert mismatched.exists()
    assert unmarked.exists()


def test_provisioner_cleans_temporary_directory_when_validation_is_interrupted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = ProvisioningRequest(destination_root=tmp_path / "datasets", source_revision=REVISION)

    def acquire(_request: ProvisioningRequest, snapshot_root: Path) -> Path:
        write_demo_filesystem(snapshot_root)
        return snapshot_root

    def interrupt(_root: Path) -> None:
        raise RuntimeError("interrupted after validation")

    monkeypatch.setattr(provisioning, "_fsync_tree", interrupt)
    with pytest.raises(RuntimeError, match="interrupted after validation"):
        provision_demo_filesystem(request, acquire=acquire)

    assert not list(request.destination_root.glob(".autosaddler-meta-are-*"))
    assert not request.revision_root.exists()


def test_provisioner_recovers_when_interrupted_after_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = ProvisioningRequest(destination_root=tmp_path / "datasets", source_revision=REVISION)

    def acquire(_request: ProvisioningRequest, snapshot_root: Path) -> Path:
        write_demo_filesystem(snapshot_root)
        return snapshot_root

    real_fsync_directory = provisioning._fsync_directory

    def interrupt_after_publish(path: Path) -> None:
        if path == request.destination_root and request.revision_root.exists():
            raise RuntimeError("interrupted after publish")
        real_fsync_directory(path)

    monkeypatch.setattr(provisioning, "_fsync_directory", interrupt_after_publish)
    with pytest.raises(RuntimeError, match="interrupted after publish"):
        provision_demo_filesystem(request, acquire=acquire)
    monkeypatch.setattr(provisioning, "_fsync_directory", real_fsync_directory)

    result = provision_demo_filesystem(request, acquire=acquire)

    assert result.reused is True
    assert request.revision_root.is_dir()


def test_provisioner_serializes_concurrent_publication(tmp_path: Path) -> None:
    request = ProvisioningRequest(destination_root=tmp_path / "datasets", source_revision=REVISION)
    start = Barrier(3)
    call_lock = Lock()
    calls = 0

    def acquire(_request: ProvisioningRequest, snapshot_root: Path) -> Path:
        nonlocal calls
        with call_lock:
            calls += 1
        write_demo_filesystem(snapshot_root)
        return snapshot_root

    def provision() -> bool:
        start.wait()
        return provision_demo_filesystem(request, acquire=acquire).reused

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(provision) for _ in range(2)]
        start.wait()
        reused = [future.result() for future in futures]

    assert sorted(reused) == [False, True]
    assert calls == 1


def test_provisioner_rejects_acquisition_outside_owned_temporary_directory(tmp_path: Path) -> None:
    request = ProvisioningRequest(destination_root=tmp_path / "datasets", source_revision=REVISION)
    outside = write_demo_filesystem(tmp_path / "outside").parent

    def acquire(_request: ProvisioningRequest, _snapshot_root: Path) -> Path:
        return outside

    with pytest.raises(ProvisioningError, match="outside"):
        provision_demo_filesystem(request, acquire=acquire)

    assert not request.revision_root.exists()


def test_provisioner_reuses_verified_destination_without_acquiring(tmp_path: Path) -> None:
    request = ProvisioningRequest(destination_root=tmp_path / "datasets", source_revision=REVISION)
    calls = 0

    def acquire(_request: ProvisioningRequest, snapshot_root: Path) -> Path:
        nonlocal calls
        calls += 1
        write_demo_filesystem(snapshot_root)
        return snapshot_root

    first = provision_demo_filesystem(request, acquire=acquire)
    second = provision_demo_filesystem(request, acquire=acquire)

    assert first.content_digest == second.content_digest
    assert first.reused is False
    assert second.reused is True
    assert calls == 1


def test_provisioner_reprobes_existing_destination_before_reuse(tmp_path: Path) -> None:
    request = ProvisioningRequest(destination_root=tmp_path / "datasets", source_revision=REVISION)
    probes = 0

    def acquire(_request: ProvisioningRequest, snapshot_root: Path) -> Path:
        write_demo_filesystem(snapshot_root)
        return snapshot_root

    def probe(filesystem_root: Path, representative_path: Path) -> dict[str, object]:
        nonlocal probes
        probes += 1
        payload = (filesystem_root / representative_path).read_bytes()
        return {
            "status": "passed",
            "representative_path": representative_path.as_posix(),
            "bytes_read": len(payload),
        }

    provision_demo_filesystem(request, acquire=acquire, probe=probe)
    result = provision_demo_filesystem(request, acquire=acquire, probe=probe)

    assert result.reused is True
    assert probes == 2


def test_provisioner_fails_on_published_content_drift(tmp_path: Path) -> None:
    request = ProvisioningRequest(destination_root=tmp_path / "datasets", source_revision=REVISION)

    def acquire(_request: ProvisioningRequest, snapshot_root: Path) -> Path:
        write_demo_filesystem(snapshot_root)
        return snapshot_root

    result = provision_demo_filesystem(request, acquire=acquire)
    (result.filesystem_root / "Documents" / "example.json").write_text("changed")

    with pytest.raises(ProvisioningError, match="digest"):
        provision_demo_filesystem(request, acquire=acquire)