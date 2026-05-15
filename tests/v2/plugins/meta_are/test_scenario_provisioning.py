from __future__ import annotations

import json
from pathlib import Path

import pytest

from autosaddler.v2.plugins.meta_are.scenario_provisioning import (
    ScenarioProvisioningError,
    ScenarioProvisioningRequest,
    provision_gaia2_scenarios,
)


REVISION = "a" * 40


def _manifest(path: Path, value: object) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def test_provisions_only_manifest_selected_scenarios_and_reuses_identical_files(tmp_path: Path) -> None:
    train = _manifest(tmp_path / "train.json", {"search": ["scenario_train"]})
    development = _manifest(tmp_path / "development.json", {"search": ["scenario_development"]})
    destination = tmp_path / "external" / "gaia2"
    rows = [
        {"scenario_id": "scenario_ignored", "data": '{"ignored": true}'},
        {"scenario_id": "scenario_train", "data": '{"train": true}'},
        {"scenario_id": "scenario_development", "data": '{"development": true}'},
    ]
    calls = []

    def loader(repo_id: str, **kwargs: object) -> object:
        calls.append((repo_id, kwargs))
        return rows

    request = ScenarioProvisioningRequest(
        destination_root=destination,
        source_revision=REVISION,
        manifests=(train, development),
    )
    first = provision_gaia2_scenarios(request, loader=loader)
    second = provision_gaia2_scenarios(request, loader=loader)

    assert first.file_count == 2
    assert first.reused_count == 0
    assert second.reused_count == 2
    assert (destination / "search/validation/0001_scenario_train.json").read_text() == '{"train": true}'
    assert (destination / "search/validation/0002_scenario_development.json").read_text() == '{"development": true}'
    assert not (destination / "search/validation/0000_scenario_ignored.json").exists()
    descriptor = json.loads(first.source_descriptor.read_text())
    assert descriptor["source_revision"] == REVISION
    assert [record["scenario_id"] for record in descriptor["files"]] == [
        "scenario_development",
        "scenario_train",
    ]
    assert calls == [
        ("meta-agents-research-environments/gaia2", {"name": "search", "split": "validation", "revision": REVISION}),
        ("meta-agents-research-environments/gaia2", {"name": "search", "split": "validation", "revision": REVISION}),
    ]


def test_missing_scenario_does_not_publish_partial_data(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path / "train.json", {"search": ["scenario_missing"]})
    destination = tmp_path / "external" / "gaia2"
    request = ScenarioProvisioningRequest(
        destination_root=destination,
        source_revision=REVISION,
        manifests=(manifest,),
    )

    with pytest.raises(ScenarioProvisioningError, match="requested scenarios were not found"):
        provision_gaia2_scenarios(request, loader=lambda *args, **kwargs: [])

    assert not destination.exists()


def test_rejects_symlinked_destination_child(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path / "train.json", {"search": ["scenario_train"]})
    destination = tmp_path / "external" / "gaia2"
    outside = tmp_path / "outside"
    destination.mkdir(parents=True)
    outside.mkdir()
    (destination / "search").symlink_to(outside, target_is_directory=True)
    request = ScenarioProvisioningRequest(
        destination_root=destination,
        source_revision=REVISION,
        manifests=(manifest,),
    )

    with pytest.raises(ScenarioProvisioningError, match="symlink"):
        provision_gaia2_scenarios(
            request,
            loader=lambda *args, **kwargs: [
                {"scenario_id": "scenario_train", "data": '{"train": true}'},
            ],
        )

    assert tuple(outside.rglob("*")) == ()


def test_rejects_destination_root_replaced_by_symlink_during_acquisition(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path / "train.json", {"search": ["scenario_train"]})
    destination = tmp_path / "external" / "gaia2"
    outside = tmp_path / "outside"
    outside.mkdir()
    request = ScenarioProvisioningRequest(
        destination_root=destination,
        source_revision=REVISION,
        manifests=(manifest,),
    )

    def loader(*args: object, **kwargs: object) -> object:
        destination.parent.mkdir(parents=True)
        destination.symlink_to(outside, target_is_directory=True)
        return [{"scenario_id": "scenario_train", "data": '{"train": true}'}]

    with pytest.raises(ScenarioProvisioningError, match="destination_root.*symlink"):
        provision_gaia2_scenarios(request, loader=loader)

    assert tuple(outside.rglob("*")) == ()


def test_rejects_preexisting_exact_name_duplicate(tmp_path: Path) -> None:
    manifest = _manifest(tmp_path / "train.json", {"search": ["scenario_train"]})
    destination = tmp_path / "external" / "gaia2"
    existing = destination / "other" / "scenario_train.json"
    existing.parent.mkdir(parents=True)
    existing.write_text('{"old": true}', encoding="utf-8")
    request = ScenarioProvisioningRequest(
        destination_root=destination,
        source_revision=REVISION,
        manifests=(manifest,),
    )

    with pytest.raises(ScenarioProvisioningError, match="unexpected path"):
        provision_gaia2_scenarios(
            request,
            loader=lambda *args, **kwargs: [
                {"scenario_id": "scenario_train", "data": '{"train": true}'},
            ],
        )

    assert existing.read_text(encoding="utf-8") == '{"old": true}'