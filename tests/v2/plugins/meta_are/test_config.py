from __future__ import annotations

import json
import hashlib
import subprocess
from pathlib import Path

import pytest


def _git(path: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=path, text=True, capture_output=True, check=True
    ).stdout.strip()


def _settings_mapping(tmp_path: Path) -> tuple[dict[str, object], str]:
    source = tmp_path / "meta-are"
    source.mkdir(parents=True)
    (source / "pyproject.toml").write_text("[project]\nname = 'meta-are-fixture'\n")
    (source / "uv.lock").write_text("version = 1\n")
    (source / "are").mkdir()
    _git(source, "init", "-q")
    _git(source, "add", ".")
    _git(source, "-c", "user.name=Test", "-c", "user.email=test@example.com", "commit", "-qm", "fixture")
    revision = _git(source, "rev-parse", "HEAD")

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    for case_id in ("train-a", "train-b", "dev-a", "test-a"):
        (dataset / f"{case_id}.json").write_text(json.dumps({"id": case_id}) + "\n")
    train_manifest = tmp_path / "train.json"
    train_manifest.write_text(json.dumps(["train-a", "train-b"]) + "\n")
    development_manifest = tmp_path / "development.json"
    development_manifest.write_text(json.dumps(["dev-a"]) + "\n")
    test_manifest = tmp_path / "test-universe-22.json"
    test_manifest.write_text(json.dumps(["test-a"]) + "\n")
    dataset_revision = "b" * 40
    dataset_descriptor = dataset / ".autosaddler-gaia2-source.json"
    dataset_descriptor.write_text(
        json.dumps(
            {
                "schema_version": "autosaddler-meta-are-scenarios/v1",
                "repo_id": "meta-agents-research-environments/gaia2",
                "source_revision": dataset_revision,
                "split": "validation",
                "manifests": [
                    {
                        "path": str(path),
                        "sha256": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest(),
                    }
                    for path in (train_manifest, development_manifest)
                ],
                "files": [
                    {
                        "scenario_id": case_id,
                        "path": f"{case_id}.json",
                        "bytes": len((dataset / f"{case_id}.json").read_bytes()),
                        "sha256": "sha256:"
                        + hashlib.sha256((dataset / f"{case_id}.json").read_bytes()).hexdigest(),
                    }
                    for case_id in ("train-a", "train-b", "dev-a")
                ],
            }
        )
        + "\n"
    )

    demo = tmp_path / "demo-filesystem"
    demo.mkdir()
    (demo / "README.txt").write_text("fixture\n")
    demo_descriptor = tmp_path / "demo-source.json"
    demo_descriptor.write_text(json.dumps({"revision": "a" * 40}) + "\n")
    demo_manifest = tmp_path / "demo-manifest.json"
    demo_manifest.write_text(json.dumps({"files": ["README.txt"]}) + "\n")

    return {
        "source_repo": str(source),
        "base_revision": revision,
        "data_mode": "local_only",
        "dataset_root": str(dataset),
        "dataset_source_revision": dataset_revision,
        "dataset_source_descriptor": str(dataset_descriptor),
        "train_manifest": str(train_manifest),
        "development_manifest": str(development_manifest),
        "test_manifest": str(test_manifest),
        "writable_paths": [
            "are/simulation/agents/default_agent",
            "are/simulation/agents/are_simulation_agent_config.py",
            "are/simulation/agents/agent_config_builder.py",
            "are/simulation/apps/agent_user_interface.py",
            "hook.json",
        ],
        "forbidden_paths": [
            "are/simulation/validation",
            "are/simulation/scenarios",
            "are/simulation/benchmark",
            "are/simulation/benchmark.py",
            "are/simulation/tests",
            "are/simulation/data",
            "are/simulation/data_handler",
            "are/simulation/checkpoint",
            "are/simulation/tutorials",
            "are/simulation/gui",
        ],
        "demo_filesystem_root": str(demo),
        "demo_filesystem_source_revision": "a" * 40,
        "demo_filesystem_source_descriptor": str(demo_descriptor),
        "demo_filesystem_manifest": str(demo_manifest),
        "benchmark_config": "search",
        "benchmark_split": "validation",
        "agent": "default",
        "model": "task-model",
        "model_provider": "openai",
        "model_wire_api": "chat_completions",
        "model_endpoint": None,
        "reasoning_effort": None,
        "judge_model": "judge-model",
        "judge_provider": "azure",
        "judge_endpoint": None,
        "scenario_timeout_seconds": 30,
        "process_completion_grace_seconds": 60,
        "verification_timeout_seconds": 30,
        "repetitions": 2,
        "max_concurrent": 1,
        "infrastructure_retries": 1,
        "import_check": "from are.simulation.agents.default_agent import base_agent",
        "capability_phase_iterations": 2,
    }, revision


def test_settings_resolve_pinned_local_sources_and_stable_cases(tmp_path: Path) -> None:
    from autosaddler.v2.plugins.meta_are.config import MetaARESettings

    mapping, revision = _settings_mapping(tmp_path)
    settings = MetaARESettings.from_mapping(mapping, base_dir=tmp_path)
    train, development = settings.load_cases()

    assert settings.base_revision == revision
    assert settings.dataset_source_revision == "b" * 40
    assert settings.dataset_digest.startswith("sha256:")
    assert [case.case_id for case in train] == ["train-a", "train-b"]
    assert [case.case_id for case in development] == ["dev-a"]
    assert train[0].payload["source_sha256"].startswith("sha256:")
    assert settings.test_case_count == 1
    assert settings.test_manifest.name == "test-universe-22.json"
    assert settings.writable_paths[0].as_posix() == "are/simulation/agents/default_agent"
    assert settings.model_wire_api == "chat_completions"
    assert settings.responses_runtime_sha256 is None
    assert settings.execution_fingerprint.startswith("sha256:")


def test_settings_reject_unknown_keys_and_symbolic_revision(tmp_path: Path) -> None:
    from autosaddler.v2.plugins.meta_are.config import MetaARESettings

    mapping, _ = _settings_mapping(tmp_path)
    with pytest.raises(ValueError, match="unknown|unexpected"):
        MetaARESettings.from_mapping({**mapping, "legacy_output_dir": "old"}, base_dir=tmp_path)
    with pytest.raises(ValueError, match="full|symbolic|revision"):
        MetaARESettings.from_mapping({**mapping, "base_revision": "HEAD"}, base_dir=tmp_path)
    with pytest.raises((TypeError, ValueError), match="scenario_timeout_seconds"):
        MetaARESettings.from_mapping(
            {**mapping, "scenario_timeout_seconds": 30.5},
            base_dir=tmp_path,
        )


def test_settings_pin_responses_runtime_and_reject_unsupported_provider(
    tmp_path: Path,
) -> None:
    from autosaddler.v2.plugins.meta_are.config import MetaARESettings

    mapping, _ = _settings_mapping(tmp_path)
    responses_mapping = {
        **mapping,
        "model_provider": "copilot",
        "model_wire_api": "responses",
    }

    settings = MetaARESettings.from_mapping(responses_mapping, base_dir=tmp_path)

    assert settings.responses_runtime_sha256 is not None
    assert settings.responses_runtime_sha256.startswith("sha256:")
    with pytest.raises(ValueError, match="copilot"):
        MetaARESettings.from_mapping(
            {**responses_mapping, "model_provider": "openai"},
            base_dir=tmp_path,
        )
    with pytest.raises(ValueError, match="provider-default reasoning"):
        MetaARESettings.from_mapping(
            {**responses_mapping, "reasoning_effort": "xhigh"},
            base_dir=tmp_path,
        )


def test_settings_reject_split_overlap_and_ambiguous_case_files(tmp_path: Path) -> None:
    from autosaddler.v2.plugins.meta_are.config import MetaARESettings

    mapping, _ = _settings_mapping(tmp_path)
    Path(str(mapping["development_manifest"])).write_text(json.dumps(["train-a"]) + "\n")
    with pytest.raises(ValueError, match="disjoint|overlap"):
        MetaARESettings.from_mapping(mapping, base_dir=tmp_path).load_cases()

    mapping, _ = _settings_mapping(tmp_path / "test-overlap")
    Path(str(mapping["test_manifest"])).write_text(json.dumps(["dev-a"]) + "\n")
    with pytest.raises(ValueError, match="disjoint|overlap"):
        MetaARESettings.from_mapping(mapping, base_dir=tmp_path / "test-overlap").load_cases()

    mapping, _ = _settings_mapping(tmp_path / "ambiguous")
    duplicate = Path(str(mapping["dataset_root"])) / "nested"
    duplicate.mkdir()
    (duplicate / "train-a.json").write_text('{"id":"train-a"}\n')
    with pytest.raises(ValueError, match="exactly one|ambiguous"):
        MetaARESettings.from_mapping(mapping, base_dir=tmp_path / "ambiguous").load_cases()


def test_settings_reject_demo_filesystem_manifest_drift(tmp_path: Path) -> None:
    from autosaddler.v2.plugins.meta_are.config import MetaARESettings

    mapping, _ = _settings_mapping(tmp_path)
    Path(str(mapping["demo_filesystem_manifest"])).write_text(
        json.dumps({"files": [{"path": "README.txt", "sha256": "sha256:" + "0" * 64}]}) + "\n"
    )

    with pytest.raises(ValueError, match="demo|digest|manifest"):
        MetaARESettings.from_mapping(mapping, base_dir=tmp_path)


def test_settings_reject_dataset_source_drift(tmp_path: Path) -> None:
    from autosaddler.v2.plugins.meta_are.config import MetaARESettings

    mapping, _ = _settings_mapping(tmp_path)
    dataset = Path(str(mapping["dataset_root"]))
    (dataset / "train-a.json").write_text('{"id":"changed"}\n')

    with pytest.raises(ValueError, match="dataset source|digest drift"):
        MetaARESettings.from_mapping(mapping, base_dir=tmp_path)

    mapping, _ = _settings_mapping(tmp_path / "revision")
    with pytest.raises(ValueError, match="dataset source revision"):
        MetaARESettings.from_mapping(
            {**mapping, "dataset_source_revision": "c" * 40},
            base_dir=tmp_path / "revision",
        )


def test_settings_reject_mutation_scope_expansion(tmp_path: Path) -> None:
    from autosaddler.v2.plugins.meta_are.config import MetaARESettings

    mapping, _ = _settings_mapping(tmp_path)
    writable_paths = list(mapping["writable_paths"])
    writable_paths.append("are/simulation/scenarios")

    with pytest.raises(ValueError, match="writable|mutation scope"):
        MetaARESettings.from_mapping(
            {**mapping, "writable_paths": writable_paths},
            base_dir=tmp_path,
        )