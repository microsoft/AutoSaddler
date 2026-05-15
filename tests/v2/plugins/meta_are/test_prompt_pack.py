from __future__ import annotations

import json
from pathlib import Path, PurePosixPath

from autosaddler.v2.core.domain import sha256_digest
from autosaddler.v2.storage.local import LocalRunStore


def test_prompt_pack_renders_phase_and_closed_diagnosis_contract(tmp_path: Path) -> None:
    from autosaddler.v2.plugins.meta_are.prompt_pack import MetaAREPromptPack

    store = _store(tmp_path)
    evidence = store.write_json(
        "evidence/train/evidence.json",
        {"schema_version": "autosaddler-meta-are-evidence/v1", "case_records": []},
        kind="meta-are-training-evidence",
    )
    pack = MetaAREPromptPack(
        store=store,
        writable_paths=(
            PurePosixPath("are/simulation/agents/default_agent"),
            PurePosixPath("hook.json"),
        ),
        capability_phase_iterations=1,
    )
    candidate_id = sha256_digest("candidate")

    specification = pack.session(
        "diagnose_patch",
        {
            "iteration": 0,
            "candidate_ids": [candidate_id],
            "train_case_ids": ["train-a"],
            "evidence": {
                "uri": evidence.uri,
                "kind": evidence.kind,
                "sha256": evidence.sha256,
                "bytes": evidence.bytes,
            },
        },
    )

    assert specification.output_schema["additionalProperties"] is False
    assert specification.output_schema["required"] == [
        "schema_version",
        "intent",
        "diagnosis",
        "expected_effect",
        "changed_paths",
    ]
    context = json.loads(specification.workspace_files[".autosaddler/session_context.json"])
    assert context["patch_phase"] == "capability"
    assert context["mutation_scope"] == [
        "are/simulation/agents/default_agent",
        "hook.json",
    ]
    assert ".autosaddler/training_evidence.json" in specification.workspace_files
    assert ".autosaddler/history/manifest.json" in specification.workspace_files
    history_manifest = json.loads(specification.workspace_files[".autosaddler/history/manifest.json"])
    assert history_manifest["schema_version"] == "autosaddler-history-manifest/v1"
    assert history_manifest["through_sequence"] == 0
    prompt_manifest = json.loads(specification.workspace_files[".autosaddler/prompt_assets.json"])
    assert prompt_manifest["schema_version"] == "autosaddler-prompt-assets/v1"
    assert {asset["asset_id"] for asset in prompt_manifest["assets"]} >= {
        "methodology.system.invariants",
        "methodology.prompt.diagnose_patch",
        "meta_are.system",
        "meta_are.prompt.diagnose_patch",
    }
    assert "GAIA2" in specification.system_context
    assert "earliest divergence" in specification.task_prompt
    assert "edit" in specification.task_prompt.lower()
    assert "current workspace" in specification.task_prompt.lower()
    assert set(specification.skills) == {
        "history-analysis",
        "diagnose",
        "capability-patch",
        "patch-verification",
    }


def test_prompt_pack_keeps_evolve_and_reflection_provider_neutral(tmp_path: Path) -> None:
    from autosaddler.v2.plugins.meta_are.prompt_pack import MetaAREPromptPack

    store = _store(tmp_path)
    parents = [sha256_digest("parent-a"), sha256_digest("parent-b")]
    pack = MetaAREPromptPack(
        store=store,
        writable_paths=(PurePosixPath("are/simulation/agents/default_agent"),),
        capability_phase_iterations=1,
    )

    evolve = pack.session(
        "evolve",
        {
            "iteration": 1,
            "candidate_ids": parents,
            "component_source_options": {"hook.json": [parents[1]]},
            "train_case_ids": ["train-a"],
        },
    )
    assert evolve.output_schema["additionalProperties"] is False
    assert evolve.output_schema["properties"]["parent_ids"]["items"]["enum"] == parents
    component_sources = evolve.output_schema["properties"]["component_sources"]
    assert component_sources["properties"] == {
        "hook.json": {"type": "string", "enum": [parents[1]]},
    }
    assert component_sources["additionalProperties"] is False
    assert "development case" not in evolve.task_prompt.lower()
    assert "Default to one measured base parent" in evolve.task_prompt

    reflect = pack.session(
        "reflect",
        {
            "iteration": 1,
            "candidate_ids": [parents[0]],
            "train_case_ids": ["train-a"],
            "development_outcome": {"aggregate_score": 0.5},
        },
    )
    assert reflect.output_schema["additionalProperties"] is False
    assert reflect.output_schema["required"] == ["schema_version", "lessons"]
    assert ".autosaddler/history/current_batch.json" in evolve.workspace_files
    assert ".autosaddler/history/current_batch.json" in reflect.workspace_files

    rendered = "\n".join(
        [
            evolve.task_prompt,
            reflect.task_prompt,
            *evolve.workspace_files.values(),
            *reflect.workspace_files.values(),
        ]
    )
    assert "CLAUDE.md" not in rendered
    assert "gc_sdk" not in rendered
    assert "direct EvoDAG" not in rendered


def _store(tmp_path: Path) -> LocalRunStore:
    store = LocalRunStore(run_dir=tmp_path / "run", run_id="run")
    store.initialize(
        resolved_config={"schema_version": "autosaddler/v2"},
        resolved_entities={"resolved/component_graph.json": {"scenario": "meta_are"}},
    )
    return store
