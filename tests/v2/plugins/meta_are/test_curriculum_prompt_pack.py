from __future__ import annotations

import json
from pathlib import Path, PurePosixPath

import pytest

from autosaddler.v2.core.domain import sha256_digest
from autosaddler.v2.storage.local import LocalRunStore


def _store(tmp_path: Path) -> LocalRunStore:
    store = LocalRunStore(run_dir=tmp_path / "run", run_id="run")
    store.initialize(
        resolved_config={"schema_version": "autosaddler/v2"},
        resolved_entities={"resolved/component_graph.json": {"scenario": "meta_are"}},
    )
    return store


def _curriculum_pack(tmp_path: Path, **overrides):
    from autosaddler.v2.plugins.meta_are.prompt_pack import MetaAREPromptPack

    values = {
        "store": _store(tmp_path),
        "writable_paths": (PurePosixPath("are/simulation/agents/default_agent"),),
        "capability_phase_iterations": 0,
        "capability_transition_mode": "full_coverage",
        "capability_phase_max_iterations": 0,
        "train_case_ids": ("train-a", "train-b", "train-c"),
    }
    values.update(overrides)
    return MetaAREPromptPack(**values)


def _sample(store: LocalRunStore, iteration: int, case_ids: list[str]) -> None:
    store.append(
        "BatchSampled",
        f"run:iteration:{iteration}:batch",
        {"iteration": iteration, "case_ids": case_ids, "provenance": {"policy": "activesaddler"}},
    )


def test_full_coverage_phase_switches_after_the_covering_iteration(tmp_path: Path) -> None:
    pack = _curriculum_pack(tmp_path)

    _sample(pack.store, 0, ["train-a", "train-b"])
    _sample(pack.store, 1, ["train-a"])
    _sample(pack.store, 2, ["train-c"])

    assert [pack.patch_phase(iteration) for iteration in range(5)] == [
        "capability",
        "capability",
        "capability",
        "steering",
        "steering",
    ]


def test_full_coverage_phase_honors_max_iteration_safety_valve(tmp_path: Path) -> None:
    pack = _curriculum_pack(tmp_path, capability_phase_max_iterations=2)

    _sample(pack.store, 0, ["train-a"])

    assert [pack.patch_phase(iteration) for iteration in range(4)] == [
        "capability",
        "capability",
        "steering",
        "steering",
    ]


def test_prompt_pack_renders_curriculum_sessions(tmp_path: Path) -> None:
    pack = _curriculum_pack(tmp_path)
    store = pack.store
    evidence = {
        name: store.write_json(
            f"evidence/{name}/evidence.json",
            {"schema_version": "autosaddler-meta-are-evidence/v1", "case_records": []},
            kind="meta-are-training-evidence",
        )
        for name in ("before", "after")
    }
    parent = sha256_digest("parent")
    curriculum = {
        "policy": "activesaddler",
        "batch_size": 2,
        "softmax_temperature": 0.15,
        "min_prob": 0.0,
        "ema_eta": 0.9,
    }

    extraction = pack.session(
        "extract_patterns",
        {
            "iteration": 0,
            "candidate_ids": [parent],
            "train_case_ids": ["train-a", "train-b"],
            "task_selection": {
                **curriculum,
                "train_before_evidence": {"uri": evidence["before"].uri, "sha256": evidence["before"].sha256},
                "train_after_evidence": {"uri": evidence["after"].uri, "sha256": evidence["after"].sha256},
            },
            "pre_patch_failures": [{"case_id": "train-a", "train_before_score": 0.0}],
            "post_patch_failures": [{"case_id": "train-b", "train_after_score": 0.0}],
            "existing_pattern_ids": [],
        },
    )
    assert set(extraction.skills) == {"history-analysis", "symptom-extract", "symptom-normalize"}
    assert {".autosaddler/training_evidence_before.json", ".autosaddler/training_evidence_after.json"} <= set(
        extraction.workspace_files
    )
    assert ".autosaddler/curriculum/manifest.json" in extraction.workspace_files
    tag_case = extraction.output_schema["properties"]["tags"]["items"]["properties"]["case_id"]
    assert tag_case["enum"] == ["train-a", "train-b"]
    assert "symptom_candidates.md" in extraction.task_prompt
    assert extraction.mutation_label is None

    scoring = pack.session(
        "score_arms",
        {"iteration": 1, "candidate_ids": [parent], "task_selection": {**curriculum, "arm_ids": ["pattern-a"]}},
    )
    assert set(scoring.skills) == {"history-analysis", "progress-scoring"}
    assert scoring.output_schema["properties"]["scores"]["minItems"] == 1
    assert "Rate Four Axes" in scoring.task_prompt

    decision = pack.session(
        "decide_arm",
        {"iteration": 1, "candidate_ids": [parent], "task_selection": {**curriculum, "arm_ids": ["pattern-a"]}},
    )
    assert decision.output_schema["properties"]["action"]["enum"] == ["pull", "draw"]
    assert set(decision.skills) == {"history-analysis"}

    evolve = pack.session(
        "evolve",
        {"iteration": 1, "candidate_ids": [parent], "train_case_ids": [], "task_selection": curriculum},
    )
    plain = pack.session("evolve", {"iteration": 1, "candidate_ids": [parent], "train_case_ids": []})
    assert "Failure-Pattern Curriculum Context" in evolve.task_prompt
    assert "Failure-Pattern Curriculum Context" not in plain.task_prompt
    assert ".autosaddler/curriculum/manifest.json" not in plain.workspace_files


def test_curriculum_context_is_appended_only_for_activesaddler(tmp_path: Path) -> None:
    pack = _curriculum_pack(tmp_path, capability_transition_mode="iterations", capability_phase_iterations=1)
    parent = sha256_digest("parent")

    passive = pack.session("evolve", {"iteration": 0, "candidate_ids": [parent], "train_case_ids": ["train-a"]})
    other = pack.session(
        "evolve",
        {"iteration": 0, "candidate_ids": [parent], "train_case_ids": [], "task_selection": {"policy": "other"}},
    )

    assert "Failure-Pattern Curriculum Context" not in passive.task_prompt
    assert passive.task_prompt == other.task_prompt
    with pytest.raises(ValueError, match="Unknown Meta-ARE session kind"):
        pack.session("decide_arm", {"iteration": 0, "candidate_ids": [parent]})


def test_curriculum_prompt_provenance_is_recorded_separately() -> None:
    from autosaddler.v2.plugins.meta_are.prompt_pack import meta_are_prompt_composition_entity
    from autosaddler.v2.prompting.assets import prompt_source_entities
    from autosaddler.v2.plugins.meta_are.prompt_pack import meta_are_curriculum_composition_entity
    from autosaddler.v2.prompting.curriculum import curriculum_prompt_source_entities
    import autosaddler.v2.plugins.meta_are.plugin as plugin

    base = {
        **prompt_source_entities(plugin_root=Path(plugin.__file__).parent, plugin_name="meta_are", exclude=("curriculum",)),
        "resolved/prompts/compositions.json": meta_are_prompt_composition_entity(),
    }
    curriculum = curriculum_prompt_source_entities(plugin_root=Path(plugin.__file__).parent, plugin_name="meta_are")
    compositions = meta_are_curriculum_composition_entity()

    assert not any("curriculum" in path for path in base)
    assert set(json.loads(json.dumps(base["resolved/prompts/compositions.json"]))["compositions"]) == {
        "evolve",
        "diagnose_patch.capability",
        "diagnose_patch.steering",
        "reflect",
    }
    sources = {asset["source"] for asset in curriculum["resolved/prompts/curriculum/assets.json"]["assets"]}
    assert "shared/curriculum_methodology/skills/progress-scoring/SKILL.md" in sources
    assert "plugins/meta_are/curriculum/prompts/score_arms.md" in sources
    assert all(path.startswith("resolved/prompts/curriculum/") for path in curriculum)
    assert set(compositions["compositions"]) == {
        "evolve.curriculum",
        "diagnose_patch.capability.curriculum",
        "diagnose_patch.steering.curriculum",
        "extract_patterns",
        "decide_arm",
        "score_arms",
    }
