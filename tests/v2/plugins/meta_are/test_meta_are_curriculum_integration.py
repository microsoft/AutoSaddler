from __future__ import annotations

import json
from pathlib import Path

import yaml

from autosaddler.v2.config.registry import build_runtime, default_registry
from autosaddler.v2.core.curriculum import CurriculumState
from autosaddler.v2.core.domain import Cost, canonical_json
from autosaddler.v2.prompting.models import SessionRequest, SessionResult
from test_meta_are_plugin_integration import (
    ScriptedMetaAREProvider,
    ScriptedMetaARERunner,
    _write_integration_fixture,
)


class ScriptedCurriculumMetaAREProvider(ScriptedMetaAREProvider):
    async def run(self, request: SessionRequest) -> SessionResult:
        if request.spec.kind != "extract_patterns":
            return await super().run(request)
        self.calls.append(request.spec.kind)
        context = json.loads(request.spec.workspace_files[".autosaddler/session_context.json"])
        failures = [
            (source, item["case_id"])
            for source, key in (("pre_patch", "pre_patch_failures"), ("post_patch", "post_patch_failures"))
            for item in context[key]
        ]
        output = {
            "schema_version": "autosaddler-meta-are-pattern-extraction/v1",
            "symptoms": [
                {
                    "case_id": case_id,
                    "source": source,
                    "root_cause": "The fixture capability is disabled.",
                    "symptom": "A required capability is unavailable to the agent.",
                    "rationale": "The trace never reaches the capability.",
                }
                for source, case_id in failures
            ],
            "new_patterns": [{"key": "disabled-capability", "label": "Required capability unavailable"}],
            "tags": [
                {
                    "case_id": case_id,
                    "source": source,
                    "pattern_refs": ["disabled-capability"],
                    "root_cause": "The fixture capability is disabled.",
                }
                for source, case_id in failures
            ],
        }
        return SessionResult(
            status="completed",
            structured_output=output,
            raw_response=canonical_json(output),
            tool_calls=(),
            usage=(),
            cost=Cost(sessions=1),
        )


def _curriculum_fixture(tmp_path: Path) -> Path:
    config_path = _write_integration_fixture(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["scenario"]["settings"].update(
        {
            "capability_phase_iterations": 0,
            "capability_transition_mode": "full_coverage",
            "capability_phase_max_iterations": 0,
        }
    )
    config["optimization"]["task_selection"] = {
        "type": "activesaddler",
        "batch_size": 1,
        "seed": 0,
        "settings": {
            "softmax_temperature": 0.15,
            "min_prob": 0.0,
            "ema_eta": 0.9,
            "pattern_extraction_timeout_seconds": 30,
            "arm_scoring_timeout_seconds": 30,
        },
    }
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path


def test_meta_are_curriculum_extracts_patterns_from_matched_training_evidence(tmp_path: Path) -> None:
    config_path = _curriculum_fixture(tmp_path)
    provider = ScriptedCurriculumMetaAREProvider()
    runner = ScriptedMetaARERunner()
    registry = default_registry()
    registry.providers["scripted_meta_are"] = lambda **_kwargs: provider

    runtime = build_runtime(config_path, run_id="meta-are-curriculum", registry=registry)
    runtime.scenario.evaluator.runner = runner
    result = runtime.engine.run()
    store = runtime.store

    assert result.development_score == 1.0
    assert provider.calls == ["evolve", "diagnose_patch", "reflect", "extract_patterns"]
    (batch,) = store.events_of_type("BatchSampled")
    assert batch.payload["provenance"]["action"] == "unseen_draw"
    requests = {
        request["spec"]["kind"]: request
        for request in (
            store.read_json(str(event.payload["request"]["uri"])) for event in store.events_of_type("SessionStarted")
        )
    }
    diagnosis_files = requests["diagnose_patch"]["spec"]["workspace_files"]
    diagnosis_context = json.loads(diagnosis_files[".autosaddler/session_context.json"])
    assert diagnosis_context["patch_phase"] == "capability"
    assert diagnosis_context["task_selection"]["sampling_action"] == "unseen_draw"
    assert ".autosaddler/curriculum/manifest.json" in diagnosis_files
    assert "Failure-Pattern Curriculum Context" in requests["diagnose_patch"]["spec"]["task_prompt"]
    extraction_files = requests["extract_patterns"]["spec"]["workspace_files"]
    before = json.loads(extraction_files[".autosaddler/training_evidence_before.json"])
    after = json.loads(extraction_files[".autosaddler/training_evidence_after.json"])
    assert (before["purpose"], after["purpose"]) == ("train_before", "train_after")
    (pattern,) = CurriculumState.replay(store.events()).patterns.values()
    assert pattern.label == "Required capability unavailable"
    assert pattern.case_ids == ("train-a",)
    assert runtime.scenario.prompt_pack.patch_phase(1) == "steering"
    compositions = store.read_json("resolved/prompts/curriculum/compositions.json")
    assert "extract_patterns" in compositions["compositions"]
    assert set(store.read_json("resolved/prompts/compositions.json")["compositions"]) == {
        "evolve",
        "diagnose_patch.capability",
        "diagnose_patch.steering",
        "reflect",
    }
    assert store.read_json("resolved/mutation_scope.json")["capability_transition_mode"] == "full_coverage"

    resumed = build_runtime(config_path, run_id="meta-are-curriculum", registry=registry)
    resumed.scenario.evaluator.runner = runner
    assert resumed.engine.run() == result
    assert provider.calls == ["evolve", "diagnose_patch", "reflect", "extract_patterns"]


def test_meta_are_epoch_run_records_no_curriculum_provenance(tmp_path: Path) -> None:
    config_path = _write_integration_fixture(tmp_path)
    provider = ScriptedMetaAREProvider()
    registry = default_registry()
    registry.providers["scripted_meta_are"] = lambda **_kwargs: provider

    runtime = build_runtime(config_path, run_id="meta-are-epoch", registry=registry)

    assert not (runtime.store.run_dir / "resolved/prompts/curriculum").exists()
    assert "capability_transition_mode" not in runtime.store.read_json("resolved/mutation_scope.json")
