from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from autosaddler.v2.config.registry import build_runtime
from autosaddler.v2.core.curriculum import CURRICULUM_NAMESPACE, CurriculumState
from autosaddler.v2.core.domain import Cost, canonical_json
from autosaddler.v2.core.engine import SessionRetriesExhausted
from autosaddler.v2.core.ports import BASE_SESSION_KINDS
from autosaddler.v2.prompting.models import SessionResult
from autosaddler.v2.providers.fake import FakeAgentProvider
from autosaddler.v2.storage.local import LocalRunStore


def write_config(root: Path, value: dict) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=False))
    return path


def curriculum_changes(store: LocalRunStore) -> list[dict]:
    return [
        dict(event.payload)
        for event in store.events_of_type("ExtensionStateChanged")
        if event.payload.get("namespace") == CURRICULUM_NAMESPACE
    ]


def session_stages(store: LocalRunStore) -> list[str]:
    return [str(event.payload.get("stage")) for event in store.events_of_type("SessionStarted")]


class ScriptedCurriculumProvider:
    """Fake provider that can fail chosen kinds and build on the newest accepted candidate."""

    def __init__(self, ledger, *failing_kinds: str, latest_parent: bool = False) -> None:
        self.delegate = FakeAgentProvider(ledger)
        self.failing_kinds = frozenset(failing_kinds)
        self.latest_parent = latest_parent

    async def run(self, request):
        if request.spec.kind in self.failing_kinds:
            return SessionResult(
                status="failed",
                structured_output=None,
                raw_response="",
                tool_calls=(),
                usage=(),
                cost=Cost(sessions=1),
                error=f"persistent {request.spec.kind} failure",
            )
        if self.latest_parent and request.spec.kind == "evolve":
            context = json.loads(request.spec.workspace_files["session_context.json"])
            response = {
                "schema_version": "autosaddler-evolution/v1",
                "parent_ids": [context["candidate_ids"][-1]],
                "component_sources": {},
                "rationale": "Build on the newest accepted candidate.",
            }
            files = dict(request.spec.workspace_files)
            files[".autosaddler/fake_response.json"] = canonical_json(response) + "\n"
            request = replace(request, spec=replace(request.spec, workspace_files=files))
        return await self.delegate.run(request)


def scripted_runtime(tmp_path, registry, config, *failing_kinds: str, latest_parent: bool = False):
    registry.providers["fake"] = lambda *, ledger, settings: ScriptedCurriculumProvider(
        ledger,
        *failing_kinds,
        latest_parent=latest_parent,
    )
    value = config(tmp_path)
    value["optimization"]["session_retries"] = 0
    return build_runtime(write_config(tmp_path, value), run_id="scripted", registry=registry)


def test_activesaddler_runs_through_the_adaptive_task_selection_interface(
    tmp_path: Path,
    curriculum_registry,
    activesaddler_config,
) -> None:
    runtime = build_runtime(
        write_config(tmp_path, activesaddler_config(tmp_path)),
        run_id="curriculum",
        registry=curriculum_registry,
    )
    result = runtime.engine.run()
    store = runtime.store

    assert result.development_score == 1.0
    assert session_stages(store) == [
        "proposal.selection",
        "proposal.patch",
        "proposal.reflection",
        "proposal.pattern_extraction",
        "proposal.selection",
        "proposal.arm_decision",
        "proposal.arm_scoring",
        "proposal.patch",
        "proposal.reflection",
        "proposal.pattern_extraction",
    ]
    batches = store.events_of_type("BatchSampled")
    assert [event.payload["provenance"]["action"] for event in batches] == ["unseen_draw", "arm_pull"]
    first_batch = list(batches[0].payload["case_ids"])
    assert batches[1].payload["case_ids"] == first_batch
    assert [change["change"] for change in curriculum_changes(store)] == [
        "arm_decision",
        "patterns_extracted",
        "observations_recorded",
        "arm_decision",
        "arm_scores",
        "patterns_extracted",
        "observations_recorded",
    ]
    operations = [
        event.operation_id.split(":", 1)[1]
        for event in store.events_of_type("ExtensionStateChanged")
        if event.payload.get("namespace") == CURRICULUM_NAMESPACE
    ]
    assert operations[0] == "iteration:0:selection:arm-decision"
    assert operations[3:5] == ["iteration:1:selection:decide-arm", "iteration:1:selection:score-arms"]
    assert curriculum_changes(store)[0]["requested_action"] is None

    state = CurriculumState.replay(store.events())
    (pattern,) = state.patterns.values()
    assert pattern.case_ids == tuple(first_batch)
    assert {tag.source for tag in pattern.tags} == {"pre_patch"}
    assert [observation.active for observation in pattern.observations] == [0.0, 0.0]
    assert batches[1].payload["provenance"]["chosen_arm"] == pattern.pattern_id

    timeouts = {}
    for started in store.events_of_type("SessionStarted"):
        request = store.read_json(str(started.payload["request"]["uri"]))
        timeouts[request["spec"]["kind"]] = request["timeout_seconds"]
    assert timeouts == {
        "evolve": 10.0,
        "diagnose_patch": 10.0,
        "reflect": 10.0,
        "extract_patterns": 11.0,
        "decide_arm": 12.0,
        "score_arms": 12.0,
    }
    evolve_request = store.read_json(str(store.events_of_type("SessionStarted")[0].payload["request"]["uri"]))
    evolve_context = json.loads(evolve_request["spec"]["workspace_files"]["session_context.json"])
    assert evolve_context["train_case_ids"] == []
    assert evolve_context["task_selection"]["policy"] == "activesaddler"
    policies = json.loads((store.run_dir / "resolved/policies.json").read_text())
    assert policies["task_selection_settings"]["arm_scoring_timeout_seconds"] == 12
    kinds = json.loads((store.run_dir / "resolved/schemas/session_outputs.json").read_text())["kinds"]
    assert kinds == ["evolve", "diagnose_patch", "reflect", "decide_arm", "extract_patterns", "score_arms"]
    resolved_config = yaml.safe_load((store.run_dir / "resolved_config.yaml").read_text())
    assert resolved_config["optimization"]["task_selection"]["settings"]["ema_eta"] == 0.9
    projection = json.loads((store.run_dir / "strategy/curriculum.json").read_text())
    assert projection["namespace"] == CURRICULUM_NAMESPACE and len(projection["changes"]) == 7
    assert sum(entry["kind"] == "session" for entry in runtime.ledger.entries()) == 10


def test_all_pass_pull_records_inactive_observation_without_diagnosis(
    tmp_path: Path,
    curriculum_registry,
    activesaddler_config,
) -> None:
    runtime = scripted_runtime(tmp_path, curriculum_registry, activesaddler_config, latest_parent=True)

    runtime.engine.run()
    store = runtime.store

    outcomes = [event.payload["outcome"] for event in store.events_of_type("IterationCompleted")]
    assert outcomes == ["accepted", "no_training_failures"]
    observations = [change for change in curriculum_changes(store) if change["change"] == "observations_recorded"]
    assert [change["iteration"] for change in observations] == [0, 1]
    assert observations[1]["observations"][0]["active"] == 0.0
    assert observations[1]["observations"][0]["tagged_case_ids"] == []
    assert session_stages(store).count("proposal.patch") == 1


def test_failed_arm_decision_defaults_to_pull(tmp_path: Path, curriculum_registry, activesaddler_config) -> None:
    runtime = scripted_runtime(tmp_path, curriculum_registry, activesaddler_config, "decide_arm", latest_parent=True)

    runtime.engine.run()

    decision = next(
        change
        for change in curriculum_changes(runtime.store)
        if change["change"] == "arm_decision" and change["iteration"] == 1
    )
    assert decision["requested_action"] == "pull"
    assert decision["action"] == "arm_pull"
    assert "decide_arm failure" in decision["fallback_reason"]
    assert "proposal.arm_scoring" in session_stages(runtime.store)


def test_failed_arm_scoring_fails_the_run(tmp_path: Path, curriculum_registry, activesaddler_config) -> None:
    runtime = scripted_runtime(tmp_path, curriculum_registry, activesaddler_config, "score_arms", latest_parent=True)

    with pytest.raises(SessionRetriesExhausted):
        runtime.engine.run()

    assert runtime.store.events()[-1].event_type == "RunFailed"
    assert len(runtime.store.events_of_type("BatchSampled")) == 1


def test_failed_pattern_extraction_is_abandoned_and_next_iteration_draws(
    tmp_path: Path,
    curriculum_registry,
    activesaddler_config,
) -> None:
    runtime = scripted_runtime(
        tmp_path,
        curriculum_registry,
        activesaddler_config,
        "extract_patterns",
        latest_parent=True,
    )

    runtime.engine.run()
    store = runtime.store

    assert store.events_of_type("DeferredWorkAbandoned")
    assert CurriculumState.replay(store.events()).patterns == {}
    actions = [event.payload["provenance"]["action"] for event in store.events_of_type("BatchSampled")]
    assert actions == ["unseen_draw", "unseen_draw"]
    assert "proposal.arm_decision" not in session_stages(store)


def test_runtime_rejects_scenarios_without_curriculum_session_kinds(
    tmp_path: Path,
    curriculum_registry,
    activesaddler_config,
) -> None:
    curriculum_factory = curriculum_registry.scenarios["fake"]
    curriculum_registry.scenarios["fake"] = lambda **kwargs: replace(
        curriculum_factory(**kwargs),
        supported_session_kinds=BASE_SESSION_KINDS,
    )

    with pytest.raises(ValueError, match="does not support session kinds"):
        build_runtime(
            write_config(tmp_path, activesaddler_config(tmp_path)),
            run_id="unsupported",
            registry=curriculum_registry,
        )


def test_builtin_fake_scenario_does_not_declare_curriculum_kinds(tmp_path: Path, activesaddler_config) -> None:
    with pytest.raises(ValueError, match="does not support session kinds"):
        build_runtime(write_config(tmp_path, activesaddler_config(tmp_path)), run_id="builtin")
