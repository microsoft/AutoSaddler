from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping
from pathlib import Path

import pytest
import yaml

from autosaddler.v2.config.registry import build_runtime, default_registry
from autosaddler.v2.core.domain import Cost, JsonValue, canonical_json, sha256_digest
from autosaddler.v2.plugins.meta_are.runner import MetaARERunResult
from autosaddler.v2.prompting.models import SessionRequest, SessionResult


class ScriptedMetaAREProvider:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.requests: list[SessionRequest] = []

    async def run(self, request: SessionRequest) -> SessionResult:
        self.calls.append(request.spec.kind)
        self.requests.append(request)
        context = json.loads(request.spec.workspace_files[".autosaddler/session_context.json"])
        if request.spec.kind == "evolve":
            output: Mapping[str, JsonValue] = {
                "schema_version": "autosaddler-meta-are-evolution/v1",
                "parent_ids": [context["candidate_ids"][0]],
                "component_sources": {},
                "rationale": "Use the seed candidate.",
            }
        elif request.spec.kind == "diagnose_patch":
            (request.workspace / "hook.json").write_text(
                json.dumps(
                    {
                        "hooks": {
                            "PreToolUse": [
                                {
                                    "matcher": ".*",
                                    "hooks": [
                                        {
                                            "type": "reminder",
                                            "reminder": "Enable the fixture capability.",
                                        }
                                    ],
                                }
                            ]
                        }
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            output = {
                "schema_version": "autosaddler-meta-are-diagnosis/v1",
                "intent": "Enable the fixture capability.",
                "diagnosis": "The training trace shows the capability is disabled.",
                "expected_effect": "The matched case should pass.",
                "changed_paths": ["hook.json"],
            }
        elif request.spec.kind == "reflect":
            output = {
                "schema_version": "autosaddler-meta-are-reflection/v1",
                "lessons": [
                    {
                        "scope": "component",
                        "statement": "Enable the capability identified by training evidence.",
                        "evidence_case_ids": context["train_case_ids"],
                    }
                ],
            }
        else:
            raise AssertionError(request.spec.kind)
        return SessionResult(
            status="completed",
            structured_output=output,
            raw_response=canonical_json(output),
            tool_calls=(),
            usage=(),
            cost=Cost(sessions=1),
        )


class ScriptedMetaARERunner:
    def __init__(
        self,
        *,
        invalid_attempts: int = 0,
        missing_attempts: int = 0,
        first_attempt_missing_case_ids: tuple[str, ...] = (),
        first_attempt_missing_keys: tuple[tuple[str, int], ...] = (),
        cost_only: bool = False,
    ) -> None:
        self.calls: list[tuple[str, tuple[str, ...]]] = []
        self.execution_settings: list[tuple[int, int | None]] = []
        self.invalid_attempts = invalid_attempts
        self.missing_attempts = missing_attempts
        self.first_attempt_missing_case_ids = frozenset(first_attempt_missing_case_ids)
        self.first_attempt_missing_keys = frozenset(first_attempt_missing_keys)
        self.cost_only = cost_only

    def run(self, *, materialized, cases, repetitions, attempt_dir, max_concurrent=None) -> MetaARERunResult:
        try:
            hook_config = json.loads((materialized.root / "hook.json").read_text(encoding="utf-8"))
            enabled = bool(hook_config.get("hooks", {}).get("PreToolUse", []))
            passed = enabled or self.cost_only
            output_tokens = 3 if enabled or not self.cost_only else 6
            output_dir = attempt_dir / "output"
            hf_dir = output_dir / "hf"
            lite_dir = output_dir / "lite"
            raw_results = output_dir / "output.jsonl"
            stdout = attempt_dir / "stdout.log"
            stderr = attempt_dir / "stderr.log"
            if raw_results.is_file():
                return MetaARERunResult(
                    raw_results=raw_results,
                    hf_trace_dir=hf_dir,
                    lite_trace_dir=lite_dir,
                )
            hf_dir.mkdir(parents=True)
            lite_dir.mkdir(parents=True)
            invalid = len(self.calls) < self.invalid_attempts
            missing = len(self.calls) < self.missing_attempts
            rows = []
            for case in cases:
                for repetition in range(repetitions):
                    if (
                        missing
                        or (not self.calls and case.case_id in self.first_attempt_missing_case_ids)
                        or (not self.calls and (case.case_id, repetition) in self.first_attempt_missing_keys)
                    ):
                        continue
                    run_number = repetition + 1
                    rows.append(
                        {
                            "task_id": case.case_id,
                            "score": 1.0 if passed else 0.0,
                            "metadata": {
                                "scenario_id": case.case_id,
                                "run_number": run_number,
                                "status": "success" if passed and not invalid else "failed",
                                "has_exception": invalid,
                                "rationale": "capability enabled" if enabled else "capability disabled",
                            },
                        }
                    )
                    (hf_dir / f"{case.case_id}-{run_number}.json").write_text(
                        json.dumps(
                            {
                                "metadata": {
                                    "definition": {
                                        "scenario_id": case.case_id,
                                        "run_number": run_number,
                                        "has_exception": invalid,
                                    },
                                    "annotation": {"validation_decision": "Valid" if passed else "Invalid"},
                                },
                                "events": [],
                                "completed_events": [],
                            }
                        )
                        + "\n",
                        encoding="utf-8",
                    )
                    (lite_dir / f"{case.case_id}-{run_number}.json").write_text(
                        json.dumps(
                            {
                                "scenario_id": case.case_id,
                                "run_number": run_number,
                                "model_id": "task-model",
                                "validation_rationale": ("capability enabled" if enabled else "capability disabled"),
                                "per_agent_interaction_histories": {
                                    "default": [
                                        {"role": "user", "content": "Use the fixture capability"},
                                        {"role": "assistant", "content": str(enabled)},
                                    ]
                                },
                                "per_agent_llm_usage_stats": {
                                    "default": {
                                        "total_llm_calls": 1,
                                        "prompt_tokens": [7],
                                        "cached_input_tokens": [3],
                                        "completion_tokens": [output_tokens],
                                        "total_tokens": [7 + output_tokens],
                                        "reasoning_tokens": [1],
                                        "completion_duration": [0.1],
                                    }
                                },
                            }
                        )
                        + "\n",
                        encoding="utf-8",
                    )
            raw_results.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            (attempt_dir / "completion.json").write_text(
                '{"returncode":0}\n',
                encoding="utf-8",
            )
            stdout.write_text("fixture benchmark complete\n", encoding="utf-8")
            stderr.write_text("", encoding="utf-8")
            self.calls.append((materialized.candidate_id, tuple(case.case_id for case in cases)))
            self.execution_settings.append((repetitions, max_concurrent))
            return MetaARERunResult(
                raw_results=raw_results,
                hf_trace_dir=hf_dir,
                lite_trace_dir=lite_dir,
            )
        finally:
            materialized.release()


class InterruptMetaAREBoundary:
    def __init__(self, boundary: str) -> None:
        self.boundary = boundary
        self.triggered = False

    def __call__(self, event) -> None:
        if self.boundary == "benchmark_started":
            matches = event.event_type == "EvaluationAttemptStarted"
        elif self.boundary == "benchmark_observation":
            matches = event.event_type == "EvaluationAttemptCompleted"
        elif self.boundary == "benchmark_usage":
            usage = event.payload.get("usage")
            matches = (
                event.event_type == "ModelUsageObserved"
                and isinstance(usage, Mapping)
                and usage.get("role") == "task_agent"
            )
        elif self.boundary == "provider_delta":
            matches = event.event_type == "SessionCompleted" and str(
                event.payload.get("logical_operation_id", "")
            ).endswith(":diagnose-patch")
        elif self.boundary == "candidate_finalized":
            matches = event.event_type == "CandidateFinalized"
        else:
            raise AssertionError(self.boundary)
        if matches:
            self.triggered = True
            raise RuntimeError(f"interrupt after {self.boundary}")


class CountingVerifier:
    def __init__(self, delegate, calls: list[str]) -> None:
        self.delegate = delegate
        self.calls = calls

    def __call__(self, context):
        self.calls.append(context.patch_label)
        return self.delegate(context)


def test_meta_are_is_registered_with_epoch_shuffled_policy() -> None:
    registry = default_registry()

    assert "meta_are" in registry.scenarios
    assert "epoch_shuffled" in registry.task_selection


def test_registry_rejects_provider_without_meta_are_capabilities(tmp_path: Path) -> None:
    config_path = _write_integration_fixture(tmp_path)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config["provider"] = {
        "type": "fake",
        "capabilities": ["read_workspace"],
        "settings": {},
    }
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="edit_workspace|run_commands|load_skills"):
        build_runtime(config_path, run_id="missing-capabilities")


def test_meta_are_package_data_includes_prompt_and_skill_assets() -> None:
    root = Path(__file__).parents[4] / "src/autosaddler/v2/plugins/meta_are"
    methodology = Path(__file__).parents[4] / "src/autosaddler/v2/prompting/methodology"
    expected = {
        root / "SYSTEM.md",
        root / "prompts/evolve.md",
        root / "prompts/diagnose_patch.md",
        root / "prompts/reflect.md",
        root / "skills/capability-patch/SKILL.md",
        root / "skills/steering-patch/SKILL.md",
        root / "skills/patch-verification/SKILL.md",
        root / "pareto/prompts/diagnose_patch.md",
        root / "pareto/prompts/reflect.md",
        methodology / "pareto/system/optimizer-invariants.md",
        methodology / "pareto/prompts/diagnose-method.md",
        methodology / "pareto/prompts/reflect-method.md",
        methodology / "pareto/skills/history-analysis/SKILL.md",
        methodology / "pareto/skills/causal-diagnosis/SKILL.md",
        methodology / "system/optimizer-invariants.md",
        methodology / "prompts/evolve-method.md",
        methodology / "prompts/diagnose-method.md",
        methodology / "prompts/reflect-method.md",
        methodology / "skills/history-analysis/SKILL.md",
        methodology / "skills/causal-diagnosis/SKILL.md",
        methodology / "skills/verification-baseline/SKILL.md",
    }

    assert all(path.is_file() for path in expected)


def test_configured_meta_are_runs_full_engine_and_resumes_without_paid_work(
    tmp_path: Path,
) -> None:
    config_path = _write_integration_fixture(tmp_path)
    provider = ScriptedMetaAREProvider()
    runner = ScriptedMetaARERunner()
    registry = default_registry()
    registry.providers["scripted_meta_are"] = lambda **_kwargs: provider

    runtime = build_runtime(config_path, run_id="meta-are-integration", registry=registry)
    runtime.scenario.evaluator.runner = runner
    dataset_source = runtime.store.read_json("resolved/sources/dataset.json")
    prompt_assets = runtime.store.read_json("resolved/prompts/assets.json")
    prompt_compositions = runtime.store.read_json("resolved/prompts/compositions.json")
    assert prompt_assets["plugin"] == "meta_are"
    assert set(prompt_compositions["compositions"]) == {
        "evolve",
        "diagnose_patch.capability",
        "diagnose_patch.steering",
        "reflect",
        "pareto.diagnose_patch.capability",
        "pareto.diagnose_patch.steering",
        "pareto.reflect",
        "pareto.evolve.capability",
        "pareto.evolve.steering",
    }
    assert any(asset["source"] == "shared/system/optimizer-invariants.md" for asset in prompt_assets["assets"])
    assert dataset_source["source_revision"] == "b" * 40
    assert dataset_source["source_descriptor"].endswith(".autosaddler-gaia2-source.json")
    assert dataset_source["content_digest"].startswith("sha256:")
    assert dataset_source["test"] == {
        "state": "opaque_to_optimizer",
        "manifest_opened": True,
        "payloads_opened": False,
        "case_count": 1,
    }
    result = runtime.engine.run()

    assert result.development_score == 1.0
    assert provider.calls == ["evolve", "diagnose_patch", "reflect"]
    assert len(runner.calls) == 4
    assert len(runtime.store.events_of_type("EvaluationAttemptStarted")) == 4
    assert len(runtime.store.events_of_type("CandidateFinalized")) == 1
    assert len(runtime.store.events_of_type("AcceptanceDecided")) == 1
    assert len(runtime.store.events_of_type("DevelopmentGateDecided")) == 1
    assert len(runtime.store.events_of_type("DeferredWorkCompleted")) == 1
    assert len(runtime.store.events_of_type("ModelUsageObserved")) == 4
    metrics = runtime.store.read_json("metrics-summary.json")
    assert metrics["model_usage_by_role"]["task_agent"]["model_calls"] == 4
    assert metrics["model_usage_by_role"]["task_agent"]["total_tokens"] == 40
    development = [
        event.payload["evaluation"]
        for event in runtime.store.events_of_type("EvaluationCompleted")
        if event.payload["evaluation"]["split"] == "development"
    ]
    assert all(item["artifact_dir"]["uri"].startswith("quarantine/dev/") for item in development)
    evidence = next((runtime.store.run_dir / "evidence").glob("*/evidence.json"))
    assert json.loads(evidence.read_text(encoding="utf-8"))["case_records"][0]["per_repetition"][0]["interactions"]

    resumed = build_runtime(config_path, run_id="meta-are-integration", registry=registry)
    assert resumed.store.read_json("resolved/prompts/compositions.json") == prompt_compositions
    resumed.scenario.evaluator.runner = runner
    resumed_result = resumed.engine.run()

    assert resumed_result == result
    assert provider.calls == ["evolve", "diagnose_patch", "reflect"]
    assert len(runner.calls) == 4


@pytest.mark.parametrize(
    "boundary",
    [
        "benchmark_usage",
        "benchmark_observation",
        "provider_delta",
        "candidate_finalized",
    ],
)
def test_meta_are_resume_boundaries_do_not_duplicate_paid_work(
    tmp_path: Path,
    boundary: str,
) -> None:
    config_path = _write_integration_fixture(tmp_path)
    provider = ScriptedMetaAREProvider()
    runner = ScriptedMetaARERunner()
    verifier_calls: list[str] = []
    registry = default_registry()
    registry.providers["scripted_meta_are"] = lambda **_kwargs: provider
    interruption = InterruptMetaAREBoundary(boundary)

    interrupted = build_runtime(config_path, run_id="meta-are-fault", registry=registry)
    interrupted.scenario.evaluator.runner = runner
    interrupted.store.transition_hook = interruption
    interrupted.scenario.harness_space.verifier = CountingVerifier(
        interrupted.scenario.harness_space.verifier,
        verifier_calls,
    )
    with pytest.raises(RuntimeError, match=f"interrupt after {boundary}"):
        interrupted.engine.run()
    assert interruption.triggered

    resumed = build_runtime(config_path, run_id="meta-are-fault", registry=registry)
    resumed.scenario.evaluator.runner = runner
    resumed.scenario.harness_space.verifier = CountingVerifier(
        resumed.scenario.harness_space.verifier,
        verifier_calls,
    )
    result = resumed.engine.run()

    assert result.development_score == 1.0
    assert provider.calls == ["evolve", "diagnose_patch", "reflect"]
    assert len(runner.calls) == 4
    assert verifier_calls == ["capability"]
    assert len(resumed.store.events_of_type("EvaluationAttemptStarted")) == 4
    assert len(resumed.store.events_of_type("ModelUsageObserved")) == 4
    assert len(resumed.store.events_of_type("SessionCompleted")) == 3
    assert len(resumed.store.events_of_type("CandidateFinalized")) == 1
    assert (resumed.store.run_dir / "result.json").is_file()


def test_meta_are_starts_attempt_before_launch_and_resumes_interrupted_start(
    tmp_path: Path,
) -> None:
    config_path = _write_integration_fixture(tmp_path)
    provider = ScriptedMetaAREProvider()
    runner = ScriptedMetaARERunner()
    registry = default_registry()
    registry.providers["scripted_meta_are"] = lambda **_kwargs: provider
    interruption = InterruptMetaAREBoundary("benchmark_started")

    interrupted = build_runtime(config_path, run_id="meta-are-prestart", registry=registry)
    interrupted.scenario.evaluator.runner = runner
    interrupted.store.transition_hook = interruption
    with pytest.raises(RuntimeError, match="interrupt after benchmark_started"):
        interrupted.engine.run()

    assert runner.calls == []
    assert len(interrupted.store.events_of_type("EvaluationAttemptStarted")) == 1

    resumed = build_runtime(config_path, run_id="meta-are-prestart", registry=registry)
    resumed.scenario.evaluator.runner = runner
    result = resumed.engine.run()

    assert result.development_score == 1.0
    assert len(runner.calls) == 4
    assert len(resumed.store.events_of_type("EvaluationAttemptStarted")) == 5
    assert any(
        event.payload["error_kind"] == "interrupted"
        for event in resumed.store.events_of_type("EvaluationAttemptFailed")
    )


@pytest.mark.parametrize("timeout_attempts", [1, 2])
def test_priced_timeout_without_usage_retries_only_affected_case_and_is_bounded(
    tmp_path: Path, timeout_attempts: int,
) -> None:
    config_path = _write_integration_fixture(
        tmp_path, infrastructure_retries=1, repetitions=3, priced=True,
        development_case_ids=("dev-a", "dev-b"),
    )
    config = yaml.safe_load(config_path.read_text())
    config["optimization"]["acceptance"] = {"type": "pareto", "parameters": {"draws": 100}}
    config["optimization"]["ranking"] = {"type": "pareto"}
    config["optimization"]["budget"]["max_rollouts"] = 1
    config_path.write_text(yaml.safe_dump(config))

    class TimeoutRunner(ScriptedMetaARERunner):
        def run(self, **kwargs):
            result = super().run(**kwargs)
            if len(self.calls) <= timeout_attempts:
                rows = [json.loads(line) for line in result.raw_results.read_text().splitlines()]
                for row in rows:
                    if row["task_id"] == "dev-a" and row["metadata"]["run_number"] == kwargs["repetitions"]:
                        row["metadata"].update(has_exception=True, exception_type="ScenarioTimeoutError")
                result.raw_results.write_text("".join(json.dumps(row) + "\n" for row in rows))
                for trace_dir in (result.hf_trace_dir, result.lite_trace_dir):
                    (trace_dir / f"dev-a-{kwargs['repetitions']}.json").unlink()
            return result

    runner = TimeoutRunner()
    provider = ScriptedMetaAREProvider()
    registry = default_registry()
    registry.providers["scripted_meta_are"] = lambda **_kwargs: provider
    runtime = build_runtime(config_path, run_id="timeout-cost", registry=registry)
    runtime.scenario.evaluator.runner = runner
    if timeout_attempts == 2:
        with pytest.raises(RuntimeError, match="retained invalid keys after bounded retries"):
            runtime.engine.run()
    else:
        result = runtime.engine.run()
        assert result.iterations == 0
        assert runtime.store.events_of_type("ParetoArchiveUpdated")
    assert [cases for _, cases in runner.calls] == [("dev-a", "dev-b"), ("dev-a",)]
    assert runner.execution_settings == [(3, None), (1, 1)]
    assert provider.calls == []
    attempts = [
        event.payload["observation"]
        for event in runtime.store.events_of_type("EvaluationAttemptCompleted")
        if event.payload["case_id"] == "dev-a" and event.payload["repetition"] == 2
    ]
    assert attempts[0]["disposition"] == "execution_error"
    assert attempts[0]["score"] is None
    assert attempts[0]["cost"]["currency_amount"] is None
    assert attempts[-1]["disposition"] == ("execution_error" if timeout_attempts == 2 else "task_failure")
    if timeout_attempts == 1:
        assert attempts[-1]["cost"]["currency_amount"] > 0


def test_meta_are_retries_invalid_benchmark_keys_and_recovers(tmp_path: Path) -> None:
    config_path = _write_integration_fixture(tmp_path, infrastructure_retries=1)
    provider = ScriptedMetaAREProvider()
    runner = ScriptedMetaARERunner(invalid_attempts=1)
    registry = default_registry()
    registry.providers["scripted_meta_are"] = lambda **_kwargs: provider

    runtime = build_runtime(config_path, run_id="meta-are-retry", registry=registry)
    runtime.scenario.evaluator.runner = runner
    result = runtime.engine.run()

    assert result.development_score == 1.0
    assert len(runner.calls) == 5
    assert len(runtime.store.events_of_type("EvaluationAttemptStarted")) == 5
    assert len(runtime.store.events_of_type("ModelUsageObserved")) == 5
    seed_attempts = [
        event.payload["observation"]
        for event in runtime.store.events_of_type("EvaluationAttemptCompleted")
        if event.payload["case_id"] == "dev-a"
        and event.payload["candidate_id"] == runtime.engine._state().accepted_candidate_ids[0]
    ]
    assert [item["disposition"] for item in seed_attempts] == ["execution_error", "task_failure"]


@pytest.mark.parametrize("repetitions", [2, 3])
def test_pareto_retries_finish_before_iteration_budget_check(tmp_path: Path, repetitions: int) -> None:
    config_path = _write_integration_fixture(tmp_path, infrastructure_retries=1, repetitions=repetitions, priced=True)
    config = yaml.safe_load(config_path.read_text())
    config["optimization"]["acceptance"] = {"type": "pareto", "parameters": {"draws": 100}}
    config["optimization"]["ranking"] = {"type": "pareto"}
    config["optimization"]["budget"]["max_rollouts"] = repetitions
    config_path.write_text(yaml.safe_dump(config))
    runner = ScriptedMetaARERunner(first_attempt_missing_keys=(("dev-a", repetitions - 1),))
    provider = ScriptedMetaAREProvider()
    registry = default_registry()
    registry.providers["scripted_meta_are"] = lambda **_kwargs: provider
    runtime = build_runtime(config_path, run_id="pareto-budget", registry=registry)
    runtime.scenario.evaluator.runner = runner
    result = runtime.engine.run()
    assert result.selected_candidate_id == runtime.engine._state().current_candidate_id
    assert result.development_score == 0.0
    assert result.iterations == 0
    assert len(runner.calls) == 2
    assert provider.calls == []
    assert len(runtime.store.events_of_type("EvaluationAttemptStarted")) == repetitions + 1
    assert len(runtime.store.events_of_type("EvaluationCompleted")) == 1
    assert len(runtime.store.events_of_type("ParetoArchiveUpdated")) == 1


def test_meta_are_fails_when_invalid_benchmark_keys_exhaust_retries(tmp_path: Path) -> None:
    config_path = _write_integration_fixture(tmp_path, infrastructure_retries=1)
    provider = ScriptedMetaAREProvider()
    runner = ScriptedMetaARERunner(invalid_attempts=2)
    registry = default_registry()
    registry.providers["scripted_meta_are"] = lambda **_kwargs: provider

    runtime = build_runtime(config_path, run_id="meta-are-retry-exhausted", registry=registry)
    runtime.scenario.evaluator.runner = runner
    with pytest.raises(RuntimeError, match="invalid keys|bounded retries"):
        runtime.engine.run()

    assert len(runner.calls) == 2
    assert provider.calls == []
    assert len(runtime.store.events_of_type("EvaluationAttemptStarted")) == 2


def test_meta_are_retries_incomplete_result_coverage_and_recovers(tmp_path: Path) -> None:
    config_path = _write_integration_fixture(tmp_path, infrastructure_retries=1)
    provider = ScriptedMetaAREProvider()
    runner = ScriptedMetaARERunner(missing_attempts=1)
    registry = default_registry()
    registry.providers["scripted_meta_are"] = lambda **_kwargs: provider

    runtime = build_runtime(config_path, run_id="meta-are-missing", registry=registry)
    runtime.scenario.evaluator.runner = runner
    result = runtime.engine.run()

    assert result.development_score == 1.0
    assert len(runner.calls) == 5
    assert len(runtime.store.events_of_type("EvaluationAttemptStarted")) == 5
    assert any(
        event.payload["error_kind"] == "incomplete_result"
        for event in runtime.store.events_of_type("EvaluationAttemptFailed")
    )


def test_meta_are_preserves_partial_batch_results_and_retries_only_missing_case(
    tmp_path: Path,
) -> None:
    config_path = _write_integration_fixture(
        tmp_path,
        infrastructure_retries=1,
        development_case_ids=("dev-a", "dev-b", "dev-c"),
    )
    provider = ScriptedMetaAREProvider()
    runner = ScriptedMetaARERunner(first_attempt_missing_case_ids=("dev-c",))
    registry = default_registry()
    registry.providers["scripted_meta_are"] = lambda **_kwargs: provider

    runtime = build_runtime(config_path, run_id="meta-are-partial", registry=registry)
    runtime.scenario.evaluator.runner = runner
    result = runtime.engine.run()

    assert result.development_score == 1.0
    assert runner.calls[0][1] == ("dev-a", "dev-b", "dev-c")
    assert runner.calls[1][1] == ("dev-c",)
    seed_candidate_id = runtime.engine._state().accepted_candidate_ids[0]
    seed_starts = [
        event
        for event in runtime.store.events_of_type("EvaluationAttemptStarted")
        if event.payload["candidate_id"] == seed_candidate_id
        and event.payload["case_id"] in {"dev-a", "dev-b", "dev-c"}
    ]
    assert [event.payload["case_id"] for event in seed_starts] == [
        "dev-a",
        "dev-b",
        "dev-c",
        "dev-c",
    ]
    seed_completed = [
        event
        for event in runtime.store.events_of_type("EvaluationAttemptCompleted")
        if event.payload["candidate_id"] == seed_candidate_id
        and event.payload["case_id"] in {"dev-a", "dev-b", "dev-c"}
    ]
    assert [event.payload["case_id"] for event in seed_completed] == [
        "dev-a",
        "dev-b",
        "dev-c",
    ]


def test_meta_are_retries_only_missing_repetition_for_partial_coverage(tmp_path: Path) -> None:
    config_path = _write_integration_fixture(
        tmp_path,
        infrastructure_retries=1,
        repetitions=2,
    )
    provider = ScriptedMetaAREProvider()
    runner = ScriptedMetaARERunner(first_attempt_missing_keys=(("dev-a", 1),))
    registry = default_registry()
    registry.providers["scripted_meta_are"] = lambda **_kwargs: provider

    runtime = build_runtime(config_path, run_id="meta-are-partial-repetition", registry=registry)
    runtime.scenario.evaluator.runner = runner
    runtime.engine.run()

    seed_candidate_id = runtime.engine._state().accepted_candidate_ids[0]
    seed_starts = [
        event
        for event in runtime.store.events_of_type("EvaluationAttemptStarted")
        if event.payload["candidate_id"] == seed_candidate_id and event.payload["case_id"] == "dev-a"
    ]
    assert [(event.payload["repetition"], event.payload["attempt_number"]) for event in seed_starts] == [
        (0, 1),
        (1, 1),
        (1, 2),
    ]
    seed_usage = [
        event
        for event in runtime.store.events_of_type("ModelUsageObserved")
        if event.payload.get("candidate_id") == seed_candidate_id and event.payload.get("case_id") == "dev-a"
    ]
    assert [(event.payload["repetition"], event.payload["attempt_number"]) for event in seed_usage] == [
        (0, 1),
        (1, 2),
    ]


@pytest.mark.parametrize("boundary", ["ModelUsageObserved", "EvaluationAttemptCompleted"])
def test_meta_are_single_retry_resume_preserves_recovered_repetitions(tmp_path: Path, boundary: str) -> None:
    config_path = _write_integration_fixture(tmp_path, infrastructure_retries=1, repetitions=3, priced=True)
    config = yaml.safe_load(config_path.read_text())
    config["optimization"]["budget"]["max_rollouts"] = 1
    config_path.write_text(yaml.safe_dump(config))
    runner = ScriptedMetaARERunner(first_attempt_missing_keys=(("dev-a", 1), ("dev-a", 2)))
    provider = ScriptedMetaAREProvider()
    registry = default_registry()
    registry.providers["scripted_meta_are"] = lambda **_kwargs: provider
    interrupted = build_runtime(config_path, run_id="single-retry-resume", registry=registry)
    interrupted.scenario.evaluator.runner = runner

    def interrupt(event):
        if (event.event_type == boundary and event.payload.get("attempt_number") == 2
                and event.payload.get("repetition") == 1):
            raise RuntimeError("interrupt during single retry")

    interrupted.store.transition_hook = interrupt
    with pytest.raises(RuntimeError, match="interrupt during single retry"):
        interrupted.engine.run()
    assert runner.execution_settings == [(3, None), (1, 1)]

    resumed = build_runtime(config_path, run_id="single-retry-resume", registry=registry)
    resumed.scenario.evaluator.runner = runner
    result = resumed.engine.run()
    assert result.iterations == 0
    assert runner.execution_settings == [(3, None), (1, 1), (1, 1)]
    attempts = resumed.store.events_of_type("EvaluationAttemptStarted")
    assert [(event.payload["repetition"], event.payload["attempt_number"]) for event in attempts] == [
        (0, 1), (1, 1), (2, 1), (1, 2), (2, 2),
    ]
    completed = resumed.store.events_of_type("EvaluationAttemptCompleted")
    assert [event.payload["observation"]["repetition"] for event in completed] == [0, 1, 2]
    usage = resumed.store.events_of_type("ModelUsageObserved")
    assert len(usage) == 3
    assert sum(event.payload["cost"]["currency_amount"] or 0 for event in completed) == pytest.approx(3 * 33.5e-6)


@pytest.mark.parametrize("last_retry_fails", [False, True])
def test_meta_are_single_retries_preserve_success_across_retry_rounds(tmp_path: Path, last_retry_fails: bool) -> None:
    config_path = _write_integration_fixture(tmp_path, infrastructure_retries=2, repetitions=3, priced=True)
    config = yaml.safe_load(config_path.read_text())
    config["optimization"]["budget"]["max_rollouts"] = 1
    config_path.write_text(yaml.safe_dump(config))

    class MixedRunner(ScriptedMetaARERunner):
        def run(self, **kwargs):
            result = super().run(**kwargs)
            if len(self.calls) == 3 or (last_retry_fails and len(self.calls) == 5):
                rows = [json.loads(line) for line in result.raw_results.read_text().splitlines()]
                for row in rows:
                    row["metadata"]["has_exception"] = True
                result.raw_results.write_text("".join(json.dumps(row) + "\n" for row in rows))
            return result

    runner = MixedRunner(invalid_attempts=1)
    registry = default_registry()
    registry.providers["scripted_meta_are"] = lambda **_kwargs: ScriptedMetaAREProvider()
    runtime = build_runtime(config_path, run_id="single-retry-rounds", registry=registry)
    runtime.scenario.evaluator.runner = runner
    if last_retry_fails:
        with pytest.raises(RuntimeError, match="retained invalid keys after bounded retries"):
            runtime.engine.run()
    else:
        runtime.engine.run()
    assert runner.execution_settings == [(3, None), (1, 1), (1, 1), (1, 1), (1, 1)]
    attempts = runtime.store.events_of_type("EvaluationAttemptStarted")
    assert [(event.payload["repetition"], event.payload["attempt_number"]) for event in attempts] == [
        (0, 1), (1, 1), (2, 1), (0, 2), (1, 2), (2, 2), (1, 3),
    ]
    assert len(runtime.store.events_of_type("ModelUsageObserved")) == 7
    completed = runtime.store.events_of_type("EvaluationAttemptCompleted")
    for repetition in (0, 2):
        recovered = [event.payload["observation"] for event in completed if event.payload["repetition"] == repetition]
        assert len(recovered) == 2
        assert recovered[-1]["disposition"] == "task_failure"


def test_meta_are_durable_batch_resume_does_not_recount_completed_repetition(tmp_path: Path) -> None:
    config_path = _write_integration_fixture(tmp_path, repetitions=2)
    provider = ScriptedMetaAREProvider()
    runner = ScriptedMetaARERunner()
    registry = default_registry()
    registry.providers["scripted_meta_are"] = lambda **_kwargs: provider

    interrupted = build_runtime(config_path, run_id="meta-are-repetition-resume", registry=registry)
    interrupted.scenario.evaluator.runner = runner

    def interrupt_after_first_observation(event) -> None:
        if event.event_type == "EvaluationAttemptCompleted":
            raise RuntimeError("interrupt after first repetition")

    interrupted.store.transition_hook = interrupt_after_first_observation
    with pytest.raises(RuntimeError, match="interrupt after first repetition"):
        interrupted.engine.run()

    resumed = build_runtime(config_path, run_id="meta-are-repetition-resume", registry=registry)
    resumed.scenario.evaluator.runner = runner
    resumed.engine.run()

    seed_candidate_id = resumed.store.events_of_type("RunStarted")[0].payload["seed_candidate"]["candidate_id"]
    seed_dev_starts = [
        event
        for event in resumed.store.events_of_type("EvaluationAttemptStarted")
        if event.payload["candidate_id"] == seed_candidate_id and event.payload["case_id"] == "dev-a"
    ]
    seed_dev_usage = [
        event
        for event in resumed.store.events_of_type("ModelUsageObserved")
        if event.payload.get("candidate_id") == seed_candidate_id and event.payload.get("case_id") == "dev-a"
    ]
    assert [(event.payload["repetition"], event.payload["attempt_number"]) for event in seed_dev_starts] == [
        (0, 1),
        (1, 1),
    ]
    assert [(event.payload["repetition"], event.payload["attempt_number"]) for event in seed_dev_usage] == [
        (0, 1),
        (1, 1),
    ]


def test_meta_are_fails_when_incomplete_result_coverage_exhausts_retries(
    tmp_path: Path,
) -> None:
    config_path = _write_integration_fixture(tmp_path, infrastructure_retries=1)
    provider = ScriptedMetaAREProvider()
    runner = ScriptedMetaARERunner(missing_attempts=2)
    registry = default_registry()
    registry.providers["scripted_meta_are"] = lambda **_kwargs: provider

    runtime = build_runtime(config_path, run_id="meta-are-missing-exhausted", registry=registry)
    runtime.scenario.evaluator.runner = runner
    with pytest.raises(RuntimeError, match="incomplete|bounded retries"):
        runtime.engine.run()

    assert len(runner.calls) == 2
    assert provider.calls == []
    assert len(runtime.store.events_of_type("EvaluationAttemptStarted")) == 2


@pytest.mark.parametrize("boundary", ["benchmark_usage", "benchmark_observation"])
def test_priced_retries_and_resume_preserve_execution_costs(tmp_path: Path, boundary: str) -> None:
    from autosaddler.v2.core.domain import ParetoScreenParameters
    from autosaddler.v2.core.policies import ParetoAcceptance
    from autosaddler.v2.core.serde import evaluation_from

    config_path = _write_integration_fixture(tmp_path, infrastructure_retries=1, repetitions=3, priced=True)
    provider = ScriptedMetaAREProvider()
    runner = ScriptedMetaARERunner(invalid_attempts=1)
    registry = default_registry()
    registry.providers["scripted_meta_are"] = lambda **_kwargs: provider
    interrupted = build_runtime(config_path, run_id="priced", registry=registry)
    interrupted.scenario.evaluator.runner = runner
    interrupted.store.transition_hook = InterruptMetaAREBoundary(boundary)
    with pytest.raises(RuntimeError, match=f"interrupt after {boundary}"):
        interrupted.engine.run()

    resumed = build_runtime(config_path, run_id="priced", registry=registry)
    resumed.scenario.evaluator.runner = runner
    result = resumed.engine.run()
    assert result.development_score == 1.0
    assert len(runner.calls) == 7
    benchmark = resumed.store.read_json("resolved/benchmark.json")
    assert benchmark["task_agent_pricing"]["input_usd_per_million"] == 2.0
    attempts = resumed.store.events_of_type("EvaluationAttemptCompleted")
    assert len(attempts) == 15
    amount = 33.5 / 1_000_000
    for event in attempts:
        assert event.payload["cost"]["currency_amount"] == pytest.approx(amount)
        assert event.payload["observation"]["cost"]["currency_amount"] == pytest.approx(amount)
        for cost in (event.payload["cost"], event.payload["observation"]["cost"]):
            assert cost["input_tokens"] == 7
            assert cost["cached_input_tokens"] == 3
            assert cost["uncached_input_tokens"] == 4
    assert sum(event.payload["cost"]["currency_amount"] for event in attempts) == pytest.approx(15 * amount)
    assert any(event.payload["observation"]["attempts"] == 2 for event in attempts)
    usage = resumed.store.events_of_type("ModelUsageObserved")
    assert len(usage) == 15
    assert all(event.payload["usage"]["cached_input_tokens"] == 3 for event in usage)
    evaluations = [evaluation_from(event.payload["evaluation"]) for event in resumed.store.events_of_type("EvaluationCompleted")]
    assert all(item.cost.currency_amount == pytest.approx(amount) for value in evaluations for item in value.observations)
    assert all(item.cost.cached_input_tokens == 3 for value in evaluations for item in value.observations)
    assert all(item.cost.uncached_input_tokens == 4 for value in evaluations for item in value.observations)
    before = next(value for value in evaluations if value.purpose == "train_before")
    after = next(value for value in evaluations if value.purpose == "train_after")
    verdict = ParetoAcceptance(ParetoScreenParameters(3)).compare(before, after)
    assert verdict.accepted
    assert verdict.samples_per_arm == 3
    assert verdict.before_cost == verdict.after_cost == pytest.approx(amount)

    completed = build_runtime(config_path, run_id="priced", registry=registry)
    completed.scenario.evaluator.runner = runner
    assert completed.engine.run() == result
    assert len(runner.calls) == 7
    assert len(completed.store.events_of_type("ModelUsageObserved")) == 15


@pytest.mark.parametrize("repetitions", [1, 3])
@pytest.mark.parametrize("capability_iterations", [0, 1])
def test_pareto_all_success_diagnosis_and_reflection_resume(
    tmp_path: Path, repetitions: int, capability_iterations: int,
) -> None:
    config_path = _write_integration_fixture(tmp_path, repetitions=repetitions, priced=True)
    config = yaml.safe_load(config_path.read_text())
    config["optimization"]["acceptance"] = {"type": "pareto", "parameters": {"draws": 100}}
    config["optimization"]["ranking"] = {"type": "pareto"}
    config["scenario"]["settings"]["capability_phase_iterations"] = capability_iterations
    config_path.write_text(yaml.safe_dump(config))
    provider = ScriptedMetaAREProvider()
    runner = ScriptedMetaARERunner(cost_only=True)
    registry = default_registry()
    registry.providers["scripted_meta_are"] = lambda **_kwargs: provider
    runtime = build_runtime(config_path, run_id="pareto-prompts", registry=registry)
    runtime.scenario.evaluator.runner = runner

    def interrupt(event):
        if event.event_type == "DeferredWorkScheduled":
            raise RuntimeError("interrupt before reflection")

    runtime.store.transition_hook = interrupt
    with pytest.raises(RuntimeError, match="interrupt before reflection"):
        runtime.engine.run()
    completed_calls = list(runner.calls)
    resumed = build_runtime(config_path, run_id="pareto-prompts", registry=registry)
    resumed.scenario.evaluator.runner = runner
    result = resumed.engine.run()
    assert runner.calls == completed_calls
    assert provider.calls == ["diagnose_patch", "reflect"]
    assert result.development_score == 1.0
    screen = resumed.store.events_of_type("AcceptanceDecided")[0].payload["verdict"]
    assert screen["accepted"] is True
    assert screen["before_score"] == screen["after_score"] == 1.0
    assert screen["after_cost"] < screen["before_cost"]
    phase = "capability" if capability_iterations else "steering"
    for request in provider.requests:
        files = request.spec.workspace_files
        context = json.loads(files[".autosaddler/session_context.json"])
        assert context["objective_mode"] == "pareto"
        assert context["patch_phase"] == phase
        assert "non-scalarized accuracy-cost objective" in request.spec.system_context
        manifest = json.loads(files[".autosaddler/prompt_assets.json"])
        assert manifest["objective_mode"] == "pareto"
        assert "dev-a" not in "\n".join(files.values())
        if request.spec.kind == "diagnose_patch":
            evidence = json.loads(files[".autosaddler/training_evidence.json"])
            assert evidence["requested_repetitions"] == repetitions
            for case in evidence["case_records"]:
                assert case["consistency"] == "consistent_pass"
                assert case["observed_repetitions"] == repetitions
                assert all(item["score"] == 1.0 and item["cost"]["currency_amount"] > 0
                           for item in case["per_repetition"])
                assert all(not usage["usage_incomplete"] for item in case["per_repetition"]
                           for usage in item["model_usage"])
        else:
            iteration = json.loads(files[".autosaddler/history/iterations/0000.json"])
            assert iteration["verdict"] == screen
            assert iteration["current_candidate_id"] == result.selected_candidate_id
    assert resumed.store.read_json("resolved/prompts/compositions.json")["compositions"][f"pareto.diagnose_patch.{phase}"]


@pytest.mark.parametrize("change", ["pricing", "model", "repetitions", "screen", "method", "mode", "final_selection"])
@pytest.mark.parametrize("fork", [False, True])
def test_pareto_runtime_preserves_provenance_on_resume_and_fork(tmp_path: Path, change: str, fork: bool) -> None:
    from autosaddler.v2.core.policies import ParetoAcceptance, ParetoDevelopmentRanking
    from autosaddler.v2.core.serde import record

    config_path = _write_integration_fixture(tmp_path, repetitions=2, priced=True)
    config = yaml.safe_load(config_path.read_text())
    config["optimization"]["acceptance"] = {"type": "pareto", "parameters": {"draws": 100}}
    config["optimization"]["ranking"] = {"type": "pareto"}
    config_path.write_text(yaml.safe_dump(config))
    provider = ScriptedMetaAREProvider()
    registry = default_registry()
    registry.providers["scripted_meta_are"] = lambda **_kwargs: provider
    runtime = build_runtime(config_path, run_id="pareto", registry=registry)
    assert isinstance(runtime.policies.acceptance, ParetoAcceptance)
    assert isinstance(runtime.policies.ranking, ParetoDevelopmentRanking)
    seed = runtime.scenario.harness_space.seed()
    runtime.store.append("RunStarted", "pareto:start", {"seed_candidate": record(seed)})
    original_events = runtime.store.events_path.read_bytes()
    original_config = (runtime.store.run_dir / "resolved_config.yaml").read_bytes()
    resumed = build_runtime(config_path, run_id="pareto", registry=registry)
    assert resumed.store.events() == runtime.store.events()
    if change == "pricing":
        config["scenario"]["settings"]["task_agent_pricing"]["input_usd_per_million"] = 3.0
    elif change == "model":
        config["scenario"]["settings"]["model"] = "other-model"
    elif change == "repetitions":
        config["scenario"]["settings"]["repetitions"] = 3
    elif change == "screen":
        config["optimization"]["acceptance"]["parameters"]["draws"] = 200
    elif change == "method":
        config["optimization"]["acceptance"]["parameters"]["method"] = "other"
    elif change == "final_selection":
        config["optimization"]["final_selection"] = "cost"
    elif change == "mode":
        config["optimization"]["acceptance"] = {"type": "matched_valid_strict_improvement"}
        config["optimization"]["ranking"] = {"type": "mean_development_score"}
    config_path.write_text(yaml.safe_dump(config))
    with pytest.raises(
        TypeError if change == "method" else ValueError,
        match="unexpected keyword argument 'method'" if change == "method" else "changed on resume|differ only",
    ):
        target = build_runtime(config_path, run_id="fork" if fork else "pareto", registry=registry)
        if fork:
            target.store.fork_from(runtime.store, through_sequence=len(runtime.store.events()))
    assert runtime.store.events_path.read_bytes() == original_events
    assert (runtime.store.run_dir / "resolved_config.yaml").read_bytes() == original_config
    assert provider.calls == []
    assert not runtime.store.events_of_type("EvaluationAttemptStarted")


def _write_integration_fixture(
    tmp_path: Path,
    *,
    infrastructure_retries: int = 0,
    development_case_ids: tuple[str, ...] = ("dev-a",),
    repetitions: int = 1,
    priced: bool = False,
) -> Path:
    source = tmp_path / "meta-are"
    source.mkdir()
    (source / "pyproject.toml").write_text(
        "[project]\nname = 'meta-are-fixture'\nversion = '0.0.0'\n",
        encoding="utf-8",
    )
    (source / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    (source / "hook.json").write_text('{"hooks":{"PreToolUse":[]}}\n', encoding="utf-8")
    subprocess.run(["git", "init", "-q"], cwd=source, check=True)
    subprocess.run(["git", "add", "."], cwd=source, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "user.name=Test",
            "-c",
            "user.email=test@example.com",
            "commit",
            "-qm",
            "fixture",
        ],
        cwd=source,
        check=True,
    )
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=source,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()

    dataset = tmp_path / "dataset"
    dataset.mkdir()
    for case_id in ("train-a", *development_case_ids):
        (dataset / f"{case_id}.json").write_text(
            json.dumps({"id": case_id}) + "\n",
            encoding="utf-8",
        )
    train_manifest = tmp_path / "train.json"
    train_manifest.write_text('["train-a"]\n', encoding="utf-8")
    development_manifest = tmp_path / "development.json"
    development_manifest.write_text(
        json.dumps(list(development_case_ids)) + "\n",
        encoding="utf-8",
    )
    test_manifest = tmp_path / "test-universe-22.json"
    test_manifest.write_text('["test-a"]\n', encoding="utf-8")
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
                    {"path": str(path), "sha256": sha256_digest(path.read_bytes())}
                    for path in (train_manifest, development_manifest)
                ],
                "files": [
                    {
                        "scenario_id": case_id,
                        "path": f"{case_id}.json",
                        "bytes": len((dataset / f"{case_id}.json").read_bytes()),
                        "sha256": sha256_digest((dataset / f"{case_id}.json").read_bytes()),
                    }
                    for case_id in ("train-a", *development_case_ids)
                ],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    demo = tmp_path / "demo-filesystem"
    demo.mkdir()
    (demo / "README.txt").write_text("fixture\n", encoding="utf-8")
    demo_revision = "a" * 40
    demo_descriptor = tmp_path / "demo-source.json"
    demo_descriptor.write_text(
        json.dumps({"source_revision": demo_revision}) + "\n",
        encoding="utf-8",
    )
    demo_manifest = tmp_path / "demo-manifest.json"
    demo_manifest.write_text(
        json.dumps({"files": ["README.txt"]}) + "\n",
        encoding="utf-8",
    )

    config = {
        "schema_version": "autosaddler/v2",
        "scenario": {
            "type": "meta_are",
            "settings": {
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
                    "are/simulation/apps",
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
                "demo_filesystem_source_revision": demo_revision,
                "demo_filesystem_source_descriptor": str(demo_descriptor),
                "demo_filesystem_manifest": str(demo_manifest),
                "benchmark_config": "search",
                "benchmark_split": "validation",
                "agent": "default",
                "model": "task-model",
                "model_provider": "fixture",
                "model_wire_api": "chat_completions",
                "model_endpoint": None,
                "reasoning_effort": None,
                "judge_model": "judge-model",
                "judge_provider": "fixture",
                "judge_endpoint": None,
                "scenario_timeout_seconds": 30,
                "process_completion_grace_seconds": 60,
                "verification_timeout_seconds": 30,
                "repetitions": repetitions,
                "max_concurrent": 1,
                "infrastructure_retries": infrastructure_retries,
                "import_check": "import json",
                "capability_phase_iterations": 1,
            },
        },
        "optimization": {
            "task_selection": {"type": "epoch_shuffled", "batch_size": 1},
            "acceptance": {"type": "matched_valid_strict_improvement"},
            "development": {"type": "full_on_accept"},
            "ranking": {"type": "mean_development_score"},
            "budget": {"max_rollouts": 20, "max_iterations": 1},
            "diagnosis_patch_timeout_seconds": 30,
        },
        "provider": {
            "type": "scripted_meta_are",
            "capabilities": [
                "read_workspace",
                "edit_workspace",
                "run_commands",
                "load_skills",
            ],
            "settings": {},
        },
        "storage": {"type": "local", "run_root": str(tmp_path / "runs")},
    }
    if priced:
        config["scenario"]["settings"]["task_agent_pricing"] = {
            "input_usd_per_million": 2.0,
            "cached_input_usd_per_million": 0.5,
            "output_usd_per_million": 8.0,
        }
    config_path = tmp_path / "autosaddler.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return config_path
