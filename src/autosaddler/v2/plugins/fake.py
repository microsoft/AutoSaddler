from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from autosaddler.v2.core.domain import ArtifactRef, Case, Cost, Evaluation, JsonValue, Observation, canonical_json, sha256_digest
from autosaddler.v2.core.ports import EvaluationContext, ScenarioComponents
from autosaddler.v2.harness.component_map import ComponentMapHarnessSpace
from autosaddler.v2.prompting.models import SessionSpec
from autosaddler.v2.providers.fake import PaidWorkLedger
from autosaddler.v2.storage.local import LocalRunStore


@dataclass(frozen=True, slots=True)
class FakeScenarioSettings:
    baseline: Mapping[str, str]
    target_component: str
    improved_text: str
    train_case_ids: tuple[str, ...]
    development_case_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if self.target_component not in self.baseline:
            raise ValueError("Fake target_component must exist in the baseline")
        if not self.improved_text.strip():
            raise ValueError("Fake improved_text must be non-empty")
        if not self.train_case_ids or not self.development_case_ids:
            raise ValueError("Fake scenario splits cannot be empty")


class FakeEvaluator:
    def __init__(
        self,
        *,
        harness_space: ComponentMapHarnessSpace,
        target_component: str,
        improved_text: str,
        ledger: PaidWorkLedger,
    ) -> None:
        self.harness_space = harness_space
        self.target_component = target_component
        self.improved_text = improved_text
        self.ledger = ledger
        self.fingerprint = "fake-evaluator/v1:deterministic"

    async def evaluate(
        self,
        candidate,
        cases,
        context: EvaluationContext,
    ) -> Evaluation:
        if not cases or any(case.split != context.split for case in cases):
            raise ValueError("Fake evaluator cases must match EvaluationContext.split")
        materialized = self.harness_space.materialize(candidate, "evaluate")
        try:
            value = json.loads((materialized.root / "candidate.json").read_text(encoding="utf-8"))
        finally:
            materialized.release()
        improved = value.get(self.target_component) == self.improved_text
        observations: list[Observation] = []
        for case in cases:
            for repetition in range(context.repetitions):
                cached = context.attempt_sink.completed(
                    candidate_id=candidate.candidate_id,
                    case_id=case.case_id,
                    repetition=repetition,
                )
                if cached is not None:
                    observations.append(cached)
                    continue
                attempt_id, attempt_number = context.attempt_sink.start(
                    candidate_id=candidate.candidate_id,
                    case_id=case.case_id,
                    repetition=repetition,
                )
                paid_key = canonical_json(
                    {
                        "candidate_id": candidate.candidate_id,
                        "case_id": case.case_id,
                        "repetition": repetition,
                        "evaluator": self.fingerprint,
                    }
                )
                self.ledger.record("rollout", paid_key)
                score = 1.0 if improved else 0.0
                observation = Observation.create(
                    candidate_id=candidate.candidate_id,
                    case_id=case.case_id,
                    split=case.split,
                    repetition=repetition,
                    disposition="success" if improved else "task_failure",
                    score=score,
                    evaluator_fingerprint=self.fingerprint,
                    attempts=attempt_number,
                    cost=Cost(rollouts=1),
                    metadata={"evaluator": self.fingerprint},
                )
                context.attempt_sink.complete(attempt_id, observation, Cost(rollouts=1))
                observations.append(observation)
        evaluation_id = sha256_digest(context.operation_id)
        artifact_prefix = {
            "train": "evaluations",
            "development": "quarantine/dev",
            "test": "post_optimization/test",
        }[context.split]
        return Evaluation(
            evaluation_id=evaluation_id,
            candidate_id=candidate.candidate_id,
            split=context.split,
            purpose=context.purpose,
            iteration=context.iteration,
            requested_case_ids=tuple(case.case_id for case in cases),
            observations=tuple(observations),
            artifact_dir=ArtifactRef(
                uri=f"{artifact_prefix}/{evaluation_id.removeprefix('sha256:')}",
                kind="evaluation-directory",
            ),
        )


class FakeEvidenceBuilder:
    def __init__(self, store: LocalRunStore) -> None:
        self.store = store

    def build(self, evaluation: Evaluation) -> ArtifactRef:
        if evaluation.split != "train":
            raise ValueError("Optimization evidence may be built from training evaluations only")
        observations = list(evaluation.observations)
        evidence_id = sha256_digest(evaluation.evaluation_id)
        return self.store.write_json(
            f"evidence/{evidence_id.removeprefix('sha256:')}/evidence.json",
            {
                "schema_version": "autosaddler-fake-evidence/v1",
                "evaluation_id": evaluation.evaluation_id,
                "case_ids": [observation.case_id for observation in observations],
                "observations": [
                    {
                        "case_id": observation.case_id,
                        "disposition": observation.disposition,
                        "score": observation.score,
                    }
                    for observation in observations
                ],
            },
            kind="training-evidence",
        )


class FakePromptPack:
    def __init__(self, *, target_component: str, improved_text: str) -> None:
        self.target_component = target_component
        self.improved_text = improved_text

    def session(self, kind: str, context: Mapping[str, JsonValue]) -> SessionSpec:
        if kind == "evolve":
            candidate_ids = context.get("candidate_ids")
            if not isinstance(candidate_ids, list) or not candidate_ids:
                raise ValueError("Evolve context requires candidate_ids")
            response: Mapping[str, JsonValue] = {
                "schema_version": "autosaddler-evolution/v1",
                "parent_ids": [candidate_ids[0]],
                "component_sources": {},
                "rationale": "Use the only accepted candidate.",
            }
            schema = _object_schema(("schema_version", "parent_ids", "component_sources", "rationale"))
        elif kind == "diagnose_patch":
            response = {
                "schema_version": "autosaddler-diagnosis-patch/v1",
                "updates": {self.target_component: self.improved_text},
                "diagnosis": "The baseline instruction is intentionally incomplete.",
            }
            schema = _object_schema(("schema_version", "updates", "diagnosis"))
        elif kind == "reflect":
            response = {
                "schema_version": "autosaddler-reflection/v1",
                "lessons": [
                    {
                        "scope": "component",
                        "statement": "Precise instructions improved every matched training case.",
                        "evidence_case_ids": context.get("train_case_ids", []),
                    }
                ],
            }
            schema = _object_schema(("schema_version", "lessons"))
        else:
            raise ValueError(f"Unknown fake session kind: {kind}")
        return SessionSpec(
            kind=kind,
            system_context="Use training evidence only and obey the structured output contract.",
            task_prompt=f"Execute the {kind} phase for this deterministic scenario.",
            skills={
                "fake-method": (
                    "---\n"
                    "name: fake-method\n"
                    'description: "Return the deterministic fake scenario contract output."\n'
                    "---\n\n"
                    "# Fake Method\n\n"
                    "Return the deterministic contract output.\n"
                )
            },
            output_schema=schema,
            workspace_files={
                "session_context.json": canonical_json(context) + "\n",
                ".autosaddler/fake_response.json": canonical_json(response) + "\n",
            },
            capabilities=frozenset({"read_workspace", "edit_workspace", "load_skills"}),
        )


def build_fake_components(
    *,
    settings: FakeScenarioSettings,
    run_dir: Path,
    store: LocalRunStore,
    ledger: PaidWorkLedger,
) -> ScenarioComponents:
    harness = ComponentMapHarnessSpace(
        baseline=settings.baseline,
        store_root=run_dir / "candidates",
        materialization_root=run_dir / "materialized",
    )
    evaluator = FakeEvaluator(
        harness_space=harness,
        target_component=settings.target_component,
        improved_text=settings.improved_text,
        ledger=ledger,
    )
    return ScenarioComponents(
        name="fake",
        version="1",
        harness_space=harness,
        evaluator=evaluator,
        evidence_builder=FakeEvidenceBuilder(store),
        prompt_pack=FakePromptPack(
            target_component=settings.target_component,
            improved_text=settings.improved_text,
        ),
        train_cases=tuple(Case(case_id=case_id, split="train", payload={}) for case_id in settings.train_case_ids),
        development_cases=tuple(
            Case(case_id=case_id, split="development", payload={}) for case_id in settings.development_case_ids
        ),
        required_capabilities=frozenset({"read_workspace", "edit_workspace", "load_skills"}),
        resolved_entities={
            "resolved/sources/harness.json": {
                "type": "structured",
                "space": "component_map",
                "content_digest": sha256_digest(canonical_json(settings.baseline)),
            },
            "resolved/sources/dataset.json": {
                "type": "fake",
                "train_case_ids": list(settings.train_case_ids),
                "development_case_ids": list(settings.development_case_ids),
                "test": {"state": "opaque", "opened": False},
            },
            "resolved/prompts/evolve.md": "# Evolve\n\nSelect one accepted parent.\n",
            "resolved/prompts/diagnosis_patch.md": "# Diagnose and patch\n\nUse training evidence only.\n",
            "resolved/prompts/reflect.md": "# Reflect\n\nUse aggregate development feedback only.\n",
        },
    )


def _object_schema(required: tuple[str, ...]) -> Mapping[str, JsonValue]:
    return {
        "type": "object",
        "required": list(required),
        "properties": {name: {} for name in required},
        "additionalProperties": False,
    }