"""Test-only support for the adaptive ActiveSaddler task-selection policy.

The built-in fake scenario supports only the passive session kinds, and its paid-work
ledger treats any re-evaluation of a case on the same candidate as duplicate work. An
arm pull re-evaluates known failure cases by design, so curriculum tests use this
wrapped scenario: it answers the curriculum sessions deterministically and scopes the
paid-work key to the evaluation operation, which is how attempt identity is scoped.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from autosaddler.v2.config.registry import Registry, default_registry
from autosaddler.v2.core.curriculum import CURRICULUM_SESSION_KINDS
from autosaddler.v2.core.domain import JsonValue, canonical_json
from autosaddler.v2.core.ports import BASE_SESSION_KINDS, ScenarioComponents
from autosaddler.v2.prompting.curriculum import arm_decision_schema, arm_scoring_schema, pattern_extraction_schema
from autosaddler.v2.prompting.models import SessionSpec


class OperationScopedLedger:
    def __init__(self, ledger: Any, operation_id: str) -> None:
        self.ledger = ledger
        self.operation_id = operation_id

    def record(self, kind: str, key: str) -> None:
        self.ledger.record(kind, canonical_json({"evaluation_operation_id": self.operation_id, "key": key}))


class OperationScopedFakeEvaluator:
    def __init__(self, evaluator: Any) -> None:
        self.evaluator = evaluator

    async def evaluate(self, candidate, cases, context):
        ledger = self.evaluator.ledger
        self.evaluator.ledger = OperationScopedLedger(ledger, context.operation_id)
        try:
            return await self.evaluator.evaluate(candidate, cases, context)
        finally:
            self.evaluator.ledger = ledger


class CurriculumFakePromptPack:
    """Delegate passive kinds to the fake pack and answer curriculum kinds deterministically."""

    def __init__(self, base: Any) -> None:
        self.base = base

    def session(self, kind: str, context: Mapping[str, JsonValue]) -> SessionSpec:
        if kind not in CURRICULUM_SESSION_KINDS:
            return self.base.session(kind, context)
        base_spec = self.base.session("reflect", {**context, "train_case_ids": context.get("train_case_ids", [])})
        if kind == "extract_patterns":
            response, schema = _pattern_extraction(context)
        elif kind == "decide_arm":
            response = {
                "schema_version": "autosaddler-fake-arm-decision/v1",
                "action": "pull",
                "rationale": "Revisit the known failure pattern before drawing new cases.",
            }
            schema = arm_decision_schema("autosaddler-fake-arm-decision/v1")
        else:
            arm_ids = _task_selection_strings(context, "arm_ids")
            response = {
                "schema_version": "autosaddler-fake-arm-scoring/v1",
                "scores": [
                    {
                        "pattern_id": arm_id,
                        "severity": 1.0,
                        "fixability": 1.0,
                        "breadth": 0.5,
                        "side_effect": 0.0,
                        "rationale": "The deterministic baseline failure remains fixable.",
                    }
                    for arm_id in arm_ids
                ],
            }
            schema = arm_scoring_schema("autosaddler-fake-arm-scoring/v1", arm_ids)
        return replace(
            base_spec,
            kind=kind,
            task_prompt=f"Execute the {kind} phase for this deterministic scenario.",
            output_schema=schema,
            workspace_files={
                "session_context.json": canonical_json(context) + "\n",
                ".autosaddler/fake_response.json": canonical_json(response) + "\n",
            },
        )


def _pattern_extraction(
    context: Mapping[str, JsonValue],
) -> tuple[Mapping[str, JsonValue], Mapping[str, JsonValue]]:
    failures = [
        (source, str(item["case_id"]))
        for source, key in (("pre_patch", "pre_patch_failures"), ("post_patch", "post_patch_failures"))
        for item in _records(context.get(key))
    ]
    existing = context.get("existing_pattern_ids")
    assert isinstance(existing, list)
    new_patterns: list[JsonValue] = [] if existing else [{"key": "incomplete-instruction", "label": "Incomplete instruction"}]
    reference = str(existing[0]) if existing else "incomplete-instruction"
    response: Mapping[str, JsonValue] = {
        "schema_version": "autosaddler-fake-pattern-extraction/v1",
        "new_patterns": new_patterns,
        "tags": [
            {
                "case_id": case_id,
                "source": source,
                "pattern_refs": [reference],
                "root_cause": "The instruction omits the required behavior.",
            }
            for source, case_id in failures
        ],
    }
    schema = pattern_extraction_schema(
        "autosaddler-fake-pattern-extraction/v1",
        [case_id for _, case_id in failures],
    )
    return response, schema


def _records(value: JsonValue | None) -> list[Mapping[str, JsonValue]]:
    assert isinstance(value, list)
    return [item for item in value if isinstance(item, Mapping)]


def _task_selection_strings(context: Mapping[str, JsonValue], key: str) -> list[str]:
    task_selection = context.get("task_selection")
    assert isinstance(task_selection, Mapping)
    value = task_selection.get(key)
    assert isinstance(value, list)
    return [str(item) for item in value]


def curriculum_fake_factory(base_factory: Callable[..., ScenarioComponents]) -> Callable[..., ScenarioComponents]:
    def build(**kwargs: Any) -> ScenarioComponents:
        base = base_factory(**kwargs)
        return replace(
            base,
            prompt_pack=CurriculumFakePromptPack(base.prompt_pack),
            evaluator=OperationScopedFakeEvaluator(base.evaluator),
            supported_session_kinds=BASE_SESSION_KINDS | CURRICULUM_SESSION_KINDS,
        )

    return build


@pytest.fixture
def curriculum_registry() -> Registry:
    """Default registry whose fake scenario supports the ActiveSaddler curriculum."""
    registry = default_registry()
    registry.scenarios["fake"] = curriculum_fake_factory(registry.scenarios["fake"])
    return registry


@pytest.fixture
def activesaddler_config() -> Callable[..., dict]:
    """Build a fake-scenario ActiveSaddler config mapping."""

    def build(
        root: Path,
        *,
        train_case_ids: tuple[str, ...] = ("train-a", "train-b", "train-c", "train-d"),
        development_case_ids: tuple[str, ...] = ("dev-a", "dev-b"),
        batch_size: int = 2,
        max_iterations: int = 2,
        extraction_timeout: float = 11,
        arm_scoring_timeout: float = 12,
    ) -> dict:
        return {
            "schema_version": "autosaddler/v2",
            "scenario": {
                "type": "fake",
                "settings": {
                    "baseline": {"instruction": "baseline"},
                    "target_component": "instruction",
                    "improved_text": "improved",
                    "train_case_ids": list(train_case_ids),
                    "development_case_ids": list(development_case_ids),
                },
            },
            "optimization": {
                "task_selection": {
                    "type": "activesaddler",
                    "batch_size": batch_size,
                    "seed": 0,
                    "settings": {
                        "softmax_temperature": 0.15,
                        "min_prob": 0.0,
                        "ema_eta": 0.9,
                        "pattern_extraction_timeout_seconds": extraction_timeout,
                        "arm_scoring_timeout_seconds": arm_scoring_timeout,
                    },
                },
                "acceptance": {"type": "matched_valid_strict_improvement"},
                "development": {"type": "full_on_accept"},
                "ranking": {"type": "mean_development_score"},
                "budget": {"max_rollouts": 100, "max_iterations": max_iterations},
                "diagnosis_patch_timeout_seconds": 10,
            },
            "provider": {
                "type": "fake",
                "capabilities": ["read_workspace", "edit_workspace", "load_skills"],
                "settings": {},
            },
            "storage": {"type": "local", "run_root": str(root / "runs")},
        }

    return build
