from __future__ import annotations

import json
from pathlib import Path

import pytest

from autosaddler.v2.core.metrics import EventModelUsageSink
from autosaddler.v2.prompting.models import Usage
from autosaddler.v2.storage.local import LocalRunStore


def initialized_store(tmp_path: Path) -> LocalRunStore:
    store = LocalRunStore(run_dir=tmp_path / "run", run_id="run-1")
    store.initialize(
        resolved_config={"schema_version": "autosaddler/v2"},
        resolved_entities={"resolved/component_graph.json": {"scenario": "fake"}},
    )
    return store


def test_event_append_is_flushed_idempotent_and_projected(tmp_path: Path) -> None:
    store = initialized_store(tmp_path)
    first = store.append("RunStarted", "run-1:start", {"seed_candidate": {"candidate_id": "sha256:seed"}})
    same = store.append("RunStarted", "run-1:start", {"seed_candidate": {"candidate_id": "sha256:seed"}})
    store.append("RunCompleted", "run-1:complete", {"candidate_id": "sha256:seed"})

    assert first == same
    assert len(store.events()) == 2
    assert json.loads((store.run_dir / "manifest.json").read_text())["status"] == "completed"
    assert json.loads((store.run_dir / "snapshot.json").read_text())["last_sequence"] == 2
    store.validate_integrity()


def test_event_idempotency_conflict_fails_closed(tmp_path: Path) -> None:
    store = initialized_store(tmp_path)
    store.append("BatchSampled", "run-1:batch", {"case_ids": ["a"]})

    with pytest.raises(RuntimeError, match="Idempotency conflict"):
        store.append("BatchSampled", "run-1:batch", {"case_ids": ["b"]})


def test_resumed_completion_supersedes_prior_failed_invocation_status(tmp_path: Path) -> None:
    store = initialized_store(tmp_path)
    store.append("RunStarted", "run-1:start", {"seed_candidate": {"candidate_id": "sha256:seed"}})
    store.append("RunFailed", "run-1:failed", {"error": "transient normalization failure"})
    assert store.read_json("manifest.json")["status"] == "failed"

    store.append("RunResumed", "run-1:resume", {"prior_sequence": 2})
    assert store.read_json("manifest.json")["status"] == "optimizing"

    store.append("RunCompleted", "run-1:complete", {"candidate_id": "sha256:seed"})

    assert store.read_json("manifest.json")["status"] == "completed"


def test_event_reader_rejects_unknown_types(tmp_path: Path) -> None:
    store = initialized_store(tmp_path)
    (store.run_dir / "events.jsonl").write_text(
        json.dumps(
            {
                "schema_version": "autosaddler-event/v1",
                "sequence": 1,
                "timestamp": "now",
                "run_id": "run-1",
                "event_type": "FutureEvent",
                "operation_id": "run-1:future",
                "payload": {},
            }
        )
        + "\n"
    )

    with pytest.raises(ValueError, match="Unknown event type"):
        store.events()


def test_metrics_projection_reconciles_roles_costs_and_all_evaluation_purposes(tmp_path: Path) -> None:
    store = initialized_store(tmp_path)
    contexts = (
        ("optimizer", "proposal.patch", None),
        ("task_agent", "evaluation.task_agent", "train_before"),
        ("judge", "evaluation.task_agent", "development"),
        ("simulated_user", "evaluation.task_agent", "test"),
    )
    for index, (role, stage, purpose) in enumerate(contexts):
        sink = EventModelUsageSink(
            store=store,
            attempt_operation_id=f"run-1:attempt:{index}",
            context={
                "stage": stage,
                "iteration": index if purpose != "test" else None,
                "candidate_id": f"sha256:candidate-{index}",
                "evaluation_purpose": purpose,
            },
        )
        sink.observe(
            Usage(
                role=role,
                model="model-a",
                input_tokens=10,
                cached_input_tokens=2,
                uncached_input_tokens=8,
                output_tokens=5,
                reasoning_tokens=1,
                agent_id="agent-a" if index % 2 else None,
                agent_scope="subagent" if index % 2 else "main",
                provider_cost=0.5,
                provider_cost_unit="premium_request_multiplier",
                provider_nano_aiu=3.0 if index else None,
                provider_ai_credits=0.25 if index == 0 else None,
            )
        )
    store.append(
        "RunCompleted",
        "run-1:complete",
        {"candidate_id": "sha256:candidate-0", "wall_seconds": 4.0},
    )

    rows = [json.loads(line) for line in (store.run_dir / "metrics.jsonl").read_text().splitlines()]
    summary = json.loads((store.run_dir / "metrics-summary.json").read_text())

    assert len([row for row in rows if row["type"] == "model_call"]) == 4
    assert summary["source_event_count"] == len(store.events())
    assert summary["model_usage"]["total_tokens"] == 60
    assert summary["model_usage"]["input_tokens"] == 40
    assert summary["model_usage"]["cached_input_tokens"] == 8
    assert summary["model_usage"]["uncached_input_tokens"] == 32
    assert summary["model_usage"]["nonreasoning_output_tokens"] == 16
    assert set(summary["model_usage_by_role"]) == {"optimizer", "task_agent", "judge", "simulated_user"}
    assert set(summary["model_usage_by_evaluation_purpose"]) == {
        "not_evaluation",
        "train_before",
        "development",
        "test",
    }
    assert set(summary["model_usage_by_agent_scope"]) == {"main", "subagent"}
    assert summary["provider_reported_cost_by_unit"]["ai_credit"] == pytest.approx(0.25 + 9e-9)
    assert summary["provider_reported_cost_by_unit"]["nano_aiu"] == 9.0
    assert summary["provider_reported_cost_by_unit"]["premium_request_multiplier"] == 2.0
    assert summary["active_wall_seconds"] == 4.0
    assert summary["accepted_candidate_count"] == 0
    assert summary["new_best_candidate_count"] == 0
    assert summary["per_new_best_candidate"]["tokens"] is None


def test_failed_session_retains_observed_usage_as_incomplete(tmp_path: Path) -> None:
    store = initialized_store(tmp_path)
    attempt_operation = "run-1:session:attempt:1"
    sink = EventModelUsageSink(
        store=store,
        attempt_operation_id=attempt_operation,
        context={"stage": "proposal.patch", "role": "optimizer"},
    )
    sink.observe(Usage(input_tokens=7, output_tokens=3, model="model-a"))
    store.append(
        "SessionFailed",
        attempt_operation,
        {"status": "timeout", "error": "TimeoutError: provider timed out", "wall_seconds": 5.0},
    )

    rows = [json.loads(line) for line in (store.run_dir / "metrics.jsonl").read_text().splitlines()]
    model_call = next(row for row in rows if row["type"] == "model_call")
    summary = json.loads((store.run_dir / "metrics-summary.json").read_text())

    assert model_call["status"] == "timeout"
    assert model_call["error_type"] == "TimeoutError"
    assert model_call["usage_incomplete"] is True
    assert summary["model_usage"]["total_tokens"] == 10
    assert summary["model_usage"]["failed_model_calls"] == 1

    before = (store.run_dir / "metrics-summary.json").read_bytes()
    store.refresh_projections()
    assert (store.run_dir / "metrics-summary.json").read_bytes() == before


def test_stage_metrics_exclude_session_and_evaluation_content(tmp_path: Path) -> None:
    store = initialized_store(tmp_path)
    store.append(
        "SessionCompleted",
        "run-1:session:attempt:1",
        {
            "stage": "proposal.patch",
            "candidate_id": "sha256:candidate",
            "wall_seconds": 2.0,
            "result": {
                "raw_response": "secret response",
                "tool_calls": [
                    {
                        "tool": "secret_tool",
                        "arguments": {"value": "secret argument"},
                        "result_preview": "secret result",
                    }
                ],
            },
        },
    )
    store.append(
        "EvaluationAttemptCompleted",
        "run-1:evaluation:attempt:1",
        {
            "evaluation_purpose": "development",
            "case_id": "case-a",
            "wall_seconds": 3.0,
            "observation": {
                "output": {"uri": "secret output"},
                "trace": {"uri": "secret trace"},
            },
        },
    )

    metrics_text = (store.run_dir / "metrics.jsonl").read_text()
    rows = [json.loads(line) for line in metrics_text.splitlines()]

    assert "secret" not in metrics_text
    assert [row["stage"] for row in rows] == ["proposal.patch", "evaluation_attempt"]
    assert all("result" not in row and "observation" not in row for row in rows)
