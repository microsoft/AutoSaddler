from __future__ import annotations

import json
from pathlib import Path

import pytest

from autosaddler.v2.core.domain import Case, sha256_digest

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "baseline_failed"
CANDIDATE_ID = sha256_digest("candidate")


def test_recorded_baseline_normalizes_single_null_run_number_to_zero(tmp_path: Path) -> None:
    from autosaddler.v2.plugins.meta_are.evaluator import MetaAREResultNormalizer

    normalizer = MetaAREResultNormalizer(evaluator_fingerprint="meta-are-test/v1")
    observations = normalizer.normalize(
        raw_results=FIXTURE_ROOT / "output.jsonl",
        hf_trace_dir=FIXTURE_ROOT / "hf",
        lite_trace_dir=FIXTURE_ROOT / "lite",
        candidate_id=CANDIDATE_ID,
        requested_cases=(Case("scenario_universe_29_096dyu", "train", {}),),
        repetitions=1,
        artifact_dir=tmp_path,
    )

    assert len(observations) == 1
    observation = observations[0]
    assert observation.case_id == "scenario_universe_29_096dyu"
    assert observation.repetition == 0
    assert observation.disposition == "task_failure"
    assert observation.score == 0.0
    assert observation.trace is not None
    assert observation.output is not None
    assert observation.metadata["producer_status"] == "failed"
    assert observation.metadata["judge_calls"] == 0
    assert observation.metadata["task_agent_calls"] == 1
    model_usage = observation.metadata["model_usage"]
    assert isinstance(model_usage, list)
    assert model_usage[0]["role"] == "task_agent"
    assert model_usage[0]["input_tokens"] == 12283
    assert model_usage[0]["output_tokens"] == 1062
    assert model_usage[0]["reasoning_tokens"] == 832


def test_normalizer_preserves_repetitions_and_classifies_exceptions_as_invalid(tmp_path: Path) -> None:
    from autosaddler.v2.plugins.meta_are.evaluator import MetaAREResultNormalizer

    raw_results, hf_dir, lite_dir = _write_results(
        tmp_path,
        [
            _row("case-a", run_number=1, status="success", score=1.0),
            _row(
                "case-a",
                run_number=2,
                status="failed",
                score=0.0,
                has_exception=True,
            ),
        ],
    )
    observations = MetaAREResultNormalizer(evaluator_fingerprint="meta-are-test/v1").normalize(
        raw_results=raw_results,
        hf_trace_dir=hf_dir,
        lite_trace_dir=lite_dir,
        candidate_id=CANDIDATE_ID,
        requested_cases=(Case("case-a", "train", {}),),
        repetitions=2,
        artifact_dir=tmp_path / "normalized",
    )

    assert [item.repetition for item in observations] == [0, 1]
    assert [item.disposition for item in observations] == ["success", "execution_error"]
    assert [item.score for item in observations] == [1.0, None]
    assert observations[0].metadata["task_agent_calls"] == 1
    assert observations[0].metadata["judge_calls"] == 1
    assert [item["role"] for item in observations[0].metadata["model_usage"]] == ["task_agent", "judge"]


def test_normalizer_scores_scenario_timeout_as_task_failure(tmp_path: Path) -> None:
    from autosaddler.v2.plugins.meta_are.evaluator import MetaAREResultNormalizer

    row = _row(
        "case-a",
        run_number=None,
        status="failed",
        score=0.0,
        has_exception=True,
    )
    metadata = row["metadata"]
    assert isinstance(metadata, dict)
    metadata["exception_type"] = "ScenarioTimeoutError"
    raw_results, hf_dir, lite_dir = _write_results(tmp_path, [row])
    for path in (*hf_dir.iterdir(), *lite_dir.iterdir()):
        path.unlink()

    observations = MetaAREResultNormalizer(evaluator_fingerprint="meta-are-test/v1").normalize(
        raw_results=raw_results,
        hf_trace_dir=hf_dir,
        lite_trace_dir=lite_dir,
        candidate_id=CANDIDATE_ID,
        requested_cases=(Case("case-a", "train", {}),),
        repetitions=1,
        artifact_dir=tmp_path / "normalized",
    )

    assert len(observations) == 1
    assert observations[0].disposition == "task_failure"
    assert observations[0].score == 0.0
    assert observations[0].is_valid is True
    assert observations[0].trace is None


@pytest.mark.parametrize("mutation", ["duplicate", "unexpected", "missing"])
def test_normalizer_rejects_inexact_requested_coverage(tmp_path: Path, mutation: str) -> None:
    from autosaddler.v2.plugins.meta_are.evaluator import MetaAREResultNormalizer

    rows = [_row("case-a", run_number=None, status="success", score=1.0)]
    if mutation == "duplicate":
        rows.append(dict(rows[0]))
    elif mutation == "unexpected":
        rows.append(_row("case-b", run_number=None, status="success", score=1.0))
    elif mutation == "missing":
        rows.clear()
    raw_results, hf_dir, lite_dir = _write_results(tmp_path, rows)

    with pytest.raises(ValueError, match="coverage|duplicate|unexpected|missing"):
        MetaAREResultNormalizer(evaluator_fingerprint="meta-are-test/v1").normalize(
            raw_results=raw_results,
            hf_trace_dir=hf_dir,
            lite_trace_dir=lite_dir,
            candidate_id=CANDIDATE_ID,
            requested_cases=(Case("case-a", "train", {}),),
            repetitions=1,
            artifact_dir=tmp_path / "normalized",
        )


def test_normalizer_rejects_malformed_json_and_non_finite_scores(tmp_path: Path) -> None:
    from autosaddler.v2.plugins.meta_are.evaluator import MetaAREResultNormalizer

    raw_results = tmp_path / "output.jsonl"
    raw_results.write_text("not-json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="JSON|json"):
        MetaAREResultNormalizer(evaluator_fingerprint="meta-are-test/v1").normalize(
            raw_results=raw_results,
            hf_trace_dir=tmp_path,
            lite_trace_dir=tmp_path,
            candidate_id=CANDIDATE_ID,
            requested_cases=(Case("case-a", "train", {}),),
            repetitions=1,
            artifact_dir=tmp_path / "normalized",
        )

    raw_results.write_text(
        json.dumps(_row("case-a", run_number=None, status="success", score=float("nan"))) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="finite|score"):
        MetaAREResultNormalizer(evaluator_fingerprint="meta-are-test/v1").normalize(
            raw_results=raw_results,
            hf_trace_dir=tmp_path,
            lite_trace_dir=tmp_path,
            candidate_id=CANDIDATE_ID,
            requested_cases=(Case("case-a", "train", {}),),
            repetitions=1,
            artifact_dir=tmp_path / "normalized-2",
        )


def _row(
    case_id: str,
    *,
    run_number: int | None,
    status: str,
    score: float,
    has_exception: bool = False,
) -> dict[str, object]:
    metadata: dict[str, object] = {
        "scenario_id": case_id,
        "status": status,
        "has_exception": has_exception,
        "rationale": "fixture rationale",
    }
    if run_number is not None:
        metadata["run_number"] = run_number
    return {
        "task_id": case_id,
        "trace_id": f"/producer/{case_id}-{run_number}.json",
        "score": score,
        "metadata": metadata,
    }


def _write_results(
    root: Path,
    rows: list[dict[str, object]],
) -> tuple[Path, Path, Path]:
    raw_results = root / "output.jsonl"
    raw_results.write_text(
        "".join(json.dumps(row, allow_nan=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    hf_dir = root / "hf"
    lite_dir = root / "lite"
    hf_dir.mkdir()
    lite_dir.mkdir()
    for index, row in enumerate(rows):
        metadata = row["metadata"]
        assert isinstance(metadata, dict)
        case_id = str(metadata["scenario_id"])
        run_number = metadata.get("run_number")
        trace_name = f"trace-{index}.json"
        (hf_dir / trace_name).write_text(
            json.dumps(
                {
                    "version": "are_simulation_v1",
                    "judge_llm_usage_stats": {
                        "total_llm_calls": 1,
                        "prompt_tokens": [7],
                        "completion_tokens": [3],
                        "total_tokens": [10],
                        "reasoning_tokens": [1],
                        "completion_duration": [0.2],
                    },
                    "metadata": {
                        "definition": {
                            "scenario_id": case_id,
                            "run_number": run_number,
                            "has_exception": metadata["has_exception"],
                        },
                        "simulation": {"model_id": "task-model"},
                        "annotation": {"validation_decision": "Valid"},
                        "runner_config": {
                            "judge_engine_config": {
                                "model_name": "judge-model",
                                "provider": "azure",
                            }
                        },
                    },
                    "events": [],
                    "completed_events": [],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (lite_dir / trace_name).write_text(
            json.dumps(
                {
                    "scenario_id": case_id,
                    "validation_rationale": metadata["rationale"],
                    "per_agent_interaction_histories": {},
                    "model_id": "task-model",
                    "per_agent_llm_usage_stats": {
                        "default": {
                            "total_llm_calls": 1,
                            "prompt_tokens": [8],
                            "completion_tokens": [2],
                            "total_tokens": [10],
                            "reasoning_tokens": [0],
                            "completion_duration": [0.1],
                        }
                    },
                }
            )
            + "\n",
            encoding="utf-8",
        )
    return raw_results, hf_dir, lite_dir
