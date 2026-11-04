from __future__ import annotations

import json
from pathlib import Path

import pytest

from autosaddler.v2.core.domain import Case, sha256_digest

FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "baseline_failed"
CANDIDATE_ID = sha256_digest("candidate")
PRICING = {
    "input_usd_per_million": 2.0,
    "cached_input_usd_per_million": 0.5,
    "output_usd_per_million": 8.0,
}


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
    assert model_usage[0]["cached_input_tokens"] is None
    assert model_usage[0]["uncached_input_tokens"] is None
    assert "provider_reported_input_tokens" not in model_usage[0]
    assert model_usage[0]["usage_incomplete"] is True
    assert observation.cost.currency_amount is None
    trace = json.loads((tmp_path / observation.trace.uri).read_text())
    assert trace["usage"]["task_agent"]["cached_input_tokens"] is None


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


@pytest.mark.parametrize("priced", [False, True])
def test_cached_usage_is_recorded_independently_of_pricing(tmp_path: Path, priced: bool) -> None:
    from autosaddler.v2.plugins.meta_are.evaluator import MetaAREResultNormalizer, _usage_from_record

    raw_results, hf_dir, lite_dir = _write_results(
        tmp_path, [_row("case-a", run_number=None, status="success", score=1.0)]
    )
    trace_path = lite_dir / "trace-0.json"
    trace = json.loads(trace_path.read_text())
    trace["per_agent_llm_usage_stats"]["default"]["cached_input_tokens"] = [3]
    trace_path.write_text(json.dumps(trace))
    observation = MetaAREResultNormalizer(
        evaluator_fingerprint="test",
        task_agent_pricing=PRICING if priced else None,
    ).normalize(
        raw_results=raw_results,
        hf_trace_dir=hf_dir,
        lite_trace_dir=lite_dir,
        candidate_id=CANDIDATE_ID,
        requested_cases=(Case("case-a", "train", {}),),
        repetitions=1,
        artifact_dir=tmp_path / "normalized",
    )[0]

    usage = _usage_from_record(observation.metadata["model_usage"][0])
    assert usage.cached_input_tokens == 3
    assert usage.uncached_input_tokens == 5
    assert not usage.usage_incomplete
    assert observation.trace is not None
    trace = json.loads((tmp_path / "normalized" / observation.trace.uri).read_text())
    assert trace["usage"]["task_agent"]["cached_input_tokens"] == 3
    if priced:
        assert observation.cost.currency_amount == pytest.approx(27.5 / 1_000_000)
    else:
        assert observation.cost.currency_amount is None
        assert observation.cost.cached_input_tokens is None
        assert observation.cost.uncached_input_tokens is None


@pytest.mark.parametrize("cached_tokens", [None, 0, 3])
def test_judge_usage_preserves_cached_input(cached_tokens: int | None) -> None:
    from autosaddler.v2.plugins.meta_are.evaluator import _model_usage_records, _usage_from_record

    stats = {"total_llm_calls": 1, "prompt_tokens": [7], "completion_tokens": [3]}
    if cached_tokens is not None:
        stats["cached_input_tokens"] = [cached_tokens]
    records = _model_usage_records({"judge_llm_usage_stats": stats}, {})

    assert len(records) == 1
    record = records[0]
    assert record["role"] == "judge"
    assert record["cached_input_tokens"] == cached_tokens
    assert record["uncached_input_tokens"] == (None if cached_tokens is None else 7 - cached_tokens)
    usage = _usage_from_record(record)
    assert usage.usage_incomplete is (cached_tokens is None)
    if cached_tokens is not None:
        assert usage.cached_input_tokens == cached_tokens
        assert usage.uncached_input_tokens == 7 - cached_tokens


@pytest.mark.parametrize("repetitions", [1, 2, 3, 5])
def test_pricing_splits_total_input_and_excludes_judge(
    tmp_path: Path, repetitions: int
) -> None:
    from autosaddler.v2.plugins.meta_are.evaluator import MetaAREResultNormalizer, _usage_from_record

    raw_results, hf_dir, lite_dir = _write_results(
        tmp_path, [_row("case-a", run_number=number + 1, status="failed", score=0.0) for number in range(repetitions)]
    )
    for path in lite_dir.iterdir():
        trace = json.loads(path.read_text())
        stats = trace["per_agent_llm_usage_stats"]["default"]
        stats["cached_input_tokens"] = [3]
        stats["prompt_tokens"] = [8]
        stats["reasoning_tokens"] = [1]
        path.write_text(json.dumps(trace))
    observations = MetaAREResultNormalizer(
        evaluator_fingerprint="priced",
        task_agent_pricing=PRICING,
    ).normalize(
        raw_results=raw_results, hf_trace_dir=hf_dir, lite_trace_dir=lite_dir,
        candidate_id=CANDIDATE_ID, requested_cases=(Case("case-a", "train", {}),),
        repetitions=repetitions, artifact_dir=tmp_path / "normalized",
    )
    assert len(observations) == repetitions
    for observation in observations:
        assert observation.score == 0.0
        assert observation.cost.currency_amount == pytest.approx(27.5 / 1_000_000)
        assert observation.cost.input_tokens == 8
        assert observation.cost.cached_input_tokens == 3
        assert observation.cost.uncached_input_tokens == 5
        assert observation.cost.output_tokens == 2
        assert not observation.objectives
        usage = _usage_from_record(observation.metadata["model_usage"][0])
        assert usage.input_tokens == 8
        assert usage.cached_input_tokens == 3
        assert usage.uncached_input_tokens == 5
        assert "input_token_semantics" not in observation.metadata["model_usage"][0]
        assert usage.input_token_semantics == "unknown"
        assert "provider_reported_input_tokens" not in observation.metadata["model_usage"][0]
        assert usage.provider_reported_input_tokens is None
        assert not usage.usage_incomplete


@pytest.mark.parametrize("lite_effort,hf_effort", [(None, None), ("low", "high"), ("high", "low")])
def test_pricing_depends_on_usage_not_reasoning_effort(
    tmp_path: Path, lite_effort: str | None, hf_effort: str | None
) -> None:
    from autosaddler.v2.plugins.meta_are.evaluator import MetaAREResultNormalizer

    raw_results, hf_dir, lite_dir = _write_results(
        tmp_path, [_row("case-a", run_number=None, status="success", score=1.0)]
    )
    lite_path = lite_dir / "trace-0.json"
    lite = json.loads(lite_path.read_text())
    lite["reasoning_effort"] = lite_effort
    lite["per_agent_llm_usage_stats"]["default"]["cached_input_tokens"] = [3]
    lite_path.write_text(json.dumps(lite))
    hf_path = hf_dir / "trace-0.json"
    hf = json.loads(hf_path.read_text())
    hf["metadata"]["runner_config"]["reasoning_effort"] = hf_effort
    hf_path.write_text(json.dumps(hf))

    observation = MetaAREResultNormalizer(
        evaluator_fingerprint="priced", task_agent_pricing=PRICING,
    ).normalize(
        raw_results=raw_results, hf_trace_dir=hf_dir, lite_trace_dir=lite_dir,
        candidate_id=CANDIDATE_ID, requested_cases=(Case("case-a", "train", {}),),
        repetitions=1, artifact_dir=tmp_path / "normalized",
    )[0]

    assert observation.score == 1.0
    assert observation.cost.currency_amount == pytest.approx(27.5 / 1_000_000)


@pytest.mark.parametrize("lite_model,hf_model", [(None, None), ("different-model", "task-model"), ("task-model", "different-model")])
def test_pricing_does_not_require_matching_model_metadata(
    tmp_path: Path, lite_model: str | None, hf_model: str | None
) -> None:
    from autosaddler.v2.plugins.meta_are.evaluator import MetaAREResultNormalizer, _usage_from_record

    raw_results, hf_dir, lite_dir = _write_results(
        tmp_path, [_row("case-a", run_number=None, status="success", score=1.0)]
    )
    lite_path = lite_dir / "trace-0.json"
    lite = json.loads(lite_path.read_text())
    if lite_model is None:
        del lite["model_id"]
    else:
        lite["model_id"] = lite_model
    lite["per_agent_llm_usage_stats"]["default"]["cached_input_tokens"] = [3]
    lite_path.write_text(json.dumps(lite))
    hf_path = hf_dir / "trace-0.json"
    hf = json.loads(hf_path.read_text())
    if hf_model is None:
        del hf["metadata"]["simulation"]["model_id"]
    else:
        hf["metadata"]["simulation"]["model_id"] = hf_model
    hf_path.write_text(json.dumps(hf))

    observation = MetaAREResultNormalizer(
        evaluator_fingerprint="priced", task_agent_pricing=PRICING,
    ).normalize(
        raw_results=raw_results, hf_trace_dir=hf_dir, lite_trace_dir=lite_dir,
        candidate_id=CANDIDATE_ID, requested_cases=(Case("case-a", "train", {}),),
        repetitions=1, artifact_dir=tmp_path / "normalized",
    )[0]

    assert observation.score == 1.0
    assert observation.cost.currency_amount == pytest.approx(27.5 / 1_000_000)
    assert _usage_from_record(observation.metadata["model_usage"][0]).model == lite_model


@pytest.mark.parametrize("mutation", ["missing_usage", "missing_input", "missing_output", "missing_cache", "missing_calls", "bad_cache", "zero_cost"])
def test_pricing_handles_missing_zero_and_malformed_usage(tmp_path: Path, mutation: str) -> None:
    from autosaddler.v2.plugins.meta_are.evaluator import MetaAREResultNormalizer

    raw_results, hf_dir, lite_dir = _write_results(
        tmp_path, [_row("case-a", run_number=None, status="success", score=1.0)]
    )
    path = lite_dir / "trace-0.json"
    trace = json.loads(path.read_text())
    stats = trace["per_agent_llm_usage_stats"]["default"]
    stats["cached_input_tokens"] = [3]
    if mutation == "missing_usage":
        del trace["per_agent_llm_usage_stats"]
    elif mutation.startswith("missing_"):
        del stats[{"missing_input": "prompt_tokens", "missing_output": "completion_tokens", "missing_cache": "cached_input_tokens", "missing_calls": "total_llm_calls"}[mutation]]
    elif mutation == "bad_cache":
        stats["cached_input_tokens"] = [9]
    elif mutation == "zero_cost":
        stats.update(prompt_tokens=[0], completion_tokens=[0], cached_input_tokens=[0])
    path.write_text(json.dumps(trace))
    normalizer = MetaAREResultNormalizer(evaluator_fingerprint="priced", task_agent_pricing=PRICING)
    arguments = dict(
        raw_results=raw_results, hf_trace_dir=hf_dir, lite_trace_dir=lite_dir,
        candidate_id=CANDIDATE_ID, requested_cases=(Case("case-a", "train", {}),),
        repetitions=1, artifact_dir=tmp_path / "normalized",
    )
    if mutation in ("missing_calls", "bad_cache"):
        with pytest.raises(ValueError, match="cached input tokens exceed|length does not match total_llm_calls"):
            normalizer.normalize(**arguments)
        return

    observation = normalizer.normalize(**arguments)[0]
    assert observation.disposition == "success"
    assert observation.score == 1.0
    assert observation.is_valid
    task_usage = [item for item in observation.metadata["model_usage"] if item["role"] == "task_agent"]
    if mutation == "zero_cost":
        assert observation.cost.currency_amount == 0.0
        assert observation.cost.input_tokens == 0
        assert observation.cost.cached_input_tokens == 0
        assert observation.cost.uncached_input_tokens == 0
        assert observation.cost.output_tokens == 0
        assert len(task_usage) == 1
        assert not task_usage[0]["usage_incomplete"]
    else:
        assert observation.cost.currency_amount is None
        assert observation.cost.cached_input_tokens is None
        assert observation.cost.uncached_input_tokens is None
        if mutation == "missing_usage":
            assert task_usage == []
        else:
            assert len(task_usage) == 1
            assert task_usage[0]["usage_incomplete"] is True


@pytest.mark.parametrize("missing_field", ["prompt_tokens", "cached_input_tokens", "completion_tokens"])
def test_infrastructure_failure_keeps_partial_usage_without_pricing(
    tmp_path: Path, missing_field: str
) -> None:
    from autosaddler.v2.plugins.meta_are.evaluator import MetaAREResultNormalizer, _usage_from_record

    raw_results, hf_dir, lite_dir = _write_results(
        tmp_path, [_row("case-a", run_number=None, status="failed", score=0.0, has_exception=True)]
    )
    path = lite_dir / "trace-0.json"
    trace = json.loads(path.read_text())
    stats = trace["per_agent_llm_usage_stats"]["default"]
    stats["prompt_tokens"] = [8]
    stats["cached_input_tokens"] = [3]
    del stats[missing_field]
    path.write_text(json.dumps(trace))
    observation = MetaAREResultNormalizer(
        evaluator_fingerprint="priced",
        task_agent_pricing=PRICING,
    ).normalize(
        raw_results=raw_results, hf_trace_dir=hf_dir, lite_trace_dir=lite_dir,
        candidate_id=CANDIDATE_ID, requested_cases=(Case("case-a", "train", {}),),
        repetitions=1, artifact_dir=tmp_path / "normalized",
    )[0]

    assert observation.disposition == "execution_error"
    assert observation.score is None
    assert observation.cost.currency_amount is None
    assert observation.cost.cached_input_tokens is None
    assert observation.cost.uncached_input_tokens is None
    record = observation.metadata["model_usage"][0]
    usage = _usage_from_record(record)
    assert usage.usage_incomplete
    assert usage.cached_input_tokens == (0 if missing_field == "cached_input_tokens" else 3)
    if missing_field == "prompt_tokens":
        assert usage.input_tokens == 3
        assert record["uncached_input_tokens"] is None
        assert usage.provider_reported_input_tokens is None
    if missing_field == "cached_input_tokens":
        assert observation.trace is not None
        normalized_trace = json.loads((tmp_path / "normalized" / observation.trace.uri).read_text())
        assert normalized_trace["usage"]["task_agent"]["cached_input_tokens"] is None


@pytest.mark.parametrize("usage", ["complete", "missing", "partial", "zero"])
def test_scenario_timeout_cost_depends_on_available_usage(tmp_path: Path, usage: str) -> None:
    from autosaddler.v2.plugins.meta_are.evaluator import MetaAREResultNormalizer

    row = _row("case-a", run_number=None, status="failed", score=0.0, has_exception=True)
    row["metadata"]["exception_type"] = "ScenarioTimeoutError"
    raw_results, hf_dir, lite_dir = _write_results(tmp_path, [row])
    path = lite_dir / "trace-0.json"
    trace = json.loads(path.read_text())
    trace["per_agent_llm_usage_stats"]["default"]["cached_input_tokens"] = [0]
    if usage == "partial":
        del trace["per_agent_llm_usage_stats"]["default"]["cached_input_tokens"]
    elif usage == "zero":
        for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
            trace["per_agent_llm_usage_stats"]["default"][key] = [0]
    path.write_text(json.dumps(trace))
    if usage == "missing":
        for path in (*hf_dir.iterdir(), *lite_dir.iterdir()):
            path.unlink()
    normalizer = MetaAREResultNormalizer(
        evaluator_fingerprint="priced", task_agent_pricing=PRICING,
    )
    arguments = dict(
        raw_results=raw_results, hf_trace_dir=hf_dir, lite_trace_dir=lite_dir,
        candidate_id=CANDIDATE_ID, requested_cases=(Case("case-a", "train", {}),),
        repetitions=1, artifact_dir=tmp_path / "normalized",
    )
    observation = normalizer.normalize(**arguments)[0]
    if usage == "complete":
        assert observation.disposition == "task_failure"
        assert observation.score == 0.0
        assert observation.is_valid
        assert observation.cost.currency_amount == pytest.approx(32 / 1_000_000)
    else:
        assert observation.disposition == "execution_error"
        assert observation.score is None
        assert not observation.is_valid
        assert observation.metadata["incomplete_reason"] == "scenario_timeout_incomplete_cost"
        assert observation.cost.currency_amount == (0.0 if usage == "zero" else None)
        if usage == "missing":
            assert observation.cost.cached_input_tokens is None
            assert observation.cost.uncached_input_tokens is None
            assert observation.metadata["model_usage"] == []


def test_recorded_trace_without_cached_usage_keeps_unknown_cost(tmp_path: Path) -> None:
    from autosaddler.v2.plugins.meta_are.evaluator import MetaAREResultNormalizer

    observation = MetaAREResultNormalizer(
        evaluator_fingerprint="priced", task_agent_pricing=PRICING,
    ).normalize(
        raw_results=FIXTURE_ROOT / "output.jsonl",
        hf_trace_dir=FIXTURE_ROOT / "hf", lite_trace_dir=FIXTURE_ROOT / "lite",
        candidate_id=CANDIDATE_ID, requested_cases=(Case("scenario_universe_29_096dyu", "train", {}),),
        repetitions=1, artifact_dir=tmp_path,
    )[0]

    assert observation.disposition == "task_failure"
    assert observation.score == 0.0
    assert observation.is_valid
    assert observation.cost.currency_amount is None
    assert observation.cost.cached_input_tokens is None
    assert observation.cost.uncached_input_tokens is None
    assert observation.metadata["model_usage"][0]["cached_input_tokens"] is None
    assert observation.metadata["model_usage"][0]["usage_incomplete"] is True


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
