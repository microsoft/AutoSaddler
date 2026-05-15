from __future__ import annotations

import json
import math
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path

from autosaddler.v2.core.domain import (
    ArtifactRef,
    Candidate,
    Case,
    Cost,
    Evaluation,
    Observation,
    canonical_json,
    sha256_digest,
)
from autosaddler.v2.core.ports import EvaluationContext, HarnessSpace
from autosaddler.v2.plugins.meta_are.runner import MetaARERunner
from autosaddler.v2.prompting.models import Usage


class IncompleteMetaAREResultsError(ValueError):
    """The benchmark omitted one or more requested case repetitions."""


class MetaAREResultNormalizer:
    def __init__(self, *, evaluator_fingerprint: str) -> None:
        if not evaluator_fingerprint:
            raise ValueError("Meta-ARE evaluator fingerprint must be non-empty")
        self.evaluator_fingerprint = evaluator_fingerprint

    def normalize(
        self,
        *,
        raw_results: Path,
        hf_trace_dir: Path,
        lite_trace_dir: Path,
        candidate_id: str,
        requested_cases: Sequence[Case],
        repetitions: int,
        artifact_dir: Path,
        artifact_root: Path | None = None,
        allow_incomplete: bool = False,
    ) -> tuple[Observation, ...]:
        artifact_root = artifact_dir if artifact_root is None else artifact_root
        if not artifact_dir.resolve().is_relative_to(artifact_root.resolve()):
            raise ValueError("Meta-ARE artifact directory must be inside its artifact root")
        rows = _json_lines(raw_results)
        expected = {(case.case_id, repetition) for case in requested_cases for repetition in range(repetitions)}
        by_key: dict[tuple[str, int], dict[str, object]] = {}
        for row in rows:
            metadata = row.get("metadata")
            if not isinstance(metadata, dict):
                raise TypeError("Meta-ARE result metadata must be an object")
            case_id = metadata.get("scenario_id")
            if not isinstance(case_id, str) or not case_id:
                raise TypeError("Meta-ARE result scenario_id must be a non-empty string")
            repetition = _normalize_run_number(metadata.get("run_number"), repetitions)
            key = (case_id, repetition)
            if key in by_key:
                raise ValueError(f"Meta-ARE result coverage contains duplicate key: {key}")
            if key not in expected:
                raise ValueError(f"Meta-ARE result coverage contains unexpected key: {key}")
            score = row.get("score")
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score):
                raise ValueError(f"Meta-ARE result score must be finite for key: {key}")
            by_key[key] = row
        missing = sorted(expected - by_key.keys())
        if missing and not allow_incomplete:
            raise IncompleteMetaAREResultsError(f"Meta-ARE result coverage is missing requested keys: {missing}")

        hf_traces = _hf_trace_index(hf_trace_dir, repetitions=repetitions)
        lite_traces = _lite_trace_index(
            lite_trace_dir,
            hf_trace_dir=hf_trace_dir,
            hf_traces=hf_traces,
        )
        case_by_id = {case.case_id: case for case in requested_cases}
        artifact_dir.mkdir(parents=True, exist_ok=True)
        observations = []
        for key in sorted(by_key):
            case_id, repetition = key
            row = by_key[key]
            metadata = row["metadata"]
            assert isinstance(metadata, dict)
            status = metadata.get("status")
            has_exception = metadata.get("has_exception") is True
            scenario_timed_out = has_exception and metadata.get("exception_type") == "ScenarioTimeoutError"
            hf_path = hf_traces.get(key)
            lite_path = lite_traces.get(key)
            if scenario_timed_out:
                disposition = "task_failure"
                normalized_score = float(row["score"])
            elif has_exception or hf_path is None or lite_path is None:
                disposition = "execution_error"
                normalized_score = None
            elif status == "success":
                disposition = "success"
                normalized_score = float(row["score"])
            elif status == "failed":
                disposition = "task_failure"
                normalized_score = float(row["score"])
            else:
                disposition = "invalid"
                normalized_score = None

            output_path = artifact_dir / "observations" / f"{case_id}-{repetition}.json"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(canonical_json(row) + "\n", encoding="utf-8")
            lite_value = _json_object(lite_path) if lite_path is not None else {}
            hf_value = _json_object(hf_path) if hf_path is not None else {}
            model_usage = _model_usage_records(hf_value, lite_value)
            task_agent_usage = [item for item in model_usage if item["role"] == "task_agent"]
            judge_usage = [item for item in model_usage if item["role"] == "judge"]
            normalized_trace_path = None
            if hf_path is not None and lite_path is not None:
                normalized_trace_path = artifact_dir / "traces" / f"{case_id}-{repetition}.json"
                normalized_trace_path.parent.mkdir(parents=True, exist_ok=True)
                normalized_trace_path.write_text(
                    canonical_json(
                        {
                            "schema_version": "autosaddler-meta-are-observation/v1",
                            "validation_decision": _validation_decision(hf_value, lite_value),
                            "validation_rationale": str(
                                lite_value.get(
                                    "validation_rationale",
                                    metadata.get("rationale", ""),
                                )
                            ),
                            "interactions": _interactions(lite_value),
                            "usage": {
                                "task_agent": _usage_summary(task_agent_usage),
                                "judge": _usage_summary(judge_usage),
                            },
                            "trace_digests": {
                                "hf": sha256_digest(hf_path.read_bytes()),
                                "lite": sha256_digest(lite_path.read_bytes()),
                            },
                        }
                    )
                    + "\n",
                    encoding="utf-8",
                )
            observations.append(
                Observation.create(
                    candidate_id=candidate_id,
                    case_id=case_id,
                    split=case_by_id[case_id].split,
                    repetition=repetition,
                    disposition=disposition,  # type: ignore[arg-type]
                    score=normalized_score,
                    evaluator_fingerprint=self.evaluator_fingerprint,
                    output=_artifact(output_path, artifact_root, "meta-are-result-row"),
                    trace=(
                        _artifact(
                            normalized_trace_path,
                            artifact_root,
                            "meta-are-normalized-trace",
                        )
                        if normalized_trace_path is not None
                        else None
                    ),
                    attempts=1,
                    cost=Cost(rollouts=1),
                    metadata={
                        "producer_status": str(status),
                        "has_exception": has_exception,
                        "producer_run_number": metadata.get("run_number"),
                        "validation_rationale": str(metadata.get("rationale", "")),
                        "lite_trace_uri": (
                            lite_path.relative_to(artifact_root).as_posix()
                            if lite_path is not None and lite_path.is_relative_to(artifact_root)
                            else None
                        ),
                        "lite_trace_sha256": (sha256_digest(lite_path.read_bytes()) if lite_path is not None else None),
                        "task_agent_calls": len(task_agent_usage),
                        "judge_calls": len(judge_usage),
                        "model_usage": model_usage,
                    },
                )
            )
        return tuple(observations)


class MetaAREEvaluator:
    def __init__(
        self,
        *,
        harness_space: HarnessSpace,
        runner: MetaARERunner,
        evaluator_fingerprint: str,
        artifact_root: Path,
        infrastructure_retries: int,
    ) -> None:
        if infrastructure_retries < 0:
            raise ValueError("Meta-ARE infrastructure retries cannot be negative")
        self.harness_space = harness_space
        self.runner = runner
        self.artifact_root = artifact_root.resolve()
        self.infrastructure_retries = infrastructure_retries
        self.normalizer = MetaAREResultNormalizer(evaluator_fingerprint=evaluator_fingerprint)
        self.fingerprint = evaluator_fingerprint

    async def evaluate(
        self,
        candidate: Candidate,
        cases: Sequence[Case],
        context: EvaluationContext,
    ) -> Evaluation:
        if not cases or any(case.split != context.split for case in cases):
            raise ValueError("Meta-ARE evaluator cases must match EvaluationContext.split")
        latest: dict[tuple[str, int], Observation] = {}
        for case in cases:
            for repetition in range(context.repetitions):
                completed = context.attempt_sink.completed(
                    candidate_id=candidate.candidate_id,
                    case_id=case.case_id,
                    repetition=repetition,
                )
                if completed is not None:
                    latest[(case.case_id, repetition)] = completed

        case_by_id = {case.case_id: case for case in cases}
        maximum_attempts = self.infrastructure_retries + 1
        while invalid_keys := _invalid_keys(cases, context.repetitions, latest):
            current_retry_case_ids = tuple(
                case_id for case_id in case_by_id if any(key[0] == case_id for key in invalid_keys)
            )
            current_retry_cases = tuple(case_by_id[case_id] for case_id in current_retry_case_ids)
            runner_attempt_number = _runner_attempt_number(
                context.artifact_dir,
                current_retry_cases,
                context.repetitions,
                latest,
            )
            if runner_attempt_number > maximum_attempts:
                raise RuntimeError(
                    f"Meta-ARE evaluation exhausted infrastructure retries for keys: {sorted(invalid_keys)}"
                )
            attempt_dir = context.artifact_dir / f"runner-attempt-{runner_attempt_number:03d}"
            retry_cases = _runner_request_cases(
                attempt_dir,
                case_by_id,
                current_retry_cases,
                context.repetitions,
            )
            runner_keys = {
                (case.case_id, repetition) for case in retry_cases for repetition in range(context.repetitions)
            }
            durable_completion = (attempt_dir / "completion.json").is_file()
            accounting_keys = runner_keys - latest.keys() if durable_completion else runner_keys
            active_attempts: dict[tuple[str, int], tuple[str, int]] = {}
            for case_id, repetition in sorted(accounting_keys):
                pending = context.attempt_sink.pending(
                    candidate_id=candidate.candidate_id,
                    case_id=case_id,
                    repetition=repetition,
                )
                if pending is not None and durable_completion:
                    active_attempts[(case_id, repetition)] = pending
                else:
                    active_attempts[(case_id, repetition)] = context.attempt_sink.start(
                        candidate_id=candidate.candidate_id,
                        case_id=case_id,
                        repetition=repetition,
                    )
            materialized = self.harness_space.materialize(candidate, "evaluate")
            try:
                run_result = self.runner.run(
                    materialized=materialized,
                    cases=retry_cases,
                    repetitions=context.repetitions,
                    attempt_dir=attempt_dir,
                )
            except (subprocess.TimeoutExpired, RuntimeError) as error:
                _write_infrastructure_error(attempt_dir, error)
                _fail_attempts(context, active_attempts, type(error).__name__)
                if runner_attempt_number >= maximum_attempts:
                    raise RuntimeError("Meta-ARE benchmark failed after bounded infrastructure retries") from error
                continue
            except Exception:
                _fail_attempts(context, active_attempts, "runner_contract_error")
                raise
            try:
                normalized = self.normalizer.normalize(
                    raw_results=run_result.raw_results,
                    hf_trace_dir=run_result.hf_trace_dir,
                    lite_trace_dir=run_result.lite_trace_dir,
                    candidate_id=candidate.candidate_id,
                    requested_cases=retry_cases,
                    repetitions=context.repetitions,
                    artifact_dir=(context.artifact_dir / f"normalized/attempt-{runner_attempt_number:03d}"),
                    artifact_root=self.artifact_root,
                    allow_incomplete=True,
                )
            except Exception:
                _fail_attempts(context, active_attempts, "invalid_result")
                raise
            by_key = {(observation.case_id, observation.repetition): observation for observation in normalized}
            returned_keys = active_attempts.keys() & by_key.keys()
            missing_keys = active_attempts.keys() - by_key.keys()
            for key in sorted(returned_keys):
                observation = by_key[key]
                attempt_id, event_attempt_number = active_attempts[key]
                raw_usage = observation.metadata.get("model_usage", [])
                if not isinstance(raw_usage, list):
                    raise TypeError("Meta-ARE observation model_usage must be a list")
                for item in raw_usage:
                    context.attempt_sink.observe_usage(attempt_id, _usage_from_record(item))
                observation = Observation.create(
                    candidate_id=observation.candidate_id,
                    case_id=observation.case_id,
                    split=observation.split,
                    repetition=observation.repetition,
                    disposition=observation.disposition,
                    score=observation.score,
                    evaluator_fingerprint=self.fingerprint,
                    objectives=observation.objectives,
                    output=observation.output,
                    trace=observation.trace,
                    attempts=event_attempt_number,
                    cost=Cost(rollouts=1),
                    metadata=observation.metadata,
                )
                context.attempt_sink.complete(attempt_id, observation, observation.cost)
                latest[key] = observation
            if missing_keys:
                error = IncompleteMetaAREResultsError(
                    f"Meta-ARE result coverage is missing requested keys: {sorted(missing_keys)}"
                )
                _write_infrastructure_error(attempt_dir, error)
                _fail_attempts(
                    context,
                    {key: active_attempts[key] for key in missing_keys},
                    "incomplete_result",
                )
                if runner_attempt_number >= maximum_attempts:
                    raise RuntimeError(
                        "Meta-ARE benchmark returned incomplete results after bounded retries"
                    ) from error
                continue
            if runner_attempt_number >= maximum_attempts:
                remaining = _invalid_keys(cases, context.repetitions, latest)
                if remaining:
                    raise RuntimeError(
                        f"Meta-ARE evaluation retained invalid keys after bounded retries: {sorted(remaining)}"
                    )
        observations = sorted(latest.values(), key=lambda item: (item.case_id, item.repetition))
        return Evaluation(
            evaluation_id=sha256_digest(context.operation_id),
            candidate_id=candidate.candidate_id,
            split=context.split,
            purpose=context.purpose,
            iteration=context.iteration,
            requested_case_ids=tuple(case.case_id for case in cases),
            observations=tuple(observations),
            artifact_dir=ArtifactRef(
                uri=context.artifact_dir.relative_to(self.artifact_root).as_posix(),
                kind="meta-are-evaluation-directory",
            ),
        )


def _runner_request_cases(
    attempt_dir: Path,
    case_by_id: Mapping[str, Case],
    requested_cases: Sequence[Case],
    repetitions: int,
) -> tuple[Case, ...]:
    request_path = attempt_dir / "request.json"
    if request_path.is_file():
        value = json.loads(request_path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise TypeError("Meta-ARE durable runner request must be an object")
        case_ids = value.get("case_ids")
        stored_repetitions = value.get("repetitions")
        if (
            not isinstance(case_ids, list)
            or not all(isinstance(case_id, str) for case_id in case_ids)
            or len(case_ids) != len(set(case_ids))
            or stored_repetitions != repetitions
        ):
            raise ValueError("Meta-ARE durable runner request is malformed or drifted")
        missing = [case_id for case_id in case_ids if case_id not in case_by_id]
        if missing:
            raise ValueError(f"Meta-ARE durable runner request has unknown cases: {missing}")
        return tuple(case_by_id[case_id] for case_id in case_ids)

    attempt_dir.mkdir(parents=True, exist_ok=True)
    value = {
        "case_ids": [case.case_id for case in requested_cases],
        "repetitions": repetitions,
    }
    request_path.write_text(canonical_json(value) + "\n", encoding="utf-8")
    return tuple(requested_cases)


def _fail_attempts(
    context: EvaluationContext,
    active_attempts: Mapping[tuple[str, int], tuple[str, int]],
    error_kind: str,
) -> None:
    for attempt_id, _ in active_attempts.values():
        context.attempt_sink.fail(attempt_id, error_kind, Cost(rollouts=1))


def _normalize_run_number(value: object, repetitions: int) -> int:
    if repetitions == 1 and value is None:
        return 0
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("Meta-ARE run_number must be one-based, or null for one repetition")
    repetition = value - 1
    if repetition < 0 or repetition >= repetitions:
        raise ValueError(f"Meta-ARE run_number is outside requested coverage: {value}")
    return repetition


def _invalid_keys(
    cases: Sequence[Case],
    repetitions: int,
    latest: Mapping[tuple[str, int], Observation],
) -> set[tuple[str, int]]:
    expected = {(case.case_id, repetition) for case in cases for repetition in range(repetitions)}
    return {key for key in expected if key not in latest or not latest[key].is_valid}


def _runner_attempt_number(
    artifact_dir: Path,
    cases: Sequence[Case],
    repetitions: int,
    latest: Mapping[tuple[str, int], Observation],
) -> int:
    existing = sorted(artifact_dir.glob("runner-attempt-[0-9][0-9][0-9]"))
    if not existing:
        return 1
    last = existing[-1]
    try:
        attempt_number = int(last.name.rsplit("-", 1)[1])
    except ValueError as error:
        raise ValueError(f"Malformed Meta-ARE runner attempt directory: {last}") from error
    if (last / "infrastructure-error.json").is_file():
        return attempt_number + 1
    expected = {(case.case_id, repetition) for case in cases for repetition in range(repetitions)}
    if all(key in latest and latest[key].attempts >= attempt_number for key in expected):
        return attempt_number + 1
    return attempt_number


def _write_infrastructure_error(attempt_dir: Path, error: Exception) -> None:
    attempt_dir.mkdir(parents=True, exist_ok=True)
    (attempt_dir / "infrastructure-error.json").write_text(
        canonical_json(
            {
                "error_type": type(error).__name__,
                "error": str(error)[:2_000],
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _json_lines(path: Path) -> tuple[dict[str, object], ...]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Malformed JSON in Meta-ARE output at line {line_number}") from error
        if not isinstance(value, dict):
            raise TypeError(f"Meta-ARE output line {line_number} must be an object")
        rows.append(value)
    return tuple(rows)


def _hf_trace_index(
    root: Path,
    *,
    repetitions: int,
) -> dict[tuple[str, int], Path]:
    result: dict[tuple[str, int], Path] = {}
    if not root.is_dir():
        return result
    for path in sorted(root.rglob("*.json")):
        value = _json_object(path)
        metadata = value.get("metadata")
        definition = metadata.get("definition") if isinstance(metadata, dict) else None
        if not isinstance(definition, dict):
            raise TypeError(f"Meta-ARE HF trace definition is malformed: {path}")
        case_id = definition.get("scenario_id")
        run_number = definition.get("run_number")
        if not isinstance(case_id, str) or not case_id:
            raise TypeError(f"Meta-ARE hf trace has no scenario_id: {path}")
        key = (case_id, _normalize_run_number(run_number, repetitions))
        if key in result:
            raise ValueError(f"Duplicate Meta-ARE hf trace identity: {key}")
        result[key] = path
    return result


def _lite_trace_index(
    root: Path,
    *,
    hf_trace_dir: Path,
    hf_traces: Mapping[tuple[str, int], Path],
) -> dict[tuple[str, int], Path]:
    result: dict[tuple[str, int], Path] = {}
    if not root.is_dir():
        return result
    matched_paths: set[Path] = set()
    for key, hf_path in hf_traces.items():
        path = root / hf_path.relative_to(hf_trace_dir)
        if not path.is_file():
            continue
        value = _json_object(path)
        if value.get("scenario_id") != key[0]:
            raise ValueError(f"Meta-ARE lite trace scenario_id does not match its HF trace: {path}")
        result[key] = path
        matched_paths.add(path.resolve())
    unmatched = sorted(path for path in root.rglob("*.json") if path.resolve() not in matched_paths)
    if unmatched:
        raise ValueError(f"Meta-ARE lite trace has no matching HF trace: {unmatched[0]}")
    return result


def _json_object(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Meta-ARE trace must contain an object: {path}")
    return value


def _artifact(path: Path, root: Path, kind: str) -> ArtifactRef:
    payload = path.read_bytes()
    return ArtifactRef(
        uri=path.relative_to(root).as_posix(),
        kind=kind,
        sha256=sha256_digest(payload),
        bytes=len(payload),
    )


def _model_usage_records(
    hf: Mapping[str, object],
    lite: Mapping[str, object],
) -> list[dict[str, object]]:
    records = _usage_records(
        lite.get("per_agent_llm_usage_stats"),
        role="task_agent",
        model=lite.get("model_id"),
    )
    raw_judge = lite.get("per_judge_llm_usage_stats", lite.get("judge_llm_usage_stats"))
    if raw_judge is None:
        raw_judge = hf.get("per_judge_llm_usage_stats", hf.get("judge_llm_usage_stats"))
    records.extend(_usage_records(raw_judge, role="judge", model=_judge_model(hf)))
    return records


def _usage_records(
    raw_stats: object,
    *,
    role: str,
    model: object,
) -> list[dict[str, object]]:
    if raw_stats is None:
        return []
    if not isinstance(raw_stats, dict):
        raise TypeError(f"Meta-ARE {role} usage stats must be an object")
    if "total_llm_calls" in raw_stats:
        stats_by_agent = {role: raw_stats}
    else:
        stats_by_agent = raw_stats
    records = []
    for agent_id in sorted(stats_by_agent):
        stats = stats_by_agent[agent_id]
        if not isinstance(agent_id, str) or not isinstance(stats, dict):
            raise TypeError(f"Meta-ARE {role} usage entries must be named objects")
        calls = stats.get("total_llm_calls", 0)
        if isinstance(calls, bool) or not isinstance(calls, int) or calls < 0:
            raise TypeError(f"Meta-ARE {role} total_llm_calls must be nonnegative")
        prompt = _usage_series(stats.get("prompt_tokens"), calls, "prompt_tokens")
        completion = _usage_series(stats.get("completion_tokens"), calls, "completion_tokens")
        total = _usage_series(stats.get("total_tokens"), calls, "total_tokens")
        reasoning = _usage_series(stats.get("reasoning_tokens"), calls, "reasoning_tokens")
        duration = _usage_series(
            stats.get("completion_duration"),
            calls,
            "completion_duration",
            integer=False,
        )
        for index in range(calls):
            input_tokens = int(prompt[index] or 0)
            output_tokens = int(completion[index] or 0)
            reasoning_tokens = int(reasoning[index] or 0)
            if reasoning_tokens > output_tokens:
                raise ValueError("Meta-ARE reasoning tokens exceed completion tokens")
            records.append(
                {
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "reasoning_tokens": reasoning_tokens,
                    "provider_reported_total_tokens": (int(total[index]) if total[index] is not None else None),
                    "model": model if isinstance(model, str) else None,
                    "role": role,
                    "duration_seconds": (float(duration[index]) if duration[index] is not None else None),
                    "agent_id": agent_id,
                    "usage_incomplete": prompt[index] is None or completion[index] is None,
                }
            )
    return records


def _usage_series(
    value: object,
    calls: int,
    label: str,
    *,
    integer: bool = True,
) -> list[int | float | None]:
    if value is None:
        return [None] * calls
    values = value if isinstance(value, list) else [value]
    if len(values) != calls:
        raise ValueError(f"Meta-ARE {label} length does not match total_llm_calls")
    for item in values:
        valid = isinstance(item, int) if integer else isinstance(item, (int, float))
        if isinstance(item, bool) or not valid or item < 0:
            raise TypeError(f"Meta-ARE {label} values must be nonnegative numbers")
    return values


def _usage_summary(records: Sequence[Mapping[str, object]]) -> dict[str, int]:
    return {
        "calls": len(records),
        "input_tokens": sum(int(item["input_tokens"]) for item in records),
        "output_tokens": sum(int(item["output_tokens"]) for item in records),
        "reasoning_tokens": sum(int(item["reasoning_tokens"]) for item in records),
        "total_tokens": sum(int(item["input_tokens"]) + int(item["output_tokens"]) for item in records),
    }


def _usage_from_record(value: object) -> Usage:
    if not isinstance(value, Mapping):
        raise TypeError("Meta-ARE model usage record must be an object")
    return Usage(
        input_tokens=int(value.get("input_tokens", 0)),
        output_tokens=int(value.get("output_tokens", 0)),
        reasoning_tokens=int(value.get("reasoning_tokens", 0)),
        provider_reported_total_tokens=(
            int(value["provider_reported_total_tokens"])
            if value.get("provider_reported_total_tokens") is not None
            else None
        ),
        model=value.get("model") if isinstance(value.get("model"), str) else None,
        role=str(value.get("role", "task_agent")),
        duration_seconds=(float(value["duration_seconds"]) if value.get("duration_seconds") is not None else None),
        agent_id=value.get("agent_id") if isinstance(value.get("agent_id"), str) else None,
        usage_incomplete=value.get("usage_incomplete") is True,
    )


def _judge_model(hf: Mapping[str, object]) -> object:
    metadata = hf.get("metadata")
    runner_config = metadata.get("runner_config") if isinstance(metadata, dict) else None
    judge_config = runner_config.get("judge_engine_config") if isinstance(runner_config, dict) else None
    return judge_config.get("model_name") if isinstance(judge_config, dict) else None


def _interactions(lite: Mapping[str, object]) -> list[object]:
    histories = lite.get("per_agent_interaction_histories")
    if not isinstance(histories, dict):
        return []
    interactions = []
    for agent_id in sorted(histories):
        history = histories[agent_id]
        if not isinstance(history, list):
            raise TypeError(f"Meta-ARE lite trace history must be a list: {agent_id}")
        for item in history:
            if isinstance(item, dict):
                interactions.append({"agent_id": agent_id, **item})
            else:
                interactions.append({"agent_id": agent_id, "value": item})
    return interactions


def _validation_decision(
    hf: Mapping[str, object],
    lite: Mapping[str, object],
) -> object:
    decision = lite.get("validation_decision")
    if decision is not None:
        return decision
    metadata = hf.get("metadata")
    annotation = metadata.get("annotation") if isinstance(metadata, dict) else None
    return annotation.get("validation_decision") if isinstance(annotation, dict) else None
