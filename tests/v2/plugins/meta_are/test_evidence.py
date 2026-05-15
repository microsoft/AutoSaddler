from __future__ import annotations

from pathlib import Path

import pytest

from autosaddler.v2.core.domain import ArtifactRef, Evaluation, Observation, sha256_digest
from autosaddler.v2.storage.local import LocalRunStore

CANDIDATE_ID = sha256_digest("candidate")


def test_evidence_is_lossless_digest_checked_and_preserves_repetitions(tmp_path: Path) -> None:
    from autosaddler.v2.plugins.meta_are.evidence import MetaAREEvidenceBuilder

    store = _store(tmp_path)
    traces = tuple(
        store.write_json(
            f"evaluations/eval/trace-{repetition}.json",
            {
                "schema_version": "autosaddler-meta-are-observation/v1",
                "validation_rationale": "r" * 100,
                "interactions": [
                    {"role": "user", "content": "u" * 100},
                    {"role": "assistant", "content": "a" * 100},
                    {"role": "tool-response", "content": "t" * 100},
                ],
                "usage": {
                    "task_agent": {"calls": 1, "total_tokens": 12},
                    "judge": {"calls": 0, "total_tokens": 0},
                },
                "trace_digests": {
                    "hf": sha256_digest(f"hf-{repetition}"),
                    "lite": sha256_digest(f"lite-{repetition}"),
                },
            },
            kind="meta-are-normalized-trace",
        )
        for repetition in range(2)
    )
    evaluation = _evaluation(traces)

    evidence_ref = MetaAREEvidenceBuilder(store=store).build(evaluation)
    evidence = store.read_json(evidence_ref.uri)

    assert evidence["schema_version"] == "autosaddler-meta-are-evidence/v1"
    assert evidence["case_records"][0]["consistency"] == "intermittent"
    repetitions = evidence["case_records"][0]["per_repetition"]
    assert [item["repetition"] for item in repetitions] == [0, 1]
    assert all(len(item["interactions"]) == 3 for item in repetitions)
    assert all(len(item["validation_rationale"]) == 100 for item in repetitions)
    assert all(len(item["interactions"][2]["content"]) == 100 for item in repetitions)
    assert repetitions[1]["exception_type"] == "ProviderError"
    assert repetitions[0]["usage"]["judge"]["calls"] == 0
    assert "completed_events" not in str(evidence)
    assert "ground_truth" not in str(evidence)


def test_evidence_rejects_non_training_data_and_trace_drift(tmp_path: Path) -> None:
    from autosaddler.v2.plugins.meta_are.evidence import MetaAREEvidenceBuilder

    store = _store(tmp_path)
    trace = store.write_json(
        "evaluations/eval/trace.json",
        {"schema_version": "autosaddler-meta-are-observation/v1", "interactions": []},
        kind="meta-are-normalized-trace",
    )
    builder = MetaAREEvidenceBuilder(store=store)
    development = Evaluation(
        evaluation_id=sha256_digest("development"),
        candidate_id=CANDIDATE_ID,
        split="development",
        purpose="development",
        iteration=0,
        requested_case_ids=("dev-a",),
        observations=(
            Observation.create(
                candidate_id=CANDIDATE_ID,
                case_id="dev-a",
                split="development",
                repetition=0,
                disposition="success",
                score=1.0,
                evaluator_fingerprint="meta-are-test/v1",
                trace=trace,
            ),
        ),
        artifact_dir=ArtifactRef(uri="quarantine/dev/eval", kind="evaluation-directory"),
    )
    with pytest.raises(ValueError, match="training evaluations"):
        builder.build(development)

    trace_path = store.run_dir / trace.uri
    trace_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="digest|drift"):
        builder.build(_evaluation((trace, trace)))


def _store(tmp_path: Path) -> LocalRunStore:
    store = LocalRunStore(run_dir=tmp_path / "run", run_id="run")
    store.initialize(
        resolved_config={"schema_version": "autosaddler/v2"},
        resolved_entities={"resolved/component_graph.json": {"scenario": "meta_are"}},
    )
    return store


def _evaluation(traces: tuple[ArtifactRef, ArtifactRef]) -> Evaluation:
    observations = (
        Observation.create(
            candidate_id=CANDIDATE_ID,
            case_id="train-a",
            split="train",
            repetition=0,
            disposition="success",
            score=1.0,
            evaluator_fingerprint="meta-are-test/v1",
            trace=traces[0],
            metadata={"producer_status": "success"},
        ),
        Observation.create(
            candidate_id=CANDIDATE_ID,
            case_id="train-a",
            split="train",
            repetition=1,
            disposition="execution_error",
            score=None,
            evaluator_fingerprint="meta-are-test/v1",
            trace=traces[1],
            metadata={
                "producer_status": "failed",
                "exception_type": "ProviderError",
                "exception_message": "provider unavailable",
            },
        ),
    )
    return Evaluation(
        evaluation_id=sha256_digest("evaluation"),
        candidate_id=CANDIDATE_ID,
        split="train",
        purpose="train_before",
        iteration=0,
        requested_case_ids=("train-a",),
        observations=observations,
        artifact_dir=ArtifactRef(uri="evaluations/eval", kind="evaluation-directory"),
    )
