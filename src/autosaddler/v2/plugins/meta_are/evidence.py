from __future__ import annotations

from collections.abc import Mapping

from autosaddler.v2.core.domain import ArtifactRef, Evaluation, JsonValue, sha256_digest, to_json_value
from autosaddler.v2.storage.local import LocalRunStore


class MetaAREEvidenceBuilder:
    def __init__(self, *, store: LocalRunStore) -> None:
        self.store = store

    def build(self, evaluation: Evaluation) -> ArtifactRef:
        if evaluation.split != "train":
            raise ValueError("Meta-ARE optimization evidence may use training evaluations only")

        case_records = []
        for case_id in evaluation.requested_case_ids:
            observations = sorted(
                (item for item in evaluation.observations if item.case_id == case_id),
                key=lambda item: item.repetition,
            )
            if not observations:
                raise ValueError(f"Meta-ARE evaluation has no observations for requested case: {case_id}")
            per_repetition = []
            for observation in observations:
                trace = self._trace(observation.trace)
                interactions = trace.get("interactions", [])
                if not isinstance(interactions, list):
                    raise TypeError("Meta-ARE normalized trace interactions must be a list")
                usage = trace.get("usage", {})
                if not isinstance(usage, Mapping):
                    raise TypeError("Meta-ARE normalized trace usage must be an object")
                trace_digests = trace.get("trace_digests", {})
                if not isinstance(trace_digests, Mapping):
                    raise TypeError("Meta-ARE normalized trace digests must be an object")
                rationale = trace.get(
                    "validation_rationale",
                    observation.metadata.get("validation_rationale", ""),
                )
                record: dict[str, JsonValue] = {
                    "repetition": observation.repetition,
                    "disposition": observation.disposition,
                    "score": observation.score,
                    "producer_status": observation.metadata.get("producer_status"),
                    "validation_rationale": str(rationale),
                    "interactions": interactions,
                    "usage": dict(usage),
                    "trace_digests": dict(trace_digests),
                    "trace_artifact": (
                        {
                            "uri": observation.trace.uri,
                            "sha256": observation.trace.sha256,
                        }
                        if observation.trace is not None
                        else None
                    ),
                    "exception_type": observation.metadata.get("exception_type"),
                    "exception_message": observation.metadata.get("exception_message"),
                }
                record_value = to_json_value(record)
                assert isinstance(record_value, dict)
                per_repetition.append(record_value)
            case_records.append(
                {
                    "case_id": case_id,
                    "consistency": _consistency(observations),
                    "per_repetition": per_repetition,
                }
            )

        evidence_id = sha256_digest(evaluation.evaluation_id)
        return self.store.write_json(
            f"evidence/{evidence_id.removeprefix('sha256:')}/evidence.json",
            {
                "schema_version": "autosaddler-meta-are-evidence/v1",
                "evaluation_id": evaluation.evaluation_id,
                "candidate_id": evaluation.candidate_id,
                "split": "train",
                "purpose": evaluation.purpose,
                "case_records": case_records,
            },
            kind="meta-are-training-evidence",
        )

    def _trace(self, artifact: ArtifactRef | None) -> Mapping[str, object]:
        if artifact is None:
            return {}
        path = self.store.run_dir / artifact.uri
        if not path.is_file():
            raise ValueError(f"Meta-ARE trace artifact is missing: {artifact.uri}")
        payload = path.read_bytes()
        if artifact.sha256 is None or sha256_digest(payload) != artifact.sha256:
            raise ValueError(f"Meta-ARE trace artifact digest drift: {artifact.uri}")
        value = self.store.read_json(artifact.uri)
        if not isinstance(value, Mapping):
            raise TypeError("Meta-ARE normalized trace must contain an object")
        return value


def _consistency(observations) -> str:
    outcomes = {(item.disposition, item.score) for item in observations}
    if len(outcomes) > 1:
        return "intermittent"
    disposition, _ = next(iter(outcomes))
    return "consistent_pass" if disposition == "success" else "consistent_failure"
