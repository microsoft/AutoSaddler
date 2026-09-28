from __future__ import annotations

from pathlib import Path


def iteration_artifact_name(
    iteration: int,
    candidate_idx: int,
    artifact_type: str,
) -> str:
    return f"iter{iteration:02d}_c{candidate_idx}_{artifact_type}.json"


def iteration_artifact_path(
    cycle_dir: str | Path,
    iteration: int,
    candidate_idx: int,
    artifact_type: str,
) -> Path:
    return Path(cycle_dir) / iteration_artifact_name(
        iteration,
        candidate_idx,
        artifact_type,
    )
