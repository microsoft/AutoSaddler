from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import autosaddler
import autosaddler.v1
import autosaddler.v2
import pytest
import yaml

from autosaddler.v1.core.adapter import EvaluationBatch
from autosaddler.v1.core.engine import GEPAEngine
from autosaddler.v1.proposer.base import CandidateProposal
from autosaddler.v1.strategies.eval_policy import FullEvaluationPolicy
from autosaddler.v2.config.registry import build_runtime


def test_v1_and_v2_use_explicit_package_namespaces() -> None:
    assert autosaddler.__all__ == ["v1", "v2"]
    assert callable(autosaddler.v1.optimize)
    assert autosaddler.v2.Candidate.__module__.startswith("autosaddler.v2.")

    obsolete_v1_namespaces = (
        "autosaddler.api",
        "autosaddler.adapters",
        "autosaddler.core",
        "autosaddler.logging",
        "autosaddler.proposer",
        "autosaddler.sdk_session",
        "autosaddler.strategies",
        "autosaddler.utils",
    )
    for namespace in obsolete_v1_namespaces:
        with pytest.raises(ModuleNotFoundError) as exc_info:
            importlib.import_module(namespace)
        assert exc_info.value.name == namespace


def test_v1_meta_are_cli_dispatches_to_v1_optimize(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    from autosaddler.v1 import sdk_session
    from autosaddler.v1.adapters.meta_are_adapter import meta_are_adapter, optimize as optimize_cli

    class FakeWorktreePool:
        def get_or_create(self, candidate: dict[str, str], patch: Any) -> tuple[Path, bool]:
            del candidate, patch
            return tmp_path / "seed-worktree", True

    class FakeAdapter:
        def __init__(self, config: dict[str, Any]) -> None:
            del config
            self.cfg = SimpleNamespace(sdk_config=None)
            self._worktree_pool = FakeWorktreePool()

        def set_session_root(self, session_root: Path) -> None:
            self.session_root = session_root

    optimize_calls: list[dict[str, Any]] = []

    def fake_optimize(**kwargs: Any) -> SimpleNamespace:
        optimize_calls.append(kwargs)
        return SimpleNamespace(
            best_idx=0,
            best_candidate={"prompt": "best"},
            candidates=[{"prompt": "best"}],
            val_aggregate_scores=[1.0],
        )

    config = {
        "dataset": {"train_file": "train.json", "val_file": "val.json"},
        "adapter": {
            "meta_are_repo": str(tmp_path / "meta-are"),
            "session_root_base": str(tmp_path / "runs"),
        },
        "optimization": {"max_metric_calls": 1},
        "sdk": {},
    }
    monkeypatch.setattr(optimize_cli, "load_config", lambda path: config)
    monkeypatch.setattr(optimize_cli, "load_scenario_ids", lambda path: [Path(path).stem])
    monkeypatch.setattr(optimize_cli, "_build_autosaddler_proposer", lambda **kwargs: "proposer")
    monkeypatch.setattr(meta_are_adapter, "MetaAREAdapter", FakeAdapter)
    monkeypatch.setattr(meta_are_adapter, "MetaAREDataInst", lambda scenario_id: scenario_id)
    monkeypatch.setattr(sdk_session, "SdkConfig", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(autosaddler.v1, "optimize", fake_optimize)
    monkeypatch.setattr(
        "sys.argv",
        ["autosaddler-v1", "--config", str(tmp_path / "config.yaml")],
    )

    optimize_cli.main()

    assert len(optimize_calls) == 1
    assert optimize_calls[0]["trainset"] == ["train"]
    assert optimize_calls[0]["valset"] == ["val"]
    result_files = list((tmp_path / "runs").glob("*/best_candidate.json"))
    assert len(result_files) == 1


def test_v1_meta_are_cli_dry_run_does_not_construct_adapter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from autosaddler.v1.adapters.meta_are_adapter import meta_are_adapter, optimize as optimize_cli

    config = {
        "dataset": {"train_file": "train.json", "val_file": "val.json"},
        "adapter": {
            "meta_are_repo": str(tmp_path / "meta-are"),
            "session_root_base": str(tmp_path / "runs"),
        },
    }
    monkeypatch.setattr(optimize_cli, "load_config", lambda path: config)
    monkeypatch.setattr(optimize_cli, "load_scenario_ids", lambda path: [Path(path).stem])

    def unexpected_adapter(*args: object, **kwargs: object) -> object:
        raise AssertionError("dry-run must not construct a Meta-ARE adapter")

    monkeypatch.setattr(meta_are_adapter, "MetaAREAdapter", unexpected_adapter)
    monkeypatch.setattr(
        "sys.argv",
        ["autosaddler-v1", "--config", str(tmp_path / "config.yaml"), "--dry-run"],
    )

    optimize_cli.main()

    assert not (tmp_path / "runs").exists()


def test_v1_evo_dag_wrapper_uses_package_import_root(tmp_path: Path) -> None:
    from autosaddler.v1.proposer.autosaddler.prompt_builder import install_evo_dag_cli

    install_evo_dag_cli(str(tmp_path), "/unused/worktree")

    wrapper = (tmp_path / "bin" / "evo-dag").read_text(encoding="utf-8")
    package_source_root = Path(autosaddler.__file__).resolve().parent.parent
    assert f'export PYTHONPATH="{package_source_root}:${{PYTHONPATH:-}}"' in wrapper


class RecordingAdapter:
    def __init__(self) -> None:
        self.calls: list[tuple[tuple[str, ...], str]] = []

    def evaluate(
        self,
        batch: list[str],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[dict[str, str], str]:
        version = candidate["prompt"]
        self.calls.append((tuple(batch), version))
        score = 1.0 if version == "improved" else 0.0
        return EvaluationBatch(
            outputs=[f"{version}:{case_id}" for case_id in batch],
            scores=[score for _ in batch],
            trajectories=None,
        )


class OneIterationProposer:
    def __init__(self, adapter: RecordingAdapter) -> None:
        self.adapter = adapter
        self.trainset = ["train-a", "train-b"]

    def propose(self, state: Any) -> CandidateProposal[int]:
        batch = list(self.trainset)
        parent = state.program_candidates[0]
        child = {"prompt": "improved"}
        before = self.adapter.evaluate(batch, parent).scores
        after = self.adapter.evaluate(batch, child).scores
        state.increment_evals(len(batch) * 2)
        return CandidateProposal(
            candidate=child,
            parent_program_ids=[0],
            subsample_indices=[0, 1],
            subsample_scores_before=before,
            subsample_scores_after=after,
        )


class StopAfterOneIteration:
    def __call__(self, state: Any) -> bool:
        return state.i >= 0


class RecordingLogger:
    def log(self, message: str) -> None:
        del message


class RecordingTracker:
    def log_config(self, config: dict[str, Any]) -> None:
        del config

    def log_metrics(self, metrics: dict[str, Any], step: int | None = None) -> None:
        del metrics, step

    def log_table(self, name: str, columns: list[str], data: list[list[Any]]) -> None:
        del name, columns, data

    def log_summary(self, summary: dict[str, Any]) -> None:
        del summary


def test_current_loop_preserves_matched_verification_and_dev_gate(tmp_path: Path) -> None:
    adapter = RecordingAdapter()
    proposer = OneIterationProposer(adapter)
    engine = GEPAEngine(
        adapter=adapter,
        run_dir=None,
        valset=["dev-a", "dev-b"],
        seed_candidate={"prompt": "seed"},
        perfect_score=1.0,
        seed=7,
        reflective_proposer=proposer,
        merge_proposer=None,
        frontier_type="instance",
        logger=RecordingLogger(),
        experiment_tracker=RecordingTracker(),
        stop_callback=StopAfterOneIteration(),
        val_evaluation_policy=FullEvaluationPolicy(),
    )

    state = engine.run()

    assert adapter.calls == [
        (("dev-a", "dev-b"), "seed"),
        (("train-a", "train-b"), "seed"),
        (("train-a", "train-b"), "improved"),
        (("dev-a", "dev-b"), "improved"),
    ]
    assert state.total_num_evals == 8
    assert state.program_candidates == [{"prompt": "seed"}, {"prompt": "improved"}]
    assert FullEvaluationPolicy().get_best_program(state) == 1

    runtime = build_runtime(_write_v2_config(tmp_path), run_id="synthetic-parity")
    result = runtime.engine.run()
    runtime_state = runtime.engine._state()
    seed_id, improved_id = runtime_state.accepted_candidate_ids
    candidate_labels = {seed_id: "seed", improved_id: "improved"}
    evaluation_order = [
        (
            tuple(event.payload["case_ids"]),
            candidate_labels[str(event.payload["candidate_id"])],
        )
        for event in runtime.store.events_of_type("EvaluationStarted")
    ]

    assert evaluation_order == adapter.calls
    acceptance = runtime.store.events_of_type("AcceptanceDecided")
    assert len(acceptance) == 1
    assert acceptance[0].payload["verdict"]["accepted"] is True
    development_gate = runtime.store.events_of_type("DevelopmentGateDecided")
    assert len(development_gate) == 1
    assert development_gate[0].payload["evaluate"] is True
    assert sum(entry["kind"] == "rollout" for entry in runtime.ledger.entries()) == state.total_num_evals
    assert sum(entry["kind"] == "session" for entry in runtime.ledger.entries()) == 3
    reflection_completed = runtime.store.events_of_type("DeferredWorkCompleted")[0]
    best_selected = runtime.store.events_of_type("BestCandidateSelected")[0]
    assert reflection_completed.sequence < best_selected.sequence
    assert result.selected_candidate_id == improved_id
    assert result.development_score == 1.0


def _write_v2_config(tmp_path: Path) -> Path:
    config = {
        "schema_version": "autosaddler/v2",
        "scenario": {
            "type": "fake",
            "settings": {
                "baseline": {"instruction": "baseline"},
                "target_component": "instruction",
                "improved_text": "improved",
                "train_case_ids": ["train-a", "train-b"],
                "development_case_ids": ["dev-a", "dev-b"],
            },
        },
        "optimization": {
            "task_selection": {"type": "fixed", "batch_size": 2},
            "acceptance": {"type": "matched_valid_strict_improvement"},
            "development": {"type": "full_on_accept"},
            "ranking": {"type": "mean_development_score"},
            "budget": {"max_rollouts": 100, "max_iterations": 1},
            "diagnosis_patch_timeout_seconds": 10,
        },
        "provider": {
            "type": "fake",
            "capabilities": ["read_workspace", "edit_workspace", "load_skills"],
            "settings": {},
        },
        "storage": {"type": "local", "run_root": str(tmp_path / "runs")},
    }
    path = tmp_path / "v2-parity.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path