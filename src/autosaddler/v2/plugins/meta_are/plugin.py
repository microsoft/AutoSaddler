from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from autosaddler.v2.core.domain import JsonValue
from autosaddler.v2.core.ports import ScenarioComponents
from autosaddler.v2.harness.git import GitHarnessSpace
from autosaddler.v2.prompting.assets import prompt_source_entities
from autosaddler.v2.plugins.meta_are.config import MetaARESettings
from autosaddler.v2.plugins.meta_are.evaluator import MetaAREEvaluator
from autosaddler.v2.plugins.meta_are.evidence import MetaAREEvidenceBuilder
from autosaddler.v2.plugins.meta_are.prompt_pack import (
    MetaAREPromptPack,
    meta_are_prompt_composition_entity,
)
from autosaddler.v2.plugins.meta_are.runner import MetaARERunner
from autosaddler.v2.plugins.meta_are.verification import MetaAREVerifier
from autosaddler.v2.storage.local import LocalRunStore

REQUIRED_CAPABILITIES = frozenset({"read_workspace", "edit_workspace", "run_commands", "load_skills"})


def build_meta_are_components(
    *,
    settings: Mapping[str, JsonValue],
    base_dir: Path,
    run_dir: Path,
    store: LocalRunStore,
    ledger: object,
) -> ScenarioComponents:
    del ledger
    resolved = MetaARESettings.from_mapping(settings, base_dir=base_dir)
    train_cases, development_cases = resolved.load_cases()
    verifier = MetaAREVerifier(
        import_check=resolved.import_check,
        verification_timeout_seconds=resolved.verification_timeout_seconds,
        train_case_ids=tuple(case.case_id for case in train_cases),
    )
    harness = GitHarnessSpace(
        source_repo=resolved.source_repo,
        base_revision=resolved.base_revision,
        store_root=run_dir / "candidates",
        worktree_root=run_dir / "worktrees",
        writable_paths=resolved.writable_paths,
        forbidden_paths=resolved.forbidden_paths,
        verifier=verifier,
    )
    runner = MetaARERunner(
        demo_filesystem_root=resolved.demo_filesystem_root,
        dataset_root=resolved.dataset_root,
        agent=resolved.agent,
        model=resolved.model,
        model_provider=resolved.model_provider,
        model_wire_api=resolved.model_wire_api,
        model_endpoint=resolved.model_endpoint,
        reasoning_effort=resolved.reasoning_effort,
        responses_runtime_sha256=resolved.responses_runtime_sha256,
        judge_model=resolved.judge_model,
        judge_provider=resolved.judge_provider,
        judge_endpoint=resolved.judge_endpoint,
        benchmark_config=resolved.benchmark_config,
        benchmark_split=resolved.benchmark_split,
        timeout_seconds=resolved.scenario_timeout_seconds,
        process_completion_grace_seconds=resolved.process_completion_grace_seconds,
        max_concurrent=resolved.max_concurrent,
    )
    evaluator = MetaAREEvaluator(
        harness_space=harness,
        runner=runner,
        evaluator_fingerprint=f"meta-are/v1:{resolved.execution_fingerprint}",
        artifact_root=run_dir,
        infrastructure_retries=resolved.infrastructure_retries,
    )
    prompt_pack = MetaAREPromptPack(
        store=store,
        writable_paths=resolved.writable_paths,
        capability_phase_iterations=resolved.capability_phase_iterations,
    )
    return ScenarioComponents(
        name="meta_are",
        version="1",
        harness_space=harness,
        evaluator=evaluator,
        evidence_builder=MetaAREEvidenceBuilder(store=store),
        prompt_pack=prompt_pack,
        train_cases=train_cases,
        development_cases=development_cases,
        required_capabilities=REQUIRED_CAPABILITIES,
        evaluation_repetitions=resolved.repetitions,
        resolved_entities=_resolved_entities(resolved, train_cases, development_cases),
    )


def _resolved_entities(settings, train_cases, development_cases):
    return {
        **prompt_source_entities(
            plugin_root=Path(__file__).parent,
            plugin_name="meta_are",
        ),
        "resolved/prompts/compositions.json": meta_are_prompt_composition_entity(),
        "resolved/sources/harness.json": {
            "type": "git",
            "source_repo": str(settings.source_repo),
            "base_revision": settings.base_revision,
            "pyproject_sha256": settings.pyproject_sha256,
            "uv_lock_sha256": settings.uv_lock_sha256,
            "execution_fingerprint": settings.execution_fingerprint,
        },
        "resolved/sources/dataset.json": {
            "type": "local_only",
            "root": str(settings.dataset_root),
            "source_revision": settings.dataset_source_revision,
            "source_descriptor": str(settings.dataset_source_descriptor),
            "content_digest": settings.dataset_digest,
            "train_manifest": str(settings.train_manifest),
            "development_manifest": str(settings.development_manifest),
            "test_manifest": str(settings.test_manifest),
            "train": [{"case_id": case.case_id, **dict(case.payload)} for case in train_cases],
            "development": [{"case_id": case.case_id, **dict(case.payload)} for case in development_cases],
            "test": {
                "state": "opaque_to_optimizer",
                "manifest_opened": True,
                "payloads_opened": False,
                "case_count": settings.test_case_count,
            },
        },
        "resolved/sources/demo_filesystem.json": {
            "root": str(settings.demo_filesystem_root),
            "source_revision": settings.demo_filesystem_source_revision,
            "source_descriptor": str(settings.demo_filesystem_source_descriptor),
            "manifest": str(settings.demo_filesystem_manifest),
            "content_digest": settings.demo_filesystem_digest,
        },
        "resolved/benchmark.json": {
            "config": settings.benchmark_config,
            "split": settings.benchmark_split,
            "agent": settings.agent,
            "model": settings.model,
            "model_provider": settings.model_provider,
            "model_wire_api": settings.model_wire_api,
            "model_endpoint": settings.model_endpoint,
            "reasoning_effort": settings.reasoning_effort,
            "responses_runtime_sha256": settings.responses_runtime_sha256,
            "judge_model": settings.judge_model,
            "judge_provider": settings.judge_provider,
            "judge_endpoint": settings.judge_endpoint,
            "timeout_seconds": settings.scenario_timeout_seconds,
            "process_completion_grace_seconds": settings.process_completion_grace_seconds,
            "repetitions": settings.repetitions,
            "max_concurrent": settings.max_concurrent,
            "infrastructure_retries": settings.infrastructure_retries,
        },
        "resolved/mutation_scope.json": {
            "writable_paths": [path.as_posix() for path in settings.writable_paths],
            "forbidden_paths": [path.as_posix() for path in settings.forbidden_paths],
            "capability_phase_iterations": settings.capability_phase_iterations,
            "verification_timeout_seconds": settings.verification_timeout_seconds,
        },
    }
