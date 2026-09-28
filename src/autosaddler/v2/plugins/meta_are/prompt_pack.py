from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath

from autosaddler.v2.core.curriculum import CURRICULUM_SESSION_KINDS
from autosaddler.v2.core.domain import JsonValue, canonical_json, sha256_digest
from autosaddler.v2.prompting.assets import (
    PromptComposition,
    ResolvedPromptAssets,
    load_prompt_asset,
    prompt_composition_record,
    resolve_prompt_composition,
)
from autosaddler.v2.prompting.curriculum import (
    arm_decision_schema,
    arm_scoring_schema,
    build_curriculum_bundle,
    pattern_extraction_schema,
)
from autosaddler.v2.prompting.history import build_history_bundle
from autosaddler.v2.prompting.models import SessionSpec
from autosaddler.v2.storage.local import LocalRunStore

_ASSET_ROOT = Path(__file__).parent
_PROMPTING_ROOT = Path(__file__).parents[2] / "prompting"
_SESSION_CONTEXT_PATH = ".autosaddler/session_context.json"
_TRAINING_EVIDENCE_PATH = ".autosaddler/training_evidence.json"
_PROMPT_ASSETS_PATH = ".autosaddler/prompt_assets.json"
_EVIDENCE_BEFORE_PATH = ".autosaddler/training_evidence_before.json"
_EVIDENCE_AFTER_PATH = ".autosaddler/training_evidence_after.json"
_CURRICULUM_METHOD_NAMES = {
    "extract_patterns": "extract-patterns",
    "decide_arm": "decide-arm",
    "score_arms": "score-arms",
}


class MetaAREPromptPack:
    def __init__(
        self,
        *,
        store: LocalRunStore,
        writable_paths: Sequence[PurePosixPath],
        capability_phase_iterations: int,
        capability_transition_mode: str = "iterations",
        capability_phase_max_iterations: int = 0,
        train_case_ids: Sequence[str] = (),
    ) -> None:
        if not writable_paths:
            raise ValueError("Meta-ARE prompt pack requires writable paths")
        if capability_phase_iterations < 0:
            raise ValueError("Meta-ARE capability phase iterations cannot be negative")
        if capability_phase_max_iterations < 0:
            raise ValueError("Meta-ARE capability phase max iterations cannot be negative")
        if capability_transition_mode not in {"iterations", "full_coverage"}:
            raise ValueError(f"Unknown Meta-ARE capability transition mode: {capability_transition_mode}")
        if capability_transition_mode == "full_coverage" and not train_case_ids:
            raise ValueError("Meta-ARE full_coverage transitions require training case IDs")
        self.store = store
        self.writable_paths = tuple(writable_paths)
        self.capability_phase_iterations = capability_phase_iterations
        self.capability_transition_mode = capability_transition_mode
        self.capability_phase_max_iterations = capability_phase_max_iterations
        self.train_case_ids = frozenset(train_case_ids)

    def session(self, kind: str, context: Mapping[str, JsonValue]) -> SessionSpec:
        iteration = context.get("iteration")
        if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
            raise ValueError("Meta-ARE prompt context requires a nonnegative iteration")
        patch_phase = self.patch_phase(iteration)
        rendered_context: dict[str, JsonValue] = {
            **context,
            "patch_phase": patch_phase,
            "mutation_scope": [path.as_posix() for path in self.writable_paths],
        }
        workspace_files = {
            _SESSION_CONTEXT_PATH: canonical_json(rendered_context) + "\n",
        }
        workspace_files.update(build_history_bundle(self.store, context).workspace_files)
        curriculum = _is_curriculum(context)
        if curriculum:
            workspace_files.update(build_curriculum_bundle(self.store, context))

        if kind == "diagnose_patch":
            workspace_files[_TRAINING_EVIDENCE_PATH] = self._evidence(context.get("evidence"))
            schema = _diagnosis_schema()
            skill_paths = {
                "history-analysis": None,
                "diagnose": None,
                f"{patch_phase}-patch": f"skills/{patch_phase}-patch/SKILL.md",
                "patch-verification": "skills/patch-verification/SKILL.md",
            }
            mutation_label = patch_phase
        elif kind == "evolve":
            candidate_ids = _strings(context.get("candidate_ids"), "candidate_ids")
            component_source_options = _component_source_options(
                context.get("component_source_options"),
                candidate_ids,
            )
            schema = _evolve_schema(candidate_ids, component_source_options)
            skill_paths = {"history-analysis": None}
            mutation_label = None
        elif kind == "reflect":
            schema = _reflection_schema()
            skill_paths = {"history-analysis": None}
            mutation_label = None
        elif kind in CURRICULUM_SESSION_KINDS and curriculum:
            task_selection = _task_selection(context)
            mutation_label = None
            if kind == "extract_patterns":
                workspace_files[_EVIDENCE_BEFORE_PATH] = self._evidence(task_selection.get("train_before_evidence"))
                workspace_files[_EVIDENCE_AFTER_PATH] = self._evidence(task_selection.get("train_after_evidence"))
                failing = [
                    str(item["case_id"])
                    for key in ("pre_patch_failures", "post_patch_failures")
                    for item in _case_records(context.get(key), key)
                ]
                schema = pattern_extraction_schema("autosaddler-meta-are-pattern-extraction/v1", failing)
                skill_paths = {"history-analysis": None, "symptom-extract": None, "symptom-normalize": None}
            elif kind == "decide_arm":
                schema = arm_decision_schema("autosaddler-meta-are-arm-decision/v1")
                skill_paths = {"history-analysis": None}
            else:
                arm_ids = _strings(task_selection.get("arm_ids"), "task_selection.arm_ids")
                schema = arm_scoring_schema("autosaddler-meta-are-arm-scoring/v1", arm_ids)
                skill_paths = {"history-analysis": None, "progress-scoring": None}
        else:
            raise ValueError(f"Unknown Meta-ARE session kind: {kind}")

        resolved = _resolved_assets(kind, skill_paths, curriculum=curriculum)
        workspace_files[_PROMPT_ASSETS_PATH] = _provenance_manifest(resolved)

        return SessionSpec(
            kind=kind,
            system_context=resolved.system_context,
            task_prompt=resolved.task_prompt,
            skills=resolved.skills,
            output_schema=schema,
            workspace_files=workspace_files,
            capabilities=frozenset({"read_workspace", "edit_workspace", "run_commands", "load_skills"}),
            mutation_label=mutation_label,
        )

    def patch_phase(self, iteration: int) -> str:
        """Return the patch phase; full_coverage switches after every training case was sampled once."""
        if self.capability_transition_mode == "iterations":
            return "capability" if iteration + 1 <= self.capability_phase_iterations else "steering"
        if self.capability_phase_max_iterations and iteration >= self.capability_phase_max_iterations:
            return "steering"
        covered: set[str] = set()
        for event in self.store.events_of_type("BatchSampled"):
            sampled_iteration = event.payload.get("iteration")
            case_ids = event.payload.get("case_ids")
            if not isinstance(sampled_iteration, int) or not isinstance(case_ids, list):
                raise TypeError("BatchSampled events require an iteration and case IDs")
            if sampled_iteration >= iteration:
                continue
            covered.update(str(case_id) for case_id in case_ids)
            if self.train_case_ids <= covered:
                return "steering"
        return "capability"

    def _evidence(self, raw_artifact: JsonValue | None) -> str:
        if not isinstance(raw_artifact, Mapping):
            raise TypeError("Meta-ARE diagnosis context requires an evidence artifact")
        uri = raw_artifact.get("uri")
        digest = raw_artifact.get("sha256")
        if not isinstance(uri, str) or not isinstance(digest, str):
            raise TypeError("Meta-ARE evidence artifact requires uri and sha256")
        path = self.store.run_dir / uri
        payload = path.read_bytes()
        if sha256_digest(payload) != digest:
            raise ValueError("Meta-ARE training evidence digest drift")
        value = json.loads(payload)
        if not isinstance(value, dict) or value.get("schema_version") != "autosaddler-meta-are-evidence/v1":
            raise ValueError("Meta-ARE training evidence schema is invalid")
        return payload.decode("utf-8")


def _diagnosis_schema() -> Mapping[str, JsonValue]:
    required = ["schema_version", "intent", "diagnosis", "expected_effect", "changed_paths"]
    return {
        "type": "object",
        "required": required,
        "properties": {
            "schema_version": {"const": "autosaddler-meta-are-diagnosis/v1"},
            "intent": {"type": "string", "minLength": 1},
            "diagnosis": {"type": "string", "minLength": 1},
            "expected_effect": {"type": "string", "minLength": 1},
            "changed_paths": {
                "type": "array",
                "items": {"type": "string", "minLength": 1},
                "uniqueItems": True,
                "minItems": 1,
            },
        },
        "additionalProperties": False,
    }


def _evolve_schema(
    candidate_ids: tuple[str, ...],
    component_source_options: Mapping[str, tuple[str, ...]],
) -> Mapping[str, JsonValue]:
    required = ["schema_version", "parent_ids", "component_sources", "rationale"]
    return {
        "type": "object",
        "required": required,
        "properties": {
            "schema_version": {"const": "autosaddler-meta-are-evolution/v1"},
            "parent_ids": {
                "type": "array",
                "items": {"type": "string", "enum": list(candidate_ids)},
                "minItems": 1,
                "uniqueItems": True,
            },
            "component_sources": {
                "type": "object",
                "properties": {
                    path: {"type": "string", "enum": list(source_ids)}
                    for path, source_ids in component_source_options.items()
                },
                "additionalProperties": False,
            },
            "rationale": {"type": "string", "minLength": 1},
        },
        "additionalProperties": False,
    }


def _component_source_options(
    value: JsonValue | None,
    candidate_ids: tuple[str, ...],
) -> dict[str, tuple[str, ...]]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("component_source_options must be an object")
    allowed = set(candidate_ids)
    options: dict[str, tuple[str, ...]] = {}
    for path, raw_source_ids in value.items():
        if not isinstance(path, str) or not path:
            raise TypeError("component_source_options keys must be non-empty strings")
        source_ids = _strings(raw_source_ids, f"component_source_options[{path!r}]")
        if any(source_id not in allowed for source_id in source_ids):
            raise ValueError(f"component_source_options[{path!r}] contains an unknown candidate")
        options[path] = source_ids
    return options


def _reflection_schema() -> Mapping[str, JsonValue]:
    return {
        "type": "object",
        "required": ["schema_version", "lessons"],
        "properties": {
            "schema_version": {"const": "autosaddler-meta-are-reflection/v1"},
            "lessons": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["scope", "statement", "evidence_case_ids"],
                    "properties": {
                        "scope": {"type": "string", "minLength": 1},
                        "statement": {"type": "string", "minLength": 1},
                        "evidence_case_ids": {
                            "type": "array",
                            "items": {"type": "string"},
                            "uniqueItems": True,
                        },
                    },
                    "additionalProperties": False,
                },
            },
        },
        "additionalProperties": False,
    }


def _resolved_assets(
    kind: str,
    skill_paths: Mapping[str, str | None],
    *,
    curriculum: bool = False,
) -> ResolvedPromptAssets:
    shared_skills = {
        "history-analysis": "methodology/skills/history-analysis/SKILL.md",
        "diagnose": "methodology/skills/causal-diagnosis/SKILL.md",
        "patch-verification": "methodology/skills/verification-baseline/SKILL.md",
        "symptom-extract": "curriculum_methodology/skills/symptom-extraction/SKILL.md",
        "symptom-normalize": "curriculum_methodology/skills/symptom-normalization/SKILL.md",
        "progress-scoring": "curriculum_methodology/skills/progress-scoring/SKILL.md",
    }
    if kind in CURRICULUM_SESSION_KINDS:
        task_assets = (
            _shared_asset(
                f"curriculum_methodology/prompts/{_CURRICULUM_METHOD_NAMES[kind]}-method.md",
                f"methodology.prompt.{kind}",
            ),
            _plugin_asset(f"curriculum/prompts/{kind}.md", f"meta_are.prompt.{kind}"),
        )
    else:
        task_assets = (
            _shared_asset(f"methodology/prompts/{_method_name(kind)}-method.md", f"methodology.prompt.{kind}"),
            _plugin_asset(f"prompts/{kind}.md", f"meta_are.prompt.{kind}"),
            *(
                (
                    _shared_asset(
                        "curriculum_methodology/prompts/curriculum-context.md",
                        "methodology.prompt.curriculum_context",
                    ),
                )
                if curriculum
                else ()
            ),
        )
    return resolve_prompt_composition(
        PromptComposition(
            system_assets=(
                _shared_asset("methodology/system/optimizer-invariants.md", "methodology.system.invariants"),
                _plugin_asset("SYSTEM.md", "meta_are.system"),
            ),
            task_assets=task_assets,
            skill_assets={
                name: (
                    *(
                        (_shared_asset(shared_skills[name], f"methodology.skill.{name}"),)
                        if name in shared_skills
                        else ()
                    ),
                    *(
                        (_plugin_asset(relative_path, f"meta_are.skill.{name}"),)
                        if relative_path is not None
                        else ()
                    ),
                )
                for name, relative_path in skill_paths.items()
            },
        )
    )


def meta_are_prompt_composition_entity() -> Mapping[str, JsonValue]:
    return prompt_composition_record(
        plugin_name="meta_are",
        compositions={
            "evolve": _resolved_assets(
                "evolve",
                {"history-analysis": None},
            ),
            "diagnose_patch.capability": _resolved_assets(
                "diagnose_patch",
                {
                    "history-analysis": None,
                    "diagnose": None,
                    "capability-patch": "skills/capability-patch/SKILL.md",
                    "patch-verification": "skills/patch-verification/SKILL.md",
                },
            ),
            "diagnose_patch.steering": _resolved_assets(
                "diagnose_patch",
                {
                    "history-analysis": None,
                    "diagnose": None,
                    "steering-patch": "skills/steering-patch/SKILL.md",
                    "patch-verification": "skills/patch-verification/SKILL.md",
                },
            ),
            "reflect": _resolved_assets(
                "reflect",
                {"history-analysis": None},
            ),
        },
    )


def meta_are_curriculum_composition_entity() -> Mapping[str, JsonValue]:
    diagnose_skills = {
        phase: {
            "history-analysis": None,
            "diagnose": None,
            f"{phase}-patch": f"skills/{phase}-patch/SKILL.md",
            "patch-verification": "skills/patch-verification/SKILL.md",
        }
        for phase in ("capability", "steering")
    }
    return prompt_composition_record(
        plugin_name="meta_are",
        compositions={
            "evolve.curriculum": _resolved_assets("evolve", {"history-analysis": None}, curriculum=True),
            "diagnose_patch.capability.curriculum": _resolved_assets(
                "diagnose_patch",
                diagnose_skills["capability"],
                curriculum=True,
            ),
            "diagnose_patch.steering.curriculum": _resolved_assets(
                "diagnose_patch",
                diagnose_skills["steering"],
                curriculum=True,
            ),
            "extract_patterns": _resolved_assets(
                "extract_patterns",
                {"history-analysis": None, "symptom-extract": None, "symptom-normalize": None},
            ),
            "decide_arm": _resolved_assets("decide_arm", {"history-analysis": None}),
            "score_arms": _resolved_assets("score_arms", {"history-analysis": None, "progress-scoring": None}),
        },
    )


def _is_curriculum(context: Mapping[str, JsonValue]) -> bool:
    value = context.get("task_selection")
    return isinstance(value, Mapping) and value.get("policy") == "activesaddler"


def _task_selection(context: Mapping[str, JsonValue]) -> Mapping[str, JsonValue]:
    value = context.get("task_selection")
    if not isinstance(value, Mapping):
        raise TypeError("Meta-ARE curriculum sessions require a task_selection context object")
    return value


def _case_records(value: JsonValue | None, label: str) -> list[Mapping[str, JsonValue]]:
    if not isinstance(value, list) or any(
        not isinstance(item, Mapping) or not isinstance(item.get("case_id"), str) for item in value
    ):
        raise TypeError(f"Meta-ARE prompt context {label} must be a list of case records")
    return [item for item in value if isinstance(item, Mapping)]


def _method_name(kind: str) -> str:
    return "diagnose" if kind == "diagnose_patch" else kind


def _shared_asset(relative_path: str, asset_id: str):
    return load_prompt_asset(
        root=_PROMPTING_ROOT,
        relative_path=relative_path,
        asset_id=asset_id,
        source=f"shared/{relative_path.removeprefix('methodology/')}",
    )


def _plugin_asset(relative_path: str, asset_id: str):
    return load_prompt_asset(
        root=_ASSET_ROOT,
        relative_path=relative_path,
        asset_id=asset_id,
        source=f"plugins/meta_are/{relative_path}",
    )


def _provenance_manifest(resolved: ResolvedPromptAssets) -> str:
    return (
        canonical_json(
            {
                "schema_version": "autosaddler-prompt-assets/v1",
                "assets": [
                    {
                        "asset_id": asset.asset_id,
                        "version": asset.version,
                        "source": asset.source,
                        "sha256": asset.sha256,
                        "bytes": asset.bytes,
                    }
                    for asset in resolved.provenance
                ],
                "resolved": {
                    "system_context_sha256": sha256_digest(resolved.system_context),
                    "task_prompt_sha256": sha256_digest(resolved.task_prompt),
                    "skill_sha256": {name: sha256_digest(content) for name, content in resolved.skills.items()},
                },
            }
        )
        + "\n"
    )


def _strings(value: JsonValue | None, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or any(not isinstance(item, str) for item in value):
        raise TypeError(f"Meta-ARE prompt context {label} must be a non-empty list of strings")
    return tuple(value)


def _optional_strings(value: JsonValue | None, label: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise TypeError(f"Meta-ARE prompt context {label} must be a list of strings")
    return tuple(value)
