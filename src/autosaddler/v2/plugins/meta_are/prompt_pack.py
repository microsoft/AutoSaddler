from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath

from autosaddler.v2.core.domain import JsonValue, canonical_json, sha256_digest
from autosaddler.v2.prompting.assets import (
    PromptComposition,
    ResolvedPromptAssets,
    load_prompt_asset,
    prompt_composition_record,
    resolve_prompt_composition,
)
from autosaddler.v2.prompting.history import build_history_bundle
from autosaddler.v2.prompting.models import SessionSpec
from autosaddler.v2.storage.local import LocalRunStore

_ASSET_ROOT = Path(__file__).parent
_PROMPTING_ROOT = Path(__file__).parents[2] / "prompting"
_SESSION_CONTEXT_PATH = ".autosaddler/session_context.json"
_TRAINING_EVIDENCE_PATH = ".autosaddler/training_evidence.json"
_PROMPT_ASSETS_PATH = ".autosaddler/prompt_assets.json"

class MetaAREPromptPack:
    def __init__(
        self,
        *,
        store: LocalRunStore,
        writable_paths: Sequence[PurePosixPath],
        capability_phase_iterations: int,
    ) -> None:
        if not writable_paths:
            raise ValueError("Meta-ARE prompt pack requires writable paths")
        if capability_phase_iterations < 0:
            raise ValueError("Meta-ARE capability phase iterations cannot be negative")
        self.store = store
        self.writable_paths = tuple(writable_paths)
        self.capability_phase_iterations = capability_phase_iterations

    def session(self, kind: str, context: Mapping[str, JsonValue]) -> SessionSpec:
        iteration = context.get("iteration")
        if isinstance(iteration, bool) or not isinstance(iteration, int) or iteration < 0:
            raise ValueError("Meta-ARE prompt context requires a nonnegative iteration")
        patch_phase = "capability" if iteration + 1 <= self.capability_phase_iterations else "steering"
        history = build_history_bundle(self.store, context)
        objective_mode = "pareto" if history.compatibility_history.get("objective_mode") == "pareto" else "accuracy"
        rendered_context: dict[str, JsonValue] = {
            **context,
            "objective_mode": objective_mode,
            "patch_phase": patch_phase,
            "mutation_scope": [path.as_posix() for path in self.writable_paths],
        }
        workspace_files = {
            _SESSION_CONTEXT_PATH: canonical_json(rendered_context) + "\n",
        }
        workspace_files.update(history.workspace_files)

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
            if objective_mode == "pareto":
                base_id = context.get("fixed_current_candidate_id")
                if candidate_ids != (base_id,):
                    raise ValueError("Pareto preparation must receive only the recorded current as its base")
                donors = _optional_strings(context.get("donor_candidate_ids"), "donor_candidate_ids")
                schema = _pareto_evolve_schema(candidate_ids[0], donors)
                skill_paths = {
                    "history-analysis": None,
                    f"{patch_phase}-patch": f"skills/{patch_phase}-patch/SKILL.md",
                    "patch-verification": "skills/patch-verification/SKILL.md",
                }
                mutation_label = patch_phase
            else:
                schema = _evolve_schema(candidate_ids, component_source_options)
                skill_paths = {"history-analysis": None}
                mutation_label = None
        elif kind == "reflect":
            schema = _reflection_schema()
            skill_paths = {"history-analysis": None}
            mutation_label = None
        else:
            raise ValueError(f"Unknown Meta-ARE session kind: {kind}")

        resolved = _resolved_assets(kind, skill_paths, objective_mode=objective_mode)
        workspace_files[_PROMPT_ASSETS_PATH] = _provenance_manifest(resolved, objective_mode=objective_mode)

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


def _pareto_evolve_schema(base_id: str, donors: tuple[str, ...]) -> Mapping[str, JsonValue]:
    schema = dict(_evolve_schema((base_id,), {}))
    properties = dict(schema["properties"])
    properties["parent_ids"] = {**properties["parent_ids"], "maxItems": 1}
    properties["changed_paths"] = {
        "type": "array", "items": {"type": "string", "minLength": 1}, "uniqueItems": True,
    }
    properties["donor_candidate_ids"] = {
        "type": "array", "uniqueItems": True,
        **({"items": {"type": "string", "enum": list(donors)}} if donors else {"maxItems": 0}),
    }
    schema["properties"] = properties
    schema["required"] = [*schema["required"], "changed_paths", "donor_candidate_ids"]
    return schema


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
    kind: str, skill_paths: Mapping[str, str | None], *, objective_mode: str = "accuracy",
) -> ResolvedPromptAssets:
    methodology_root = "methodology/pareto" if objective_mode == "pareto" else "methodology"
    methodology_id = "methodology.pareto" if objective_mode == "pareto" else "methodology"
    shared_skills = {
        "history-analysis": f"{methodology_root}/skills/history-analysis/SKILL.md",
        "diagnose": f"{methodology_root}/skills/causal-diagnosis/SKILL.md",
        "patch-verification": "methodology/skills/verification-baseline/SKILL.md",
    }
    return resolve_prompt_composition(
        PromptComposition(
            system_assets=(
                _shared_asset(f"{methodology_root}/system/optimizer-invariants.md", f"{methodology_id}.system.invariants"),
                _plugin_asset("SYSTEM.md", "meta_are.system"),
            ),
            task_assets=(
                _shared_asset(f"{methodology_root}/prompts/{_method_name(kind)}-method.md", f"{methodology_id}.prompt.{kind}"),
                *(
                    () if objective_mode == "pareto" and kind == "evolve"
                    else (_plugin_asset(f"prompts/{kind}.md", f"meta_are.prompt.{kind}"),)
                ),
                *(
                    (_plugin_asset(f"pareto/prompts/{kind}.md", f"meta_are.pareto.prompt.{kind}"),)
                    if objective_mode == "pareto" else ()
                ),
            ),
            skill_assets={
                name: (
                    *(
                        (_shared_asset(
                            shared_skills[name],
                            f"{methodology_id if name != 'patch-verification' else 'methodology'}.skill.{name}",
                        ),)
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
    skills = {
        "evolve": {"history-analysis": None},
        "diagnose_patch.capability": {
            "history-analysis": None,
            "diagnose": None,
            "capability-patch": "skills/capability-patch/SKILL.md",
            "patch-verification": "skills/patch-verification/SKILL.md",
        },
        "diagnose_patch.steering": {
            "history-analysis": None,
            "diagnose": None,
            "steering-patch": "skills/steering-patch/SKILL.md",
            "patch-verification": "skills/patch-verification/SKILL.md",
        },
        "reflect": {"history-analysis": None},
    }
    return prompt_composition_record(
        plugin_name="meta_are",
        compositions={
            (name if mode == "accuracy" else f"pareto.{name}"): _resolved_assets(
                name.split(".")[0], skill_paths, objective_mode=mode,
            )
            for mode, variants in (
                ("accuracy", skills),
                ("pareto", {
                    **{name: paths for name, paths in skills.items() if name != "evolve"},
                    **{f"evolve.{phase}": {
                        "history-analysis": None,
                        f"{phase}-patch": f"skills/{phase}-patch/SKILL.md",
                        "patch-verification": "skills/patch-verification/SKILL.md",
                    } for phase in ("capability", "steering")},
                }),
            )
            for name, skill_paths in variants.items()
        },
    )


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


def _provenance_manifest(resolved: ResolvedPromptAssets, *, objective_mode: str) -> str:
    return (
        canonical_json(
            {
                "schema_version": "autosaddler-prompt-assets/v1",
                "objective_mode": objective_mode,
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
