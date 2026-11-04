from __future__ import annotations

import json
import shutil
from pathlib import Path, PurePosixPath

import pytest

from autosaddler.v2.core.domain import ArtifactRef, Candidate, Cost, Evaluation, Observation, sha256_digest
from autosaddler.v2.core.serde import record
from autosaddler.v2.storage.local import LocalRunStore


def test_prompt_pack_renders_phase_and_closed_diagnosis_contract(tmp_path: Path) -> None:
    from autosaddler.v2.plugins.meta_are.prompt_pack import MetaAREPromptPack

    store = _store(tmp_path)
    evidence = store.write_json(
        "evidence/train/evidence.json",
        {"schema_version": "autosaddler-meta-are-evidence/v1", "case_records": []},
        kind="meta-are-training-evidence",
    )
    pack = MetaAREPromptPack(
        store=store,
        writable_paths=(
            PurePosixPath("are/simulation/agents/default_agent"),
            PurePosixPath("hook.json"),
        ),
        capability_phase_iterations=1,
    )
    candidate_id = sha256_digest("candidate")

    specification = pack.session(
        "diagnose_patch",
        {
            "iteration": 0,
            "candidate_ids": [candidate_id],
            "train_case_ids": ["train-a"],
            "evidence": {
                "uri": evidence.uri,
                "kind": evidence.kind,
                "sha256": evidence.sha256,
                "bytes": evidence.bytes,
            },
        },
    )

    assert specification.output_schema["additionalProperties"] is False
    assert specification.output_schema["required"] == [
        "schema_version",
        "intent",
        "diagnosis",
        "expected_effect",
        "changed_paths",
    ]
    context = json.loads(specification.workspace_files[".autosaddler/session_context.json"])
    assert context["patch_phase"] == "capability"
    assert context["mutation_scope"] == [
        "are/simulation/agents/default_agent",
        "hook.json",
    ]
    assert ".autosaddler/training_evidence.json" in specification.workspace_files
    assert ".autosaddler/history/manifest.json" in specification.workspace_files
    history_manifest = json.loads(specification.workspace_files[".autosaddler/history/manifest.json"])
    assert history_manifest["schema_version"] == "autosaddler-history-manifest/v1"
    assert history_manifest["through_sequence"] == 0
    prompt_manifest = json.loads(specification.workspace_files[".autosaddler/prompt_assets.json"])
    assert prompt_manifest["schema_version"] == "autosaddler-prompt-assets/v1"
    assert {asset["asset_id"] for asset in prompt_manifest["assets"]} >= {
        "methodology.system.invariants",
        "methodology.prompt.diagnose_patch",
        "meta_are.system",
        "meta_are.prompt.diagnose_patch",
    }
    assert "GAIA2" in specification.system_context
    assert "earliest divergence" in specification.task_prompt
    assert "edit" in specification.task_prompt.lower()
    assert "current workspace" in specification.task_prompt.lower()
    assert set(specification.skills) == {
        "history-analysis",
        "diagnose",
        "capability-patch",
        "patch-verification",
    }


def test_prompt_pack_keeps_evolve_and_reflection_provider_neutral(tmp_path: Path) -> None:
    from autosaddler.v2.plugins.meta_are.prompt_pack import MetaAREPromptPack

    store = _store(tmp_path)
    parents = [sha256_digest("parent-a"), sha256_digest("parent-b")]
    pack = MetaAREPromptPack(
        store=store,
        writable_paths=(PurePosixPath("are/simulation/agents/default_agent"),),
        capability_phase_iterations=1,
    )

    evolve = pack.session(
        "evolve",
        {
            "iteration": 1,
            "candidate_ids": parents,
            "component_source_options": {"hook.json": [parents[1]]},
            "train_case_ids": ["train-a"],
        },
    )
    assert evolve.output_schema["additionalProperties"] is False
    assert evolve.output_schema["properties"]["parent_ids"]["items"]["enum"] == parents
    component_sources = evolve.output_schema["properties"]["component_sources"]
    assert component_sources["properties"] == {
        "hook.json": {"type": "string", "enum": [parents[1]]},
    }
    assert component_sources["additionalProperties"] is False
    assert "development case" not in evolve.task_prompt.lower()
    assert "Default to one measured base parent" in evolve.task_prompt
    assert "pareto" not in "\n".join((evolve.system_context, evolve.task_prompt, *evolve.skills.values())).lower()

    reflect = pack.session(
        "reflect",
        {
            "iteration": 1,
            "candidate_ids": [parents[0]],
            "train_case_ids": ["train-a"],
            "development_outcome": {"aggregate_score": 0.5},
        },
    )
    assert reflect.output_schema["additionalProperties"] is False
    assert reflect.output_schema["required"] == ["schema_version", "lessons"]
    assert ".autosaddler/history/current_batch.json" in evolve.workspace_files
    assert ".autosaddler/history/current_batch.json" in reflect.workspace_files

    rendered = "\n".join(
        [
            evolve.task_prompt,
            reflect.task_prompt,
            *evolve.workspace_files.values(),
            *reflect.workspace_files.values(),
        ]
    )
    assert "CLAUDE.md" not in rendered
    assert "gc_sdk" not in rendered
    assert "direct EvoDAG" not in rendered


@pytest.mark.parametrize(("relative_path", "sections"), [
    ("system/optimizer-invariants.md", [
        "Evidence And Generalization", "History And Durable Learning", "Optimization Invariants",
    ]),
    ("prompts/diagnose-method.md", ["Verify And Report"]),
    ("prompts/reflect-method.md", ["Record Durable Lessons"]),
    ("skills/causal-diagnosis/SKILL.md", ["Intervention Test", "Quality Checklist"]),
    ("skills/history-analysis/SKILL.md", ["Classify Prior Changes"]),
])
def test_pareto_preserves_objective_independent_sections(relative_path, sections):
    from autosaddler.v2.plugins.meta_are.prompt_pack import _PROMPTING_ROOT

    methodology = _PROMPTING_ROOT / "methodology"
    accuracy = (methodology / relative_path).read_text(encoding="utf-8")
    pareto = (methodology / "pareto" / relative_path).read_text(encoding="utf-8")
    for section in sections:
        heading = f"## {section} (Core)\n"
        assert heading in accuracy and heading in pareto
        original = accuracy.split(heading, 1)[1].split("\n## ", 1)[0].strip()
        adapted = pareto.split(heading, 1)[1].split("\n## ", 1)[0].strip()
        assert adapted.startswith(original), (relative_path, section)


@pytest.mark.parametrize("kind", ["diagnose_patch", "reflect"])
def test_pareto_uses_exclusive_objective_methodology(kind: str) -> None:
    from autosaddler.v2.plugins.meta_are.prompt_pack import _resolved_assets

    skills = {"history-analysis": None}
    if kind == "diagnose_patch":
        skills["diagnose"] = None
    resolved = _resolved_assets(kind, skills, objective_mode="pareto")
    rendered = "\n".join((resolved.system_context, resolved.task_prompt, *resolved.skills.values()))
    assert "Select or compose a parent" not in rendered
    assert "For every target failure:" not in rendered
    assert "If failures exist" not in rendered
    assert "These instructions replace" not in rendered
    assert any(asset.source.startswith("shared/pareto/") for asset in resolved.provenance)
    assert not any(asset.source.startswith("shared/system/") for asset in resolved.provenance)


@pytest.mark.parametrize("kind", ["diagnose_patch", "reflect"])
def test_pareto_prompts_treat_accuracy_and_cost_as_separate_objectives(kind: str) -> None:
    from autosaddler.v2.plugins.meta_are.prompt_pack import _resolved_assets

    skills = {"history-analysis": None}
    if kind == "diagnose_patch":
        skills["diagnose"] = None
    resolved = _resolved_assets(kind, skills, objective_mode="pareto")
    system = " ".join(resolved.system_context.split())
    assert "improves reliable task performance while preserving behavior" not in system
    assert "non-scalarized accuracy-cost objective: maximize accuracy and minimize mean task-agent USD cost" in system
    assert "Joint improvements, accuracy improvements, cost improvements" in system
    if kind == "diagnose_patch":
        task = " ".join(resolved.task_prompt.split())
        assert "Preserve required actions and accuracy" not in task
        assert "configured Bayesian bootstrap screen" in task
        assert "persisted joint support" in task
        assert "expected effects on both accuracy and cost" in resolved.skills["diagnose"]
    else:
        assert "training score/cost measurements" in resolved.task_prompt
        assert "cost reduction or increase" in resolved.task_prompt
        assert "accuracy/cost trade-off" in resolved.task_prompt
    assert "accuracy/cost trade-offs" in resolved.skills["history-analysis"]


@pytest.mark.parametrize("kind", ["diagnose_patch", "reflect"])
def test_pareto_prompts_preserve_reference_objective_reasoning(kind: str) -> None:
    from autosaddler.v2.plugins.meta_are.prompt_pack import _resolved_assets

    skills = {"history-analysis": None}
    if kind == "diagnose_patch":
        skills["diagnose"] = None
    resolved = _resolved_assets(kind, skills, objective_mode="pareto")
    system = " ".join(resolved.system_context.split())
    assert "grow a useful accuracy-cost Pareto frontier across the full task distribution" in system
    assert "either direction of accuracy-cost trade-off" in system
    history = " ".join(resolved.skills["history-analysis"].split())
    assert "Added cost is not automatically a regression" in history
    if kind == "diagnose_patch":
        diagnosis = " ".join(resolved.skills["diagnose"].split())
        assert "cheap versus expensive and high- versus low-score runs" in diagnosis
        assert "execution shape" in diagnosis
        assert "context shape" in diagnosis
        assert "what added work bought" in diagnosis
    else:
        task = " ".join(resolved.task_prompt.split())
        assert "changed path actually ran" in task
        assert "patch-caused, stochastic, mixed, or uncertain" in task
        assert "seed, immediate parent, owning iteration's current" in task
    rendered = "\n".join((resolved.system_context, resolved.task_prompt, *resolved.skills.values()))
    assert "seed-relative accuracy floor" not in rendered
    assert "evo-dag" not in rendered


@pytest.mark.parametrize("repetitions", [1, 2, 3, 5])
@pytest.mark.parametrize("iteration", [0, 1, 26])
def test_pareto_prompts_preserve_phase_and_schema_with_mode_provenance(
    tmp_path: Path, repetitions: int, iteration: int,
) -> None:
    from autosaddler.v2.plugins.meta_are.prompt_pack import MetaAREPromptPack, meta_are_prompt_composition_entity

    store = _store(tmp_path)
    pack = MetaAREPromptPack(
        store=store, writable_paths=(PurePosixPath("hook.json"),), capability_phase_iterations=1,
    )
    evidence = store.write_json("evidence/train/evidence.json", {
        "schema_version": "autosaddler-meta-are-evidence/v1", "requested_repetitions": repetitions,
        "case_records": [{"case_id": "train-a", "observed_repetitions": repetitions, "per_repetition": [
            {"repetition": repetition, "score": 1.0, "cost": {"currency_amount": 0.02}}
            for repetition in range(repetitions)
        ]}],
    }, kind="meta-are-training-evidence")
    context = {"iteration": iteration, "train_case_ids": ["train-a"], "evidence": record(evidence)}
    accuracy = {kind: pack.session(kind, context) for kind in ("diagnose_patch", "reflect")}
    for spec in accuracy.values():
        rendered = "\n".join((spec.system_context, spec.task_prompt, *spec.skills.values()))
        assert "pareto" not in rendered.lower()
    seed = Candidate(
        candidate_id=sha256_digest("seed"), parent_ids=(), space="git",
        artifact=ArtifactRef(uri="candidate.json", kind="git-candidate"),
    )
    development = Evaluation(
        evaluation_id=sha256_digest("dev"), candidate_id=seed.candidate_id, split="development",
        purpose="development", iteration=0, requested_case_ids=("DEV-SECRET",),
        observations=tuple(Observation.create(
            candidate_id=seed.candidate_id, case_id="DEV-SECRET", split="development",
            repetition=repetition, disposition="success", score=1.0, cost=Cost(currency_amount=0.02),
            evaluator_fingerprint="test", metadata={"answer": "HIDDEN-ANSWER"},
        ) for repetition in range(repetitions)),
        artifact_dir=ArtifactRef(uri="quarantine/development", kind="evaluation-directory"),
    )
    store.append("RunStarted", "start", {"seed_candidate": record(seed)})
    store.append("EvaluationCompleted", "dev", {"evaluation": record(development)})
    store.append("ParetoArchiveUpdated", "archive", {
        "schema_version": "autosaddler-pareto-decision/v1", "iteration": 0, "repetitions": repetitions,
        "evaluation_ids": [development.evaluation_id], "archive_candidate_ids": [seed.candidate_id],
        "current_candidate_id": seed.candidate_id,
    })
    compositions = meta_are_prompt_composition_entity()["compositions"]
    assert "pareto.evolve" not in compositions
    for kind, original in accuracy.items():
        spec = pack.session(kind, context)
        session = json.loads(spec.workspace_files[".autosaddler/session_context.json"])
        phase = "capability" if iteration == 0 else "steering"
        assert session["objective_mode"] == "pareto"
        assert session["patch_phase"] == phase
        assert spec.mutation_label == original.mutation_label
        assert spec.capabilities == original.capabilities
        assert spec.output_schema == original.output_schema
        assert spec.skills.keys() == original.skills.keys()
        assert not spec.system_context.startswith(original.system_context)
        assert not spec.task_prompt.startswith(original.task_prompt)
        for name, content in original.skills.items():
            if name in {"history-analysis", "diagnose"}:
                assert content.split("\n---\n", 1)[1] not in spec.skills[name]
            else:
                assert spec.skills[name] == content
        assert "non-scalarized accuracy-cost objective" in spec.system_context
        assert "fixed current" in spec.system_context
        assert "{{" not in spec.system_context
        manifest = json.loads(spec.workspace_files[".autosaddler/prompt_assets.json"])
        assert manifest["objective_mode"] == "pareto"
        original_manifest = json.loads(original.workspace_files[".autosaddler/prompt_assets.json"])
        original_assets = {asset["asset_id"]: asset for asset in original_manifest["assets"]}
        composed_assets = {asset["asset_id"]: asset for asset in manifest["assets"]}
        methodology_names = {"system.invariants", f"prompt.{kind}", "skill.history-analysis"}
        if kind == "diagnose_patch":
            methodology_names.add("skill.diagnose")
        replaced_names = {f"methodology.{name}" for name in methodology_names}
        assert original_assets.keys() - composed_assets.keys() == replaced_names
        assert {name: composed_assets[name] for name in original_assets.keys() - replaced_names} == {
            name: asset for name, asset in original_assets.items() if name not in replaced_names
        }
        assert not any(".pareto." in name for name in original_assets)
        assert composed_assets.keys() - original_assets.keys() == {
            *(f"methodology.pareto.{name}" for name in methodology_names),
            f"meta_are.pareto.prompt.{kind}",
        }
        key = f"pareto.diagnose_patch.{phase}" if kind == "diagnose_patch" else "pareto.reflect"
        assert compositions[key]["system_context"]["sha256"] == sha256_digest(spec.system_context)
        assert compositions[key]["task_prompt"]["sha256"] == sha256_digest(spec.task_prompt)
        assert all(compositions[key]["skills"][name]["sha256"] == sha256_digest(text)
                   for name, text in spec.skills.items())
        rendered = "\n".join(spec.workspace_files.values())
        assert "DEV-SECRET" not in rendered and "HIDDEN-ANSWER" not in rendered
        if kind == "diagnose_patch":
            assert "all-success batch" in spec.task_prompt
            assert "usage_incomplete" in spec.task_prompt
            assert "cost.currency_amount" in spec.task_prompt
            assert "cost mechanism" in spec.skills["diagnose"] or "avoidable measured cost" in spec.skills["diagnose"]
            staged = json.loads(spec.workspace_files[".autosaddler/training_evidence.json"])
            assert staged["requested_repetitions"] == repetitions
            assert spec.skills[f"{phase}-patch"] == original.skills[f"{phase}-patch"]
        else:
            assert "joint support" in spec.task_prompt
            assert "owning iteration" in spec.task_prompt
            assert "development-dominated" in spec.task_prompt


@pytest.mark.parametrize(("root_name", "relative_path", "kind", "component"), [
    ("_PROMPTING_ROOT", "methodology/pareto/system/optimizer-invariants.md", "", "system_context"),
    ("_PROMPTING_ROOT", "methodology/pareto/prompts/diagnose-method.md", "diagnose_patch", "task_prompt"),
    ("_PROMPTING_ROOT", "methodology/pareto/prompts/reflect-method.md", "reflect", "task_prompt"),
    ("_PROMPTING_ROOT", "methodology/pareto/skills/causal-diagnosis/SKILL.md", "diagnose_patch", "skills"),
    ("_PROMPTING_ROOT", "methodology/pareto/skills/history-analysis/SKILL.md", "", "skills"),
    ("_ASSET_ROOT", "pareto/prompts/diagnose_patch.md", "diagnose_patch", "task_prompt"),
    ("_ASSET_ROOT", "pareto/prompts/reflect.md", "reflect", "task_prompt"),
])
def test_pareto_resources_have_separate_composition_fingerprints(
    tmp_path, monkeypatch, root_name, relative_path, kind, component,
):
    from autosaddler.v2.plugins.meta_are import prompt_pack

    root = tmp_path / "resources"
    shutil.copytree(getattr(prompt_pack, root_name), root)
    monkeypatch.setattr(prompt_pack, root_name, root)
    before = prompt_pack.meta_are_prompt_composition_entity()["compositions"]
    path = root / relative_path
    path.write_text(path.read_text(encoding="utf-8") + "\nAdditional Pareto guidance.\n", encoding="utf-8")
    after = prompt_pack.meta_are_prompt_composition_entity()["compositions"]
    for name, composition in before.items():
        if name.startswith(f"pareto.{kind}"):
            assert after[name][component] != composition[component]
            for unchanged in {"system_context", "task_prompt", "skills"} - {component}:
                assert after[name][unchanged] == composition[unchanged]
        else:
            assert after[name] == composition


def _store(tmp_path: Path) -> LocalRunStore:
    store = LocalRunStore(run_dir=tmp_path / "run", run_id="run")
    store.initialize(
        resolved_config={"schema_version": "autosaddler/v2"},
        resolved_entities={"resolved/component_graph.json": {"scenario": "meta_are"}},
    )
    return store


@pytest.mark.parametrize("capability_iterations,phase", [(2, "capability"), (1, "steering")])
def test_pareto_evolve_prepares_fixed_base_with_phase_specific_provenance(tmp_path, capability_iterations, phase):
    from autosaddler.v2.core.domain import ArtifactRef, Candidate, Cost, Evaluation, Observation
    from autosaddler.v2.core.serde import record
    from autosaddler.v2.plugins.meta_are.prompt_pack import MetaAREPromptPack, meta_are_prompt_composition_entity
    from autosaddler.v2.prompting.models import session_output_validation_error

    store = _store(tmp_path)
    seed = Candidate(sha256_digest("seed"), (), "git", ArtifactRef(uri="candidate.json", kind="candidate"))
    observation = Observation.create(candidate_id=seed.candidate_id, case_id="DEV-SECRET", split="development",
                                     repetition=0, disposition="success", score=1.0, cost=Cost(currency_amount=1),
                                     evaluator_fingerprint="test", metadata={"answer": "HIDDEN-ANSWER"})
    evaluation = Evaluation(sha256_digest("dev"), seed.candidate_id, "development", "development", 0,
                            ("DEV-SECRET",), (observation,), ArtifactRef(uri="dev", kind="evaluation"))
    store.append("RunStarted", "start", {"seed_candidate": record(seed)})
    store.append("EvaluationCompleted", "dev", {"evaluation": record(evaluation)})
    store.append("ParetoArchiveUpdated", "archive", {
        "iteration": 0, "repetitions": 1, "evaluation_ids": [evaluation.evaluation_id],
        "archive_candidate_ids": [seed.candidate_id], "current_candidate_id": seed.candidate_id,
    })
    pack = MetaAREPromptPack(store=store, writable_paths=(PurePosixPath("hook.json"),),
                            capability_phase_iterations=capability_iterations)
    donor = sha256_digest("donor")
    context = {"iteration": 1, "candidate_ids": [seed.candidate_id], "fixed_current_candidate_id": seed.candidate_id,
               "donor_candidate_ids": [donor], "train_case_ids": ["train-a"]}
    spec = pack.session("evolve", context)
    assert spec.mutation_label == phase
    assert f"{phase}-patch" in spec.skills and "patch-verification" in spec.skills
    assert "Whole-file replacement is the only supported" not in spec.task_prompt
    assert "prepare" in spec.task_prompt.lower() or "preparation" in spec.task_prompt.lower()
    output = {"schema_version": "autosaddler-meta-are-evolution/v1", "parent_ids": [seed.candidate_id],
              "component_sources": {}, "changed_paths": [], "donor_candidate_ids": [], "rationale": "Keep base."}

    def validate(value):
        return session_output_validation_error(spec.output_schema, value)

    assert validate(output) is None
    assert validate({**output, "parent_ids": [donor]}) is not None
    assert validate({**output, "parent_ids": [seed.candidate_id, donor]}) is not None
    assert validate({**output, "component_sources": {"hook.json": donor}}) is not None
    assert validate({**output, "donor_candidate_ids": [sha256_digest("ineligible")]}) is not None
    assert validate({**output, "donor_candidate_ids": [donor]}) is None
    rendered = "\n".join(spec.workspace_files.values())
    assert "DEV-SECRET" not in rendered and "HIDDEN-ANSWER" not in rendered
    composition = meta_are_prompt_composition_entity()["compositions"][f"pareto.evolve.{phase}"]
    assert composition["task_prompt"]["sha256"] == sha256_digest(spec.task_prompt)
    assert all(composition["skills"][name]["sha256"] == sha256_digest(value) for name, value in spec.skills.items())
