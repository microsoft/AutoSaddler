from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from autosaddler.v2.prompting.assets import (
    PromptAsset,
    PromptComposition,
    load_prompt_asset,
    prompt_composition_record,
    prompt_source_entities,
    resolve_prompt_composition,
)
from autosaddler.v2.plugins.meta_are.prompt_pack import meta_are_prompt_composition_entity


_V2_ROOT = Path(__file__).parents[3] / "src/autosaddler/v2"


def _asset(asset_id: str, content: str) -> PromptAsset:
    return PromptAsset(asset_id=asset_id, version="1", content=content, source=f"methodology/{asset_id}.md")


def _skill_asset(asset_id: str, name: str, description: str, body: str) -> PromptAsset:
    return _asset(
        asset_id,
        f'---\nname: {name}\ndescription: "{description}"\n---\n\n{body}',
    )


def test_resolve_prompt_composition_is_ordered_and_auditable() -> None:
    resolved = resolve_prompt_composition(
        PromptComposition(
            system_assets=(_asset("shared-system", "Shared system."), _asset("plugin-system", "Plugin system.")),
            task_assets=(
                _asset("shared-task", "Review {{HISTORY_PATH}}."),
                _asset("plugin-task", "Return one result."),
            ),
            skill_assets={
                "history-analysis": (
                    _skill_asset(
                        "shared-history",
                        "history-analysis",
                        "Analyze complete history.",
                        "Treat lessons as hypotheses.",
                    ),
                )
            },
            replacements={"HISTORY_PATH": ".autosaddler/history/manifest.json"},
        )
    )

    assert resolved.system_context == "Shared system.\n\nPlugin system.\n"
    assert resolved.task_prompt == ("Review .autosaddler/history/manifest.json.\n\nReturn one result.\n")
    assert resolved.skills["history-analysis"] == (
        "---\n"
        "name: history-analysis\n"
        'description: "Analyze complete history."\n'
        "---\n\n"
        "Treat lessons as hypotheses.\n"
    )
    assert [asset.asset_id for asset in resolved.provenance] == [
        "shared-system",
        "plugin-system",
        "shared-task",
        "plugin-task",
        "shared-history",
    ]
    assert all(asset.sha256.startswith("sha256:") for asset in resolved.provenance)
    assert [asset.bytes for asset in resolved.provenance] == [
        len(text.encode("utf-8"))
        for text in (
            "Shared system.",
            "Plugin system.",
            "Review {{HISTORY_PATH}}.",
            "Return one result.",
            '---\nname: history-analysis\ndescription: "Analyze complete history."\n---\n\n'
            "Treat lessons as hypotheses.",
        )
    ]


def test_resolve_prompt_composition_emits_one_header_for_composed_skill() -> None:
    resolved = resolve_prompt_composition(
        PromptComposition(
            system_assets=(_asset("system", "System."),),
            task_assets=(_asset("task", "Task."),),
            skill_assets={
                "patch-verification": (
                    _skill_asset("baseline", "verification-baseline", "Check the candidate.", "Core procedure."),
                    _skill_asset("plugin", "patch-verification", "Check Meta-ARE.", "Plugin procedure."),
                )
            },
        )
    )

    content = resolved.skills["patch-verification"]
    assert content.count("\n---\n") == 1
    _, frontmatter, body = content.split("---\n", 2)
    assert yaml.safe_load(frontmatter) == {
        "name": "patch-verification",
        "description": "Check the candidate. Check Meta-ARE.",
    }
    assert body == "\nCore procedure.\n\nPlugin procedure.\n"


@pytest.mark.parametrize(
    ("composition", "message"),
    [
        (
            PromptComposition(
                system_assets=(_asset("duplicate", "System."),),
                task_assets=(_asset("duplicate", "Task."),),
                skill_assets={},
            ),
            "Duplicate prompt asset IDs",
        ),
        (
            PromptComposition(
                system_assets=(_asset("system", "{{MISSING}}"),),
                task_assets=(_asset("task", "Task."),),
                skill_assets={},
            ),
            "Unknown prompt replacements",
        ),
        (
            PromptComposition(
                system_assets=(_asset("system", "System."),),
                task_assets=(_asset("task", "Task."),),
                skill_assets={},
                replacements={"UNUSED": "value"},
            ),
            "Unused prompt replacements",
        ),
        (
            PromptComposition(
                system_assets=(_asset("system", "{{VALUE}}"),),
                task_assets=(_asset("task", "Task."),),
                skill_assets={},
                replacements={"VALUE": "{{UNRESOLVED}}"},
            ),
            "Unresolved prompt replacements",
        ),
    ],
)
def test_resolve_prompt_composition_rejects_ambiguous_inputs(
    composition: PromptComposition,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        resolve_prompt_composition(composition)


def test_prompt_source_entities_snapshot_shared_and_plugin_assets(tmp_path) -> None:
    plugin_root = tmp_path / "plugin"
    plugin_root.mkdir()
    (plugin_root / "SYSTEM.md").write_text("Plugin system.\n", encoding="utf-8")

    entities = prompt_source_entities(plugin_root=plugin_root, plugin_name="example")

    assert entities["resolved/prompts/sources/plugins/example/SYSTEM.md"] == "Plugin system.\n"
    inventory = entities["resolved/prompts/assets.json"]
    assert isinstance(inventory, dict)
    assert inventory["schema_version"] == "autosaddler-prompt-source-assets/v1"
    sources = {asset["source"] for asset in inventory["assets"]}
    assert "plugins/example/SYSTEM.md" in sources
    assert "shared/system/optimizer-invariants.md" in sources


@pytest.mark.parametrize(
    ("plugin_name", "plugin_root", "composition"),
    [
        (
            "meta_are",
            Path(__file__).parents[3] / "src/autosaddler/v2/plugins/meta_are",
            meta_are_prompt_composition_entity(),
        ),
    ],
)
def test_composition_sources_join_resolved_inventory(plugin_name, plugin_root, composition) -> None:
    entities = prompt_source_entities(plugin_root=plugin_root, plugin_name=plugin_name)
    inventory = entities["resolved/prompts/assets.json"]
    assert isinstance(inventory, dict)
    inventory_sources = {asset["source"] for asset in inventory["assets"]}
    composition_sources = {
        asset["source"] for value in composition["compositions"].values() for asset in value["assets"]
    }

    assert composition_sources <= inventory_sources


def test_prompt_composition_record_pins_final_order_and_digests() -> None:
    first = resolve_prompt_composition(
        PromptComposition(
            system_assets=(_asset("shared", "Shared."), _asset("plugin", "Plugin.")),
            task_assets=(_asset("task", "Task."),),
            skill_assets={},
        )
    )
    reversed_system = resolve_prompt_composition(
        PromptComposition(
            system_assets=(_asset("plugin", "Plugin."), _asset("shared", "Shared.")),
            task_assets=(_asset("task", "Task."),),
            skill_assets={},
        )
    )

    first_record = prompt_composition_record(
        plugin_name="example",
        compositions={"evolve": first},
    )
    reversed_record = prompt_composition_record(
        plugin_name="example",
        compositions={"evolve": reversed_system},
    )

    composition = first_record["compositions"]["evolve"]
    assert [asset["asset_id"] for asset in composition["assets"]] == [
        "shared",
        "plugin",
        "task",
    ]
    assert composition["system_context"]["sha256"].startswith("sha256:")
    assert first_record != reversed_record


@pytest.mark.parametrize(
    ("root", "owner_tag"),
    [
        (_V2_ROOT / "prompting/methodology", "(Core)"),
        (_V2_ROOT / "plugins/meta_are", "(Plugin-specific)"),
    ],
)
def test_composed_prompt_asset_headings_declare_ownership(root: Path, owner_tag: str) -> None:
    markdown_paths = sorted(root.rglob("*.md"))
    assert markdown_paths
    for path in markdown_paths:
        headings = [line for line in path.read_text(encoding="utf-8").splitlines() if line.startswith("#")]
        assert headings, path
        assert all(heading.endswith(owner_tag) for heading in headings), path


def test_all_skill_sources_have_strict_discovery_frontmatter() -> None:
    skill_paths = sorted((_V2_ROOT.parent / "proposer/autosaddler/skills").rglob("SKILL.md"))
    skill_paths.extend(sorted(_V2_ROOT.rglob("skills/*/SKILL.md")))
    assert skill_paths
    for path in skill_paths:
        content = path.read_text(encoding="utf-8")
        assert content.startswith("---\n"), path
        _, frontmatter, body = content.split("---\n", 2)
        metadata = yaml.safe_load(frontmatter)
        assert set(metadata) == {"name", "description"}, path
        assert metadata["name"] == path.parent.name, path
        assert isinstance(metadata["description"], str) and metadata["description"].strip(), path
        assert len(metadata["description"]) <= 1024, path
        assert body.strip(), path


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("# Missing frontmatter\n", "must start with YAML frontmatter"),
        (
            "---\nname: example\ndescription: Example.\nuser-invocable: false\n---\n\n# Example\n",
            "must contain exactly name and description",
        ),
        (
            "---\nname: wrong-name\ndescription: Example.\n---\n\n# Example\n",
            "does not match folder",
        ),
    ],
)
def test_load_prompt_asset_rejects_invalid_skill_frontmatter(
    tmp_path: Path,
    content: str,
    message: str,
) -> None:
    skill_path = tmp_path / "skills/example/SKILL.md"
    skill_path.parent.mkdir(parents=True)
    skill_path.write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        load_prompt_asset(
            root=tmp_path,
            relative_path="skills/example/SKILL.md",
            asset_id="example.skill",
        )


@pytest.mark.parametrize(
    "composition",
    [meta_are_prompt_composition_entity()],
)
def test_generic_skills_are_core_only(composition) -> None:
    compositions = composition["compositions"]
    for name, value in compositions.items():
        asset_ids = [asset["asset_id"] for asset in value["assets"]]
        assert "methodology.skill.history-analysis" in asset_ids
        assert [asset_id for asset_id in asset_ids if asset_id.endswith(".skill.history-analysis")] == [
            "methodology.skill.history-analysis"
        ]
        if name.startswith("diagnose_patch"):
            assert "methodology.skill.diagnose" in asset_ids
            assert [asset_id for asset_id in asset_ids if asset_id.endswith(".skill.diagnose")] == [
                "methodology.skill.diagnose"
            ]


@pytest.mark.parametrize(
    "composition",
    [meta_are_prompt_composition_entity()],
)
def test_actual_compositions_order_core_before_plugin(composition) -> None:
    for value in composition["compositions"].values():
        asset_ids = [asset["asset_id"] for asset in value["assets"]]
        system_index = asset_ids.index("methodology.system.invariants")
        plugin_system_index = next(index for index, asset_id in enumerate(asset_ids) if asset_id.endswith(".system"))
        assert system_index < plugin_system_index

        core_task_index = next(
            index for index, asset_id in enumerate(asset_ids) if asset_id.startswith("methodology.prompt.")
        )
        plugin_task_index = next(
            index
            for index, asset_id in enumerate(asset_ids)
            if ".prompt." in asset_id and not asset_id.startswith("methodology.prompt.")
        )
        assert core_task_index < plugin_task_index
