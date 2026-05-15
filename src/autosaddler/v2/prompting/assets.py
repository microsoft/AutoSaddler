from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from types import MappingProxyType

import yaml

from autosaddler.v2.core.domain import JsonValue, sha256_digest

_REPLACEMENT = re.compile(r"\{\{([A-Z][A-Z0-9_]*)\}\}")
_SKILL_NAME = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_SKILL_FRONTMATTER_KEYS = frozenset({"name", "description"})


@dataclass(frozen=True, slots=True)
class PromptAsset:
    asset_id: str
    version: str
    content: str
    source: str

    def __post_init__(self) -> None:
        for label, value in (
            ("asset_id", self.asset_id),
            ("version", self.version),
            ("content", self.content),
            ("source", self.source),
        ):
            if not value:
                raise ValueError(f"PromptAsset.{label} must be non-empty")


@dataclass(frozen=True, slots=True)
class PromptComposition:
    system_assets: tuple[PromptAsset, ...]
    task_assets: tuple[PromptAsset, ...]
    skill_assets: Mapping[str, tuple[PromptAsset, ...]]
    replacements: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.system_assets or not self.task_assets:
            raise ValueError("Prompt composition requires system and task assets")
        if any(not name for name in self.skill_assets):
            raise ValueError("Prompt composition skill names must be non-empty")
        if any(not assets for assets in self.skill_assets.values()):
            raise ValueError("Prompt composition skills must contain at least one asset")
        invalid_replacements = sorted(
            key for key in self.replacements if _REPLACEMENT.fullmatch(f"{{{{{key}}}}}") is None
        )
        if invalid_replacements:
            raise ValueError(f"Invalid prompt replacement names: {invalid_replacements}")
        object.__setattr__(self, "skill_assets", MappingProxyType(dict(self.skill_assets)))
        object.__setattr__(self, "replacements", MappingProxyType(dict(self.replacements)))


@dataclass(frozen=True, slots=True)
class ResolvedPromptAsset:
    asset_id: str
    version: str
    source: str
    sha256: str
    bytes: int


@dataclass(frozen=True, slots=True)
class ResolvedPromptAssets:
    system_context: str
    task_prompt: str
    skills: Mapping[str, str]
    provenance: tuple[ResolvedPromptAsset, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "skills", MappingProxyType(dict(self.skills)))


@dataclass(frozen=True, slots=True)
class _SkillDocument:
    name: str
    description: str
    body: str


def load_prompt_asset(
    *,
    root: Path,
    relative_path: str,
    asset_id: str,
    source: str | None = None,
    version: str = "1",
) -> PromptAsset:
    relative = PurePosixPath(relative_path)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Unsafe prompt asset path: {relative_path}")
    path = root.joinpath(*relative.parts)
    if not path.is_file():
        raise FileNotFoundError(f"Missing prompt asset: {relative_path}")
    content = path.read_text(encoding="utf-8")
    if relative.name == "SKILL.md":
        validate_skill_document(content, expected_name=relative.parent.name)
    return PromptAsset(
        asset_id=asset_id,
        version=version,
        content=content,
        source=source or relative.as_posix(),
    )


def prompt_source_entities(
    *,
    plugin_root: Path,
    plugin_name: str,
) -> dict[str, str | Mapping[str, JsonValue]]:
    if not plugin_name or PurePosixPath(plugin_name).name != plugin_name:
        raise ValueError(f"Unsafe prompt plugin name: {plugin_name}")
    methodology_root = Path(__file__).parent / "methodology"
    sources = {
        **{
            f"shared/{path.relative_to(methodology_root).as_posix()}": path
            for path in sorted(methodology_root.rglob("*.md"))
        },
        **{
            f"plugins/{plugin_name}/{path.relative_to(plugin_root).as_posix()}": path
            for path in sorted(plugin_root.rglob("*.md"))
        },
    }
    if not sources:
        raise ValueError(f"No prompt source assets found for {plugin_name}")
    entities: dict[str, str | Mapping[str, JsonValue]] = {}
    inventory: list[dict[str, JsonValue]] = []
    for source, path in sorted(sources.items()):
        text = path.read_text(encoding="utf-8")
        resolved_path = f"resolved/prompts/sources/{source}"
        entities[resolved_path] = text
        inventory.append(
            {
                "source": source,
                "resolved_path": resolved_path,
                "sha256": sha256_digest(text),
                "bytes": len(text.encode("utf-8")),
            }
        )
    entities["resolved/prompts/assets.json"] = {
        "schema_version": "autosaddler-prompt-source-assets/v1",
        "plugin": plugin_name,
        "assets": inventory,
    }
    return entities


def prompt_composition_record(
    *,
    plugin_name: str,
    compositions: Mapping[str, ResolvedPromptAssets],
) -> Mapping[str, JsonValue]:
    if not plugin_name or PurePosixPath(plugin_name).name != plugin_name:
        raise ValueError(f"Unsafe prompt plugin name: {plugin_name}")
    if not compositions:
        raise ValueError("Prompt composition inventory must not be empty")
    return {
        "schema_version": "autosaddler-prompt-compositions/v1",
        "plugin": plugin_name,
        "compositions": {
            name: {
                "system_context": _content_record(resolved.system_context),
                "task_prompt": _content_record(resolved.task_prompt),
                "skills": {
                    skill_name: _content_record(content) for skill_name, content in sorted(resolved.skills.items())
                },
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
            }
            for name, resolved in sorted(compositions.items())
        },
    }


def _content_record(content: str) -> Mapping[str, JsonValue]:
    return {
        "sha256": sha256_digest(content),
        "bytes": len(content.encode("utf-8")),
    }


def resolve_prompt_composition(composition: PromptComposition) -> ResolvedPromptAssets:
    ordered_assets = (
        *composition.system_assets,
        *composition.task_assets,
        *(asset for assets in composition.skill_assets.values() for asset in assets),
    )
    asset_ids = [asset.asset_id for asset in ordered_assets]
    duplicate_ids = sorted({asset_id for asset_id in asset_ids if asset_ids.count(asset_id) > 1})
    if duplicate_ids:
        raise ValueError(f"Duplicate prompt asset IDs: {duplicate_ids}")

    replacement_uses: set[str] = set()

    def apply_replacements(content: str) -> str:
        placeholders = set(_REPLACEMENT.findall(content))
        unknown = sorted(placeholders - composition.replacements.keys())
        if unknown:
            raise ValueError(f"Unknown prompt replacements: {unknown}")
        replacement_uses.update(placeholders)
        for name in sorted(placeholders):
            content = content.replace(f"{{{{{name}}}}}", composition.replacements[name])
        unresolved = sorted(set(_REPLACEMENT.findall(content)))
        if unresolved:
            raise ValueError(f"Unresolved prompt replacements: {unresolved}")
        return content

    def resolve(assets: tuple[PromptAsset, ...]) -> str:
        return apply_replacements("\n\n".join(asset.content.rstrip("\n") for asset in assets) + "\n")

    system_context = resolve(composition.system_assets)
    task_prompt = resolve(composition.task_assets)
    skills = {
        name: apply_replacements(_compose_skill(name, assets))
        for name, assets in composition.skill_assets.items()
    }
    unused = sorted(composition.replacements.keys() - replacement_uses)
    if unused:
        raise ValueError(f"Unused prompt replacements: {unused}")

    return ResolvedPromptAssets(
        system_context=system_context,
        task_prompt=task_prompt,
        skills=skills,
        provenance=tuple(
            ResolvedPromptAsset(
                asset_id=asset.asset_id,
                version=asset.version,
                source=asset.source,
                sha256=sha256_digest(asset.content),
                bytes=len(asset.content.encode("utf-8")),
            )
            for asset in ordered_assets
        ),
    )


def _parse_skill_document(content: str, *, expected_name: str | None = None) -> _SkillDocument:
    if not content.startswith("---\n"):
        raise ValueError("SKILL.md must start with YAML frontmatter")
    delimiter = content.find("\n---\n", 4)
    if delimiter < 0:
        raise ValueError("SKILL.md frontmatter must end with ---")
    metadata = yaml.safe_load(content[4:delimiter])
    if not isinstance(metadata, dict) or set(metadata) != _SKILL_FRONTMATTER_KEYS:
        raise ValueError("SKILL.md frontmatter must contain exactly name and description")
    name = metadata.get("name")
    description = metadata.get("description")
    if not isinstance(name, str) or not 1 <= len(name) <= 64 or _SKILL_NAME.fullmatch(name) is None:
        raise ValueError("SKILL.md name must be 1-64 lowercase alphanumeric or hyphen characters")
    if expected_name is not None and name != expected_name:
        raise ValueError(f"SKILL.md name {name!r} does not match folder {expected_name!r}")
    if not isinstance(description, str) or not description.strip() or len(description) > 1024:
        raise ValueError("SKILL.md description must be a non-empty string of at most 1024 characters")
    body = content[delimiter + len("\n---\n") :].lstrip("\n")
    if not body.strip():
        raise ValueError("SKILL.md body must be non-empty")
    return _SkillDocument(name=name, description=description.strip(), body=body)


def validate_skill_document(content: str, *, expected_name: str) -> None:
    _parse_skill_document(content, expected_name=expected_name)


def _compose_skill(name: str, assets: tuple[PromptAsset, ...]) -> str:
    if not 1 <= len(name) <= 64 or _SKILL_NAME.fullmatch(name) is None:
        raise ValueError(f"Invalid composed skill name: {name!r}")
    documents = tuple(_parse_skill_document(asset.content) for asset in assets)
    descriptions = tuple(dict.fromkeys(document.description for document in documents))
    description = " ".join(descriptions)
    if len(description) > 1024:
        raise ValueError(f"Composed skill description exceeds 1024 characters: {name!r}")
    body = "\n\n".join(document.body.rstrip("\n") for document in documents)
    return (
        "---\n"
        f"name: {name}\n"
        f"description: {json.dumps(description, ensure_ascii=True)}\n"
        "---\n\n"
        f"{body}\n"
    )
