from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from types import MappingProxyType
from typing import Mapping

from autosaddler.v2.core.domain import to_json_value
from autosaddler.v2.prompting.assets import validate_skill_document
from autosaddler.v2.prompting.models import Capability, SessionSpec


@dataclass(frozen=True, slots=True)
class RenderedSession:
    provider: str
    workspace: Path
    task_prompt: str
    system_context: str
    instruction_path: Path
    skill_directory: Path
    output_schema_path: Path
    output_path: Path
    allowed_tools: tuple[str, ...]
    session_id: str | None = None
    trace_dir: Path | None = None


class WorkspaceRenderer:
    def __init__(
        self,
        *,
        provider: str,
        instruction_file: str,
        skill_directory: str,
        capability_tools: Mapping[Capability, tuple[str, ...]],
    ) -> None:
        self.provider = provider
        self.instruction_file = instruction_file
        self.skill_directory = skill_directory
        self.capability_tools = MappingProxyType(dict(capability_tools))

    def render(
        self,
        spec: SessionSpec,
        workspace: Path,
        *,
        session_id: str | None = None,
        trace_dir: Path | None = None,
    ) -> RenderedSession:
        root = workspace.resolve()
        root.mkdir(parents=True, exist_ok=True)
        instruction_path = _contained_path(root, self.instruction_file)
        skill_directory = _contained_path(root, self.skill_directory)
        output_schema_path = _contained_path(root, ".autosaddler/session_output_schema.json")
        output_path = _contained_path(root, ".autosaddler/session_output.json")
        output_path.unlink(missing_ok=True)

        if skill_directory.is_dir():
            shutil.rmtree(skill_directory)
        elif skill_directory.exists():
            skill_directory.unlink()

        reserved = {instruction_path, output_schema_path, output_path}
        for relative_path, content in spec.workspace_files.items():
            target = _contained_path(root, relative_path)
            if target in reserved:
                raise ValueError(f"Session workspace file collides with provider asset: {relative_path}")
            if target == skill_directory or target.is_relative_to(skill_directory):
                raise ValueError(f"Session workspace file collides with provider skill directory: {relative_path}")
            _write_text(target, content)

        _write_text(
            instruction_path,
            (
                f"{spec.system_context.rstrip()}\n\n"
                "## Structured output\n\n"
                "Follow `.autosaddler/session_output_schema.json` and write the final JSON object to "
                "`.autosaddler/session_output.json`.\n"
            ),
        )
        _write_text(
            output_schema_path,
            json.dumps(to_json_value(spec.output_schema), indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        )
        for name, content in spec.skills.items():
            if not name or PurePosixPath(name).name != name:
                raise ValueError(f"Skill names must be one safe path segment: {name!r}")
            validate_skill_document(content, expected_name=name)
            _write_text(skill_directory / name / "SKILL.md", content)

        tools: list[str] = []
        for capability in sorted(spec.capabilities):
            if capability not in self.capability_tools:
                raise ValueError(f"Provider {self.provider!r} does not support capability {capability!r}")
            tools.extend(self.capability_tools[capability])

        return RenderedSession(
            provider=self.provider,
            workspace=root,
            task_prompt=spec.task_prompt,
            system_context=spec.system_context,
            instruction_path=instruction_path,
            skill_directory=skill_directory,
            output_schema_path=output_schema_path,
            output_path=output_path,
            allowed_tools=tuple(dict.fromkeys(tools)),
            session_id=session_id,
            trace_dir=trace_dir,
        )


def _contained_path(root: Path, relative_path: str) -> Path:
    path = PurePosixPath(relative_path)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Provider asset path must be workspace-relative: {relative_path}")
    target = root.joinpath(*path.parts).resolve()
    if not target.is_relative_to(root):
        raise ValueError(f"Provider asset escapes the workspace: {relative_path}")
    return target


def _write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


CLAUDE_CAPABILITY_TO_TOOLS: Mapping[Capability, tuple[str, ...]] = MappingProxyType(
    {
        "read_workspace": ("Read", "Glob", "Grep"),
        "edit_workspace": ("Edit", "Write"),
        "run_commands": ("Bash",),
        "load_skills": ("Skill",),
        "network": ("WebFetch", "WebSearch"),
    }
)

COPILOT_CAPABILITY_TO_TOOLS: Mapping[Capability, tuple[str, ...]] = MappingProxyType(
    {
        "read_workspace": ("view", "glob", "grep"),
        "edit_workspace": ("edit",),
        "run_commands": ("bash",),
        "load_skills": ("skill",),
        "network": ("web",),
    }
)


def claude_renderer() -> WorkspaceRenderer:
    return WorkspaceRenderer(
        provider="claude",
        instruction_file="CLAUDE.md",
        skill_directory=".claude/skills",
        capability_tools=CLAUDE_CAPABILITY_TO_TOOLS,
    )


def copilot_renderer() -> WorkspaceRenderer:
    return WorkspaceRenderer(
        provider="copilot",
        instruction_file="AGENTS.md",
        skill_directory=".copilot/skills",
        capability_tools=COPILOT_CAPABILITY_TO_TOOLS,
    )