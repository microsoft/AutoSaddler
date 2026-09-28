"""Load inspectable, strategy-specific CLAUDE and session prompt assets."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from autosaddler.v1.proposer.autosaddler.strategy import (
    UNSEEN_SCENARIO_EXPLORATION_SESSION,
    SessionNumber,
    StrategySpec,
    resolve_strategy,
)

COMMON_SKILLS = (
    "history-analysis",
    "diagnose",
    "capability-patch",
    "steering-patch",
    "patch-verification",
)

_SESSION_FILENAMES = {
    0: "session0_candidate_selection.md",
    1: "session1_diagnose_patch.md",
    2: "session2_reflection.md",
    3: "session3_pattern_extraction.md",
    UNSEEN_SCENARIO_EXPLORATION_SESSION: (
        "session3_5_unseen_scenario_exploration.md"
    ),
}


@dataclass(frozen=True)
class PromptBundle:
    """Resolved static instructions and installed capabilities."""

    strategy: StrategySpec
    claude_path: Path
    claude_md: str
    skill_names: tuple[str, ...]
    session_numbers: tuple[SessionNumber, ...]
    session_prompt_paths: dict[SessionNumber, Path]
    pattern_cli_capabilities: frozenset[str]
    renderer_sha256: str
    sha256: str


def _validate_options(session_scope: str) -> None:
    if session_scope not in ("full", "diagnosis_only"):
        raise ValueError("session_scope must be 'full' or 'diagnosis_only'")


def get_claude_prompt_path(
    asset_root: Path,
    canonical_root: Path,
    spec: StrategySpec,
) -> Path:
    if spec.name.value == "autosaddler":
        return canonical_root / "CLAUDE.md"
    return asset_root / spec.name.value / "CLAUDE.md"


def get_session_prompt_path(
    asset_root: Path,
    canonical_root: Path,
    spec: StrategySpec,
    session: SessionNumber,
) -> Path:
    if session in (0, 1, 2):
        return canonical_root / "session_prompts" / _SESSION_FILENAMES[session]
    if session == 3:
        return asset_root / "shared_session_prompts" / _SESSION_FILENAMES[session]
    session_dir = asset_root / spec.name.value / "session_prompts"
    if session == 4:
        return asset_root / "shared_session_prompts" / "session4_arm_scoring.md"
    return session_dir / _SESSION_FILENAMES[session]


def load_prompt_bundle(
    *,
    asset_root: Path,
    canonical_root: Path,
    skill_root: Path,
    sampling_strategy: str = "autosaddler",
    session_scope: str = "full",
    renderer_source: str = "",
) -> PromptBundle:
    """Load one complete static strategy bundle and compute its identity hash."""
    _validate_options(session_scope)
    configured_spec = resolve_strategy(sampling_strategy)
    effective_spec = (
        resolve_strategy("autosaddler")
        if session_scope == "diagnosis_only"
        else configured_spec
    )
    sessions = (
        (1,)
        if session_scope == "diagnosis_only"
        else effective_spec.enabled_sessions
    )
    claude_path = get_claude_prompt_path(
        asset_root,
        canonical_root,
        effective_spec,
    )
    session_prompt_paths = {
        session: get_session_prompt_path(
            asset_root,
            canonical_root,
            effective_spec,
            session,
        )
        for session in sessions
    }
    skills = COMMON_SKILLS + effective_spec.extra_skills
    claude_md = claude_path.read_text(encoding="utf-8")
    renderer_sha256 = hashlib.sha256(renderer_source.encode()).hexdigest()
    digest_parts = [
        effective_spec.name.value,
        session_scope,
        claude_md,
        renderer_sha256,
        *sorted(effective_spec.pattern_cli_capabilities),
    ]
    digest_parts.extend(
        f"skill:{skill}\0{(skill_root / skill / 'SKILL.md').read_text(encoding='utf-8')}"
        for skill in skills
    )
    digest_parts.extend(
        f"session:{session}\0{path.read_text(encoding='utf-8')}"
        for session, path in session_prompt_paths.items()
    )
    return PromptBundle(
        strategy=effective_spec,
        claude_path=claude_path,
        claude_md=claude_md,
        skill_names=skills,
        session_numbers=sessions,
        session_prompt_paths=session_prompt_paths,
        pattern_cli_capabilities=effective_spec.pattern_cli_capabilities,
        renderer_sha256=renderer_sha256,
        sha256=hashlib.sha256("\n".join(digest_parts).encode()).hexdigest(),
    )
