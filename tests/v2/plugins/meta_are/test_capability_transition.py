from __future__ import annotations

from pathlib import Path

import pytest

from test_config import _settings_mapping


def test_settings_without_transition_keys_keep_iteration_schedule(tmp_path: Path) -> None:
    from autosaddler.v2.plugins.meta_are.config import MetaARESettings

    mapping, _ = _settings_mapping(tmp_path)
    settings = MetaARESettings.from_mapping(mapping, base_dir=tmp_path)

    assert settings.capability_transition_mode is None
    assert settings.capability_phase_max_iterations == 0


def test_settings_accept_full_coverage_transition(tmp_path: Path) -> None:
    from autosaddler.v2.plugins.meta_are.config import MetaARESettings

    mapping, _ = _settings_mapping(tmp_path)
    legacy = MetaARESettings.from_mapping(mapping, base_dir=tmp_path)
    mapping.update(
        {
            "capability_phase_iterations": 0,
            "capability_transition_mode": "full_coverage",
            "capability_phase_max_iterations": 30,
        }
    )
    settings = MetaARESettings.from_mapping(mapping, base_dir=tmp_path)

    assert settings.capability_transition_mode == "full_coverage"
    assert settings.capability_phase_max_iterations == 30
    assert settings.execution_fingerprint != legacy.execution_fingerprint


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"capability_transition_mode": "full_coverage"}, "both or neither"),
        (
            {"capability_transition_mode": "epochs", "capability_phase_max_iterations": 0},
            "capability_transition_mode",
        ),
        (
            {"capability_transition_mode": "iterations", "capability_phase_max_iterations": 5},
            "applies only to full_coverage",
        ),
        (
            {"capability_transition_mode": "full_coverage", "capability_phase_max_iterations": 0},
            "must be 0 for full_coverage",
        ),
    ],
)
def test_settings_reject_inconsistent_capability_transitions(tmp_path: Path, overrides: dict, error: str) -> None:
    from autosaddler.v2.plugins.meta_are.config import MetaARESettings

    mapping, _ = _settings_mapping(tmp_path)
    mapping.update(overrides)
    with pytest.raises(ValueError, match=error):
        MetaARESettings.from_mapping(mapping, base_dir=tmp_path)
