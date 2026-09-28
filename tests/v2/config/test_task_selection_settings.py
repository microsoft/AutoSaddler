from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from autosaddler.v2.config.models import RunConfig
from autosaddler.v2.config.registry import build_runtime
from autosaddler.v2.core.curriculum import ActiveSaddlerTaskSelectionPolicy


def write_config(root: Path, value: dict) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(value, sort_keys=False))
    return path


def test_settings_are_serialized_only_when_configured(tmp_path: Path, activesaddler_config) -> None:
    adaptive = RunConfig.load(write_config(tmp_path / "adaptive", activesaddler_config(tmp_path)))
    passive_value = activesaddler_config(tmp_path)
    passive_value["optimization"]["task_selection"] = {"type": "epoch_shuffled", "batch_size": 2, "seed": 1}
    passive = RunConfig.load(write_config(tmp_path / "passive", passive_value))

    assert adaptive.optimization.task_selection.settings["pattern_extraction_timeout_seconds"] == 11
    assert adaptive.as_mapping()["optimization"]["task_selection"]["settings"]["ema_eta"] == 0.9
    assert passive.optimization.task_selection.settings is None
    assert passive.as_mapping()["optimization"]["task_selection"] == {
        "type": "epoch_shuffled",
        "batch_size": 2,
        "seed": 1,
    }


def test_registry_builds_activesaddler_from_its_settings(
    tmp_path: Path,
    curriculum_registry,
    activesaddler_config,
) -> None:
    runtime = build_runtime(
        write_config(tmp_path, activesaddler_config(tmp_path)),
        run_id="activesaddler",
        registry=curriculum_registry,
    )
    policy = runtime.policies.task_selection

    assert isinstance(policy, ActiveSaddlerTaskSelectionPolicy)
    assert policy.settings_record() == {
        "softmax_temperature": 0.15,
        "min_prob": 0.0,
        "ema_eta": 0.9,
        "pattern_extraction_timeout_seconds": 11.0,
        "arm_scoring_timeout_seconds": 12.0,
    }


@pytest.mark.parametrize(
    ("policy_type", "settings", "error"),
    [
        ("epoch_shuffled", {"ema_eta": 0.9}, "not supported by 'epoch_shuffled'"),
        ("fixed", {}, "not supported by 'fixed'"),
        ("activesaddler", None, "settings is required"),
        ("activesaddler", {"softmax_temperature": 0.15}, r"missing=\['arm_scoring_timeout_seconds'"),
        ("activesaddler", {"extra": 1}, r"extra=\['extra'\]"),
    ],
)
def test_registry_rejects_invalid_task_selection_settings(
    tmp_path: Path,
    curriculum_registry,
    activesaddler_config,
    policy_type: str,
    settings: dict | None,
    error: str,
) -> None:
    value = activesaddler_config(tmp_path)
    task_selection = value["optimization"]["task_selection"]
    task_selection["type"] = policy_type
    if settings is None:
        del task_selection["settings"]
    elif set(settings) - {"extra"} or not settings:
        task_selection["settings"] = settings
    else:
        task_selection["settings"] = {**task_selection["settings"], **settings}
    with pytest.raises(ValueError, match=error):
        build_runtime(write_config(tmp_path, value), run_id="invalid", registry=curriculum_registry)


@pytest.mark.parametrize(
    ("key", "value", "error"),
    [
        ("softmax_temperature", 0.0, "softmax_temperature"),
        ("min_prob", 1.0, "min_prob"),
        ("ema_eta", 1.5, "ema_eta"),
        ("arm_scoring_timeout_seconds", 0, "timeouts"),
    ],
)
def test_activesaddler_rejects_out_of_range_settings(
    tmp_path: Path,
    curriculum_registry,
    activesaddler_config,
    key: str,
    value: float,
    error: str,
) -> None:
    config = activesaddler_config(tmp_path)
    config["optimization"]["task_selection"]["settings"][key] = value
    with pytest.raises(ValueError, match=error):
        build_runtime(write_config(tmp_path, config), run_id="range", registry=curriculum_registry)
