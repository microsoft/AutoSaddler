from __future__ import annotations

import asyncio
from pathlib import Path

from autosaddler.v2.core.domain import Cost
from autosaddler.v2.plugins.fake import FakePromptPack
from autosaddler.v2.prompting.models import SessionRequest
from autosaddler.v2.providers.fake import FakeAgentProvider, PaidWorkLedger


def test_fake_provider_executes_plugin_owned_session_contract(tmp_path: Path) -> None:
    pack = FakePromptPack(target_component="instruction", improved_text="improved")
    spec = pack.session(
        "diagnose_patch",
        {"candidate_ids": ["sha256:" + "a" * 64], "train_case_ids": ["train-a"]},
    )
    provider = FakeAgentProvider(PaidWorkLedger(tmp_path / "audit.jsonl"))
    request = SessionRequest(
        session_id="session-1",
        operation_id="run:diagnose:attempt:1",
        spec=spec,
        workspace=tmp_path / "workspace",
        timeout_seconds=10,
    )

    result = asyncio.run(provider.run(request))

    assert result.status == "completed"
    assert result.structured_output["updates"] == {"instruction": "improved"}
    assert result.cost == Cost(sessions=1, input_tokens=5, output_tokens=3)