from __future__ import annotations

import json
import os
from pathlib import Path

from autosaddler.v2.core.domain import Cost, canonical_json
from autosaddler.v2.prompting.models import SessionRequest, SessionResult, Usage


class PaidWorkLedger:
    """A deterministic fake-only tripwire for repeated logical paid work."""

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()

    def record(self, kind: str, key: str) -> None:
        existing = {(entry["kind"], entry["key"]) for entry in self.entries()}
        if (kind, key) in existing:
            raise RuntimeError(f"Duplicate paid work detected: {kind} {key}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as destination:
            destination.write(canonical_json({"kind": kind, "key": key}) + "\n")
            destination.flush()
            os.fsync(destination.fileno())

    def entries(self) -> tuple[dict[str, str], ...]:
        if not self.path.exists():
            return ()
        entries = []
        for line in self.path.read_text(encoding="utf-8").splitlines():
            value = json.loads(line)
            if not isinstance(value, dict) or not isinstance(value.get("kind"), str) or not isinstance(value.get("key"), str):
                raise TypeError("Fake paid-work ledger contains a malformed entry")
            entries.append({"kind": value["kind"], "key": value["key"]})
        return tuple(entries)


class FakeAgentProvider:
    def __init__(self, ledger: PaidWorkLedger) -> None:
        self.ledger = ledger

    async def run(self, request: SessionRequest) -> SessionResult:
        self.ledger.record("session", request.operation_id)
        response_text = request.spec.workspace_files.get(".autosaddler/fake_response.json")
        if response_text is None:
            return SessionResult(
                status="failed",
                structured_output=None,
                raw_response="",
                tool_calls=(),
                usage=(),
                cost=Cost(sessions=1),
                error="Fake session response asset is missing",
            )
        response = json.loads(response_text)
        if not isinstance(response, dict):
            raise TypeError("Fake session response must be a JSON object")
        usage = Usage(input_tokens=5, output_tokens=3, model="fake-deterministic-v1")
        return SessionResult(
            status="completed",
            structured_output=response,
            raw_response=canonical_json(response),
            tool_calls=(),
            usage=(usage,),
            cost=Cost(sessions=1, input_tokens=usage.input_tokens, output_tokens=usage.output_tokens),
        )