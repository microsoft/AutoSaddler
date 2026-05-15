from __future__ import annotations

from types import SimpleNamespace

import pytest


class FakeResponses:
    def __init__(self, response: object) -> None:
        self.response = response
        self.request: dict[str, object] | None = None

    def create(self, **request: object) -> object:
        self.request = request
        return self.response


def _engine(response: object, *, provider: str = "copilot", reasoning_effort=None):
    responses = FakeResponses(response)
    engine = SimpleNamespace(
        _client=SimpleNamespace(responses=responses),
        model_config=SimpleNamespace(
            model_name="gpt-5.5",
            provider=provider,
            reasoning_effort=reasoning_effort,
        ),
        _convert_message_to_litellm_format=lambda message: message,
    )
    return engine, responses


def test_responses_adapter_converts_input_and_normalizes_usage() -> None:
    from autosaddler.v2.plugins.meta_are.responses_runtime import (
        _responses_chat_completion,
    )

    response = SimpleNamespace(
        status="completed",
        output_text="True then False<end_action>ignored",
        usage=SimpleNamespace(
            input_tokens=11,
            output_tokens=7,
            total_tokens=18,
            output_tokens_details=SimpleNamespace(reasoning_tokens=3),
        ),
    )
    engine, responses = _engine(response)

    result, metadata = _responses_chat_completion(
        engine,
        [
            {"role": "system", "content": "system text"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "inspect"},
                    {
                        "type": "image_url",
                        "image_url": {"url": "data:image/png;base64,AAAA"},
                    },
                ],
            },
        ],
        stop_sequences=["<end_action>"],
    )

    assert result == "true then false"
    assert metadata["prompt_tokens"] == 11
    assert metadata["completion_tokens"] == 7
    assert metadata["total_tokens"] == 18
    assert metadata["reasoning_tokens"] == 3
    assert metadata["completion_duration"] >= 0
    assert responses.request == {
        "model": "gpt-5.5",
        "input": [
            {"role": "system", "content": "system text"},
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": "inspect"},
                    {
                        "type": "input_image",
                        "detail": "auto",
                        "image_url": "data:image/png;base64,AAAA",
                    },
                ],
            },
        ],
    }


def test_responses_adapter_sends_only_explicit_reasoning() -> None:
    from autosaddler.v2.plugins.meta_are.responses_runtime import (
        _responses_chat_completion,
    )

    response = SimpleNamespace(
        status="completed",
        output_text="done",
        usage=SimpleNamespace(
            input_tokens=1,
            output_tokens=2,
            total_tokens=3,
            output_tokens_details=None,
        ),
    )
    engine, responses = _engine(response, reasoning_effort="xhigh")

    _responses_chat_completion(engine, [{"role": "user", "content": "work"}])

    assert responses.request is not None
    assert responses.request["reasoning"] == {"effort": "xhigh"}


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (
            SimpleNamespace(status="completed", output_text="", usage=object()),
            "output text",
        ),
        (
            SimpleNamespace(status="completed", output_text="done", usage=None),
            "token usage",
        ),
        (
            SimpleNamespace(status="incomplete", output_text="partial", usage=object()),
            "did not complete",
        ),
    ],
)
def test_responses_adapter_rejects_incomplete_or_malformed_responses(
    response: object,
    message: str,
) -> None:
    from autosaddler.v2.plugins.meta_are.responses_runtime import (
        ResponsesAdapterError,
        _responses_chat_completion,
    )

    engine, _ = _engine(response)

    with pytest.raises(ResponsesAdapterError, match=message):
        _responses_chat_completion(engine, [{"role": "user", "content": "work"}])


def test_responses_dispatch_preserves_non_copilot_engine() -> None:
    from autosaddler.v2.plugins.meta_are.responses_runtime import (
        _make_responses_dispatch,
    )

    calls: list[tuple[object, object]] = []

    def original(engine, messages, **kwargs):
        calls.append((engine, kwargs["stop_sequences"]))
        return "judge", None

    engine, _ = _engine(object(), provider="azure")
    dispatch = _make_responses_dispatch(original)

    assert dispatch(engine, [], stop_sequences=["stop"]) == ("judge", None)
    assert calls == [(engine, ["stop"])]