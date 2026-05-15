from __future__ import annotations

import re
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

RESPONSES_RUNTIME_PATH = Path(__file__).resolve()


class ResponsesAdapterError(RuntimeError):
    pass


def _responses_input(messages: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        if role not in {"assistant", "developer", "system", "user"}:
            raise ResponsesAdapterError(f"Unsupported Responses message role: {role!r}")
        content = message.get("content")
        if isinstance(content, str):
            converted.append({"role": role, "content": content})
            continue
        if not isinstance(content, list) or not content:
            raise ResponsesAdapterError("Responses message content must be text or a non-empty list")

        parts: list[dict[str, Any]] = []
        for part in content:
            if not isinstance(part, dict):
                raise ResponsesAdapterError("Responses message content parts must be mappings")
            part_type = part.get("type")
            if part_type == "text":
                text = part.get("text")
                if not isinstance(text, str):
                    raise ResponsesAdapterError("Responses text content must be a string")
                parts.append({"type": "input_text", "text": text})
            elif part_type == "image_url":
                image = part.get("image_url")
                image_url = image.get("url") if isinstance(image, dict) else None
                if not isinstance(image_url, str) or not image_url:
                    raise ResponsesAdapterError("Responses image content requires a URL")
                parts.append(
                    {"type": "input_image", "detail": "auto", "image_url": image_url}
                )
            else:
                raise ResponsesAdapterError(
                    f"Unsupported Responses message content type: {part_type!r}"
                )
        converted.append({"role": role, "content": parts})
    return converted


def _required_usage_int(usage: object, name: str) -> int:
    value = getattr(usage, name, None)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ResponsesAdapterError(f"Responses usage.{name} must be a non-negative integer")
    return value


def _responses_chat_completion(
    engine: Any,
    messages: list[dict[str, Any]],
    stop_sequences: Sequence[str] = (),
    **kwargs: Any,
) -> tuple[str, dict[str, int | float]]:
    del kwargs
    client = engine._client
    if client is None:
        raise ResponsesAdapterError("Copilot client not initialized")

    converted_messages = [
        engine._convert_message_to_litellm_format(message) for message in messages
    ]
    request: dict[str, Any] = {
        "model": engine.model_config.model_name,
        "input": _responses_input(converted_messages),
    }
    reasoning_effort = engine.model_config.reasoning_effort
    if reasoning_effort is not None:
        request["reasoning"] = {"effort": reasoning_effort}

    started = time.monotonic()
    try:
        from openai import OpenAIError

        response = client.responses.create(**request)
    except OpenAIError as error:
        raise ResponsesAdapterError("Copilot Responses request failed") from error
    completion_duration = time.monotonic() - started

    status = getattr(response, "status", None)
    if status != "completed":
        raise ResponsesAdapterError(f"Copilot Responses request did not complete: {status!r}")
    output_text = getattr(response, "output_text", None)
    if not isinstance(output_text, str) or not output_text:
        raise ResponsesAdapterError("Copilot Responses request produced no output text")

    result = re.sub(
        r"\[\[(True|False)\]\]|(True|False)",
        lambda match: match.group(0) if match.group(1) else match.group(2).lower(),
        output_text,
    )
    for stop_token in stop_sequences:
        result = result.split(stop_token, maxsplit=1)[0]

    usage = getattr(response, "usage", None)
    if usage is None:
        raise ResponsesAdapterError("Copilot Responses request omitted token usage")
    details = getattr(usage, "output_tokens_details", None)
    reasoning_tokens = 0
    if details is not None:
        reasoning_tokens = _required_usage_int(details, "reasoning_tokens")
    metadata: dict[str, int | float] = {
        "prompt_tokens": _required_usage_int(usage, "input_tokens"),
        "completion_tokens": _required_usage_int(usage, "output_tokens"),
        "total_tokens": _required_usage_int(usage, "total_tokens"),
        "reasoning_tokens": reasoning_tokens,
        "completion_duration": completion_duration,
    }
    return result, metadata


def _make_responses_dispatch(
    original: Callable[..., tuple[str, dict[str, int | float] | None]],
) -> Callable[..., tuple[str, dict[str, int | float] | None]]:
    def dispatch(
        engine: Any,
        messages: list[dict[str, Any]],
        stop_sequences: Sequence[str] = (),
        **kwargs: Any,
    ) -> tuple[str, dict[str, int | float] | None]:
        if engine.model_config.provider != "copilot":
            return original(engine, messages, stop_sequences=stop_sequences, **kwargs)
        try:
            return _responses_chat_completion(
                engine,
                messages,
                stop_sequences=stop_sequences,
                **kwargs,
            )
        except ResponsesAdapterError as error:
            from are.simulation.agents.llm.llm_engine import LLMEngineException

            raise LLMEngineException(str(error)) from error

    return dispatch


def install_responses_adapter() -> None:
    from are.simulation.agents.llm.litellm.litellm_engine import LiteLLMEngine

    if getattr(LiteLLMEngine, "_autosaddler_responses_adapter_installed", False):
        return
    original = LiteLLMEngine.chat_completion
    LiteLLMEngine.chat_completion = _make_responses_dispatch(original)
    LiteLLMEngine._autosaddler_responses_adapter_installed = True


def main(argv: Sequence[str] | None = None) -> None:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments[:3] != ["--model-wire-api", "responses", "--"]:
        raise SystemExit(
            "expected '--model-wire-api responses --' before are-benchmark arguments"
        )
    install_responses_adapter()

    from are.simulation.benchmark.cli import main as benchmark_main

    benchmark_main(args=arguments[3:], prog_name="are-benchmark")


if __name__ == "__main__":
    main()