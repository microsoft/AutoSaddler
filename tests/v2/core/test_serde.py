from __future__ import annotations

from autosaddler.v2.core.serde import session_result_from


def test_legacy_cache_fields_migrate_to_canonical_input_buckets() -> None:
    result = session_result_from(
        {
            "status": "completed",
            "structured_output": {},
            "raw_response": "",
            "tool_calls": [],
            "usage": [
                {
                    "role": "optimizer",
                    "model": "claude-test",
                    "input_tokens": 10,
                    "cache_read_tokens": 3,
                    "cache_write_tokens": 2,
                    "output_tokens": 4,
                    "total_tokens": 19,
                }
            ],
            "cost": {
                "rollouts": 0,
                "sessions": 1,
                "input_tokens": 15,
                "output_tokens": 4,
                "wall_seconds": 1.0,
                "currency_amount": None,
            },
            "error": None,
        }
    )

    usage = result.usage[0]
    assert usage.input_tokens == 15
    assert usage.cached_input_tokens == 3
    assert usage.uncached_input_tokens == 12
    assert usage.total_tokens == 19
    assert usage.provider_metadata == {"cache_creation_input_tokens": 2}