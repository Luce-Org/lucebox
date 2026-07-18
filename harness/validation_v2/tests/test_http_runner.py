import json

import pytest

from harness.validation_v2.http_runner import (
    parse_sse_lines,
    validate_speculative_evidence,
)


def _config() -> dict[str, object]:
    return {
        "expected_configured_contract": "exact",
        "expected_effective_decode_mode": "autoregressive",
        "expected_fallback_reason": "exact_contract_requires_ar",
        "expected_draft_attached": True,
        "expected_draft_loaded": False,
    }


def _props() -> dict[str, object]:
    return {
        "speculative": {
            "configured_contract": "exact",
            "draft_attached": True,
            "draft_loaded": False,
        }
    }


def _usage() -> dict[str, object]:
    return {
        "completion_tokens": 2,
        "speculative": {
            "configured_contract": "exact",
            "effective_decode_mode": "autoregressive",
            "fallback_reason": "exact_contract_requires_ar",
        },
    }


def test_sse_parser_reconstructs_content_and_usage() -> None:
    chunks = [
        b'data: {"choices":[{"delta":{"content":"hello "}}]}\n',
        b'data: {"choices":[{"delta":{"content":"world"},"finish_reason":"stop"}],"usage":{"completion_tokens":2}}\n',
        b"data: [DONE]\n",
    ]
    text, usage, parsed = parse_sse_lines(chunks)
    assert text == "hello world"
    assert usage["completion_tokens"] == 2
    assert len(parsed) == 2


def test_sse_parser_preserves_nested_speculative_evidence() -> None:
    chunks = [
        b'data: {"choices":[{"delta":{"content":"ok"}}]}\n',
        (
            "data: "
            + json.dumps(
                {
                    "choices": [{"delta": {}, "finish_reason": "stop"}],
                    "usage": _usage(),
                }
            )
            + "\n"
        ).encode(),
        b"data: [DONE]\n",
    ]
    _, usage, _ = parse_sse_lines(chunks)
    assert validate_speculative_evidence(
        usage=usage, props=_props(), config=_config()
    ) == {
        "configured_contract": "exact",
        "effective_decode_mode": "autoregressive",
        "fallback_reason": "exact_contract_requires_ar",
    }


def test_nonstream_usage_requires_nested_speculative_evidence() -> None:
    assert validate_speculative_evidence(
        usage=_usage(), props=_props(), config=_config()
    )["configured_contract"] == "exact"
    flattened = {
        "completion_tokens": 2,
        "configured_contract": "exact",
        "effective_decode_mode": "autoregressive",
        "fallback_reason": "exact_contract_requires_ar",
    }
    with pytest.raises(ValueError, match="nested speculative"):
        validate_speculative_evidence(
            usage=flattened, props=_props(), config=_config()
        )


def test_speculative_evidence_must_match_props_and_matrix() -> None:
    wrong_props = _props()
    wrong_props["speculative"]["draft_loaded"] = True  # type: ignore[index]
    with pytest.raises(ValueError, match="draft_loaded"):
        validate_speculative_evidence(
            usage=_usage(), props=wrong_props, config=_config()
        )

    missing_expectation = _config()
    del missing_expectation["expected_fallback_reason"]
    with pytest.raises(ValueError, match="lacks speculative evidence expectations"):
        validate_speculative_evidence(
            usage=_usage(), props=_props(), config=missing_expectation
        )


def test_sse_parser_rejects_non_data_lines() -> None:
    with pytest.raises(ValueError, match="non-SSE"):
        parse_sse_lines([b"not data\n"])


def test_sse_parser_rejects_non_object_chunks() -> None:
    with pytest.raises(ValueError, match="not a JSON object"):
        parse_sse_lines([f"data: {json.dumps([1, 2])}\n".encode()])


def test_sse_parser_rejects_truncated_or_unfinished_streams() -> None:
    with pytest.raises(ValueError, match=r"without \[DONE\]"):
        parse_sse_lines(
            [b'data: {"choices":[{"delta":{"content":"partial"}}]}\n']
        )
    with pytest.raises(ValueError, match="without a finish reason"):
        parse_sse_lines(
            [
                b'data: {"choices":[{"delta":{"content":"partial"}}]}\n',
                b"data: [DONE]\n",
            ]
        )
