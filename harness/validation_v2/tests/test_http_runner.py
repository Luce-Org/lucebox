import json

import pytest

from harness.validation_v2.http_runner import parse_sse_lines


def test_sse_parser_reconstructs_content_and_usage() -> None:
    chunks = [
        b'data: {"choices":[{"delta":{"content":"hello "}}]}\n',
        b'data: {"choices":[{"delta":{"content":"world"}}],"usage":{"completion_tokens":2}}\n',
        b"data: [DONE]\n",
    ]
    text, usage, parsed = parse_sse_lines(chunks)
    assert text == "hello world"
    assert usage["completion_tokens"] == 2
    assert len(parsed) == 2


def test_sse_parser_rejects_non_data_lines() -> None:
    with pytest.raises(ValueError, match="non-SSE"):
        parse_sse_lines([b"not data\n"])


def test_sse_parser_rejects_non_object_chunks() -> None:
    with pytest.raises(ValueError, match="not a JSON object"):
        parse_sse_lines([f"data: {json.dumps([1, 2])}\n".encode()])
