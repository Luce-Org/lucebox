import os
import struct
import json
import asyncio
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from server import (
    build_app, MODEL_NAME, parse_reasoning,
    consume_stream_piece, flush_stream_deltas,
)


@pytest.fixture
def mock_tokenizer():
    tokenizer = MagicMock()
    tokenizer.encode.return_value = [1]
    tokenizer.decode.return_value = "hello"
    tokenizer.apply_chat_template.return_value = "prompt"
    return tokenizer


def test_parse_reasoning_headless_think():
    cleaned, reasoning = parse_reasoning("private chain of thought</think>\n\nvisible answer")
    assert cleaned == "visible answer"
    assert reasoning == "private chain of thought"


def test_parse_reasoning_full_think_tags():
    cleaned, reasoning = parse_reasoning("<think>my reasoning</think>\n\nthe answer")
    assert cleaned == "the answer"
    assert reasoning == "my reasoning"


def test_parse_reasoning_plain_content_when_no_think_segment_present():
    cleaned, reasoning = parse_reasoning("visible answer only")
    assert cleaned == "visible answer only"
    assert reasoning is None


def test_parse_reasoning_truncated_when_prompt_started_in_thinking():
    cleaned, reasoning = parse_reasoning(
        "unfinished private chain of thought",
        started_in_thinking=True,
    )
    assert cleaned == ""
    assert reasoning == "unfinished private chain of thought"


# -- consume_stream_piece / flush_stream_deltas -------------------------

def test_consume_stream_piece_reasoning_to_content():
    """Full transition: reasoning tokens, close tag, content tokens."""
    window, mode = "", "reasoning"
    assert mode == "reasoning"

    all_outputs = []

    # Feed reasoning text
    outputs, window, mode = consume_stream_piece(window, mode, "deep thought")
    all_outputs.extend(outputs)
    assert mode == "reasoning"

    # Feed close tag
    outputs, window, mode = consume_stream_piece(window, mode, "</think>")
    all_outputs.extend(outputs)
    reasoning_parts = [t for k, t in all_outputs if k == "reasoning_content"]
    assert "deep thought" in "".join(reasoning_parts)
    assert mode == "content"

    # Feed content
    outputs, window, mode = consume_stream_piece(window, mode, "visible answer")
    all_outputs.extend(outputs)
    assert mode == "content"

    # Flush remaining
    flushed = flush_stream_deltas(window, mode)
    all_content = [t for k, t in all_outputs if k == "content"] + [t for k, t in flushed if k == "content"]
    assert "visible answer" in "".join(all_content)


def test_consume_stream_piece_tag_split_across_pieces():
    """The </think> tag arrives split across two pieces."""
    window, mode = "", "reasoning"
    all_outputs = []

    outputs, window, mode = consume_stream_piece(window, mode, "thought</th")
    all_outputs.extend(outputs)
    # Tag not yet complete, should stay in reasoning mode
    assert mode == "reasoning"

    outputs, window, mode = consume_stream_piece(window, mode, "ink>answer")
    all_outputs.extend(outputs)
    # Now the tag is complete, should have transitioned
    assert mode == "content"
    # Collect everything emitted across both calls
    all_reasoning = [t for k, t in all_outputs if k == "reasoning_content"]
    all_content = [t for k, t in all_outputs if k == "content"]
    flushed = flush_stream_deltas(window, mode)
    all_content += [t for k, t in flushed if k == "content"]
    assert "thought" in "".join(all_reasoning)
    assert "answer" in "".join(all_content)


def test_consume_stream_piece_content_mode_no_tags():
    """Plain content with no think tags passes through."""
    window, mode = "", "content"
    assert mode == "content"

    outputs, window, mode = consume_stream_piece(window, mode, "hello world")
    assert mode == "content"
    flushed = flush_stream_deltas(window, mode)
    all_text = [t for k, t in outputs if k == "content"] + [t for k, t in flushed if k == "content"]
    assert "hello world" in "".join(all_text)


def test_flush_empty_window():
    assert flush_stream_deltas("", "content") == []
    assert flush_stream_deltas("", "reasoning") == []

@patch("server.subprocess.Popen")
def test_models_endpoint(mock_popen, mock_tokenizer):
    app = build_app(
        target=Path("target.gguf"),
        draft=Path("draft.safetensors"),
        bin_path=Path("test_dflash"),
        budget=22,
        max_ctx=131072,
        tokenizer=mock_tokenizer,
        stop_ids={2}
    )
    client = TestClient(app)
    response = client.get("/v1/models")
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "list"
    assert len(data["data"]) == 1
    assert data["data"][0]["id"] == MODEL_NAME

@patch("server.os.pipe")
@patch("server.subprocess.Popen")
@patch("server.os.read")
def test_chat_completions_non_streaming(mock_os_read, mock_popen, mock_pipe, mock_tokenizer):
    mock_pipe.return_value = (1, 2)
    mock_popen.return_value.poll.return_value = None  # daemon alive
    mock_tokenizer.decode.return_value = "private chain of thought</think>\n\nvisible answer"
    
    app = build_app(
        target=Path("target.gguf"),
        draft=Path("draft.safetensors"),
        bin_path=Path("test_dflash"),
        budget=22,
        max_ctx=131072,
        tokenizer=mock_tokenizer,
        stop_ids={2}
    )
    
    # Mock os.read to return a single token (e.g. 10) and then -1
    mock_os_read.side_effect = [
        struct.pack("<i", 10),
        struct.pack("<i", -1)
    ]
    
    client = TestClient(app)
    response = client.post("/v1/chat/completions", json={
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": False
    })
    
    assert response.status_code == 200
    data = response.json()
    assert data["object"] == "chat.completion"
    assert data["choices"][0]["message"]["content"] == "visible answer"
    assert data["choices"][0]["message"]["reasoning_content"] == "private chain of thought"

@patch("server.os.pipe")
@patch("server.subprocess.Popen")
@patch("server.os.read")
def test_chat_completions_streaming(mock_os_read, mock_popen, mock_pipe, mock_tokenizer):
    mock_pipe.return_value = (1, 2)
    mock_popen.return_value.poll.return_value = None  # daemon alive
    mock_tokenizer.apply_chat_template.return_value = "<think>\n"
    mock_tokenizer.decode.side_effect = [
        "private thought",
        "</think>",
        "visible answer",
    ]
    
    app = build_app(
        target=Path("target.gguf"),
        draft=Path("draft.safetensors"),
        bin_path=Path("test_dflash"),
        budget=22,
        max_ctx=131072,
        tokenizer=mock_tokenizer,
        stop_ids={2}
    )
    
    mock_os_read.side_effect = [
        struct.pack("<i", 10),
        struct.pack("<i", 11),
        struct.pack("<i", 12),
        struct.pack("<i", -1)
    ]
    
    client = TestClient(app)
    response = client.post("/v1/chat/completions", json={
        "model": MODEL_NAME,
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True
    })
    
    assert response.status_code == 200
    assert '"reasoning_content"' in response.text
    assert "</think>" not in response.text
    assert "data: [DONE]" in response.text
