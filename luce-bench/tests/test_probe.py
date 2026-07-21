"""Smoke tests for the ``luce-bench-probe`` console script (lucebench.probe).

Covers the pure pieces (mode-table construction, case lookup, row
extraction, summary rendering) plus an end-to-end ``main()`` run against
the in-process mock OpenAI server from conftest.py. The probe had zero
tests before; these guard its arg parsing and per-mode request shapes.
"""

from __future__ import annotations

import json
import sys

import pytest

from lucebench import probe

_VALID_CASE_ID = "aime2025-01"


def _a_case() -> dict:
    return {"id": _VALID_CASE_ID, "question": "What is 2+2?"}


def test_load_case_found_and_missing():
    case = probe.load_case(_VALID_CASE_ID)
    assert case["id"] == _VALID_CASE_ID
    with pytest.raises(SystemExit):
        probe.load_case("does-not-exist-zzz")


@pytest.mark.parametrize("mode", probe.DEFAULT_MODES.split(","))
def test_build_body_default_modes_set_thinking(mode):
    body = probe.build_body("m", _a_case(), mode, 256, 0.0, 1.0, 0)
    # Every default mode emits a thinking block + the chat-template toggle.
    assert "thinking" in body
    assert "enable_thinking" in body["chat_template_kwargs"]
    assert body["model"] == "m"
    assert body["max_tokens"] == 256


def test_build_body_mode_specifics():
    low = probe.build_body("m", _a_case(), "think-low", 256, 0.0, 1.0, 0)
    assert low["thinking"] == {"type": "enabled", "budget_tokens": 1024}
    med = probe.build_body("m", _a_case(), "think-medium", 256, 0.0, 1.0, 0)
    assert med["thinking"] == {"type": "enabled", "budget_tokens": 4096}
    nothink = probe.build_body("m", _a_case(), "nothink", 256, 0.0, 1.0, 0)
    assert nothink["thinking"] == {"type": "disabled"}
    raw = probe.build_body("m", _a_case(), "think-raw-noprompt", 256, 0.0, 1.0, 0)
    # System message is blanked for the raw-noprompt probe.
    assert raw["messages"][0]["content"] == ""


def test_build_body_top_k_only_when_positive():
    assert "top_k" not in probe.build_body("m", _a_case(), "nothink", 256, 0.0, 1.0, 0)
    assert probe.build_body("m", _a_case(), "nothink", 256, 0.0, 1.0, 5)["top_k"] == 5


def test_build_body_unknown_mode_raises():
    with pytest.raises(SystemExit):
        probe.build_body("m", _a_case(), "bogus-mode", 256, 0.0, 1.0, 0)


def test_extract_row_pulls_usage_and_timings():
    raw = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "42", "reasoning_content": "think"},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "thinking_tokens": 3,
            "timings": {"prefill_ms": 5.0, "decode_ms": 50.0, "decode_tokens_per_sec": 100.0},
        },
    }
    row = probe.extract_row(_VALID_CASE_ID, "nothink", {}, raw, 1.234)
    assert row["prompt_tokens"] == 10
    assert row["completion_tokens"] == 5
    assert row["thinking_tokens"] == 3
    assert row["content_len_chars"] == 2
    assert row["reasoning_len_chars"] == 5
    assert row["decode_tok_per_s"] == 100.0
    assert row["wall_s"] == 1.234


def test_render_summary_has_header_and_row():
    rows = [
        {
            "mode": "nothink",
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "thinking_tokens": 0,
            "content_len_chars": 2,
            "reasoning_len_chars": 0,
            "finish_reason": "stop",
            "wall_s": 0.1,
            "decode_tok_per_s": 100.0,
        }
    ]
    out = probe.render_summary(rows)
    assert "| mode |" in out
    assert "| nothink |" in out


def test_probe_main_against_mock_server(mock_openai_server, tmp_path, monkeypatch):
    """End-to-end: parse args, POST one mode, write the snapshot files."""
    url, captured, _ = mock_openai_server
    out_dir = tmp_path / "probe-out"
    argv = [
        "luce-bench-probe",
        "--url",
        url,
        "--model",
        "mock-model",
        "--case-id",
        _VALID_CASE_ID,
        "--modes",
        "nothink",
        "--max-tokens",
        "32",
        "--timeout",
        "10",
        "--out-dir",
        str(out_dir),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    rc = probe.main()
    assert rc == 0

    # Snapshot files written.
    assert (out_dir / "nothink.json").exists()
    summary = json.loads((out_dir / "_summary.json").read_text())
    assert summary["case_id"] == _VALID_CASE_ID
    assert summary["model"] == "mock-model"
    assert len(summary["rows"]) == 1
    assert (out_dir / "_summary.md").exists()

    # The server saw exactly one POST carrying the requested model.
    posts = [c for c in captured if c["path"] == "/v1/chat/completions"]
    assert len(posts) == 1
    assert posts[0]["body"]["model"] == "mock-model"
