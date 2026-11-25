"""Unit tests for lucebench._display (console row formatting).

Table-driven coverage of the easy-to-regress display conventions:
the throughput ``*``-fallback rules and the prefill-vs-ttft precedence
ladder in the timing column.
"""

from __future__ import annotations

import pytest

from lucebench._display import (
    _format_throughput,
    _format_timing,
    _format_tokens,
    format_row,
)


# ── Throughput column ───────────────────────────────────────────────────
@pytest.mark.parametrize(
    "row, timings, wall, expected",
    [
        # Server-reported decode rate wins (no `*`).
        ({"completion_tokens": 100}, {"decode_tokens_per_sec": 87.4}, 2.0, "87tps"),
        # Sub-10 rate renders with one decimal.
        ({"completion_tokens": 100}, {"decode_tokens_per_sec": 0.3}, 2.0, "0.3tps"),
        # No server rate, enough tokens + wall → wall-clock fallback marked `*`.
        ({"completion_tokens": 20}, {}, 2.0, "10tps*"),
        # Fewer than 8 completion tokens → drop the rate entirely.
        ({"completion_tokens": 5}, {}, 2.0, ""),
        # No wall → no fallback.
        ({"completion_tokens": 20}, {}, 0, ""),
        # No usable counts at all.
        ({}, {}, 2.0, ""),
    ],
)
def test_format_throughput(row, timings, wall, expected):
    assert _format_throughput(row, timings, wall) == expected


# ── Timing column precedence ────────────────────────────────────────────
@pytest.mark.parametrize(
    "row, timings, wall, expected",
    [
        # Both server prefill+decode present → no `*`, no wall.
        ({}, {"prefill_ms": 200, "decode_ms": 800}, 1.5, "prefill=200ms decode=800ms"),
        # prefill only → prefill + wall.
        ({}, {"prefill_ms": 200}, 1.5, "prefill=200ms wall=1.50s"),
        # Server prefill_ms beats streaming ttft when both exist (ttft dropped).
        (
            {"ttft_seconds": 0.5},
            {"prefill_ms": 200, "decode_ms": 800},
            1.5,
            "prefill=200ms decode=800ms",
        ),
        # No prefill, ttft + decode → ttft marked `*`.
        ({"ttft_seconds": 0.5}, {"decode_ms": 800}, 1.5, "ttft=500ms* decode=800ms"),
        # ttft only → ttft* + wall.
        ({"ttft_seconds": 0.5}, {}, 1.5, "ttft=500ms* wall=1.50s"),
        # decode only → decode + wall.
        ({}, {"decode_ms": 800}, 1.5, "decode=800ms wall=1.50s"),
        # Nothing → plain wall.
        ({}, {}, 1.5, "wall=1.50s"),
        # >=1s renders as seconds.
        ({}, {"prefill_ms": 1500, "decode_ms": 3500}, 1.0, "prefill=1.5s decode=3.5s"),
    ],
)
def test_format_timing(row, timings, wall, expected):
    assert _format_timing(row, timings, wall) == expected


# ── Token column ────────────────────────────────────────────────────────
def test_format_tokens_splits_thinking():
    out = _format_tokens(
        {"prompt_tokens": 10, "completion_tokens": 30, "reasoning_tokens": 12}
    )
    assert out == "in=10 think=12 out=18"


def test_format_tokens_no_reasoning():
    assert _format_tokens({"prompt_tokens": 10, "completion_tokens": 30}) == "in=10 out=30"


def test_format_tokens_empty():
    assert _format_tokens({}) == ""


# ── Full row line ───────────────────────────────────────────────────────
def test_format_row_pass_and_columns():
    line = format_row(
        1,
        {
            "source": "ds4",
            "case_id": "c1",
            "wall_seconds": 2.0,
            "completion_tokens": 20,
            "prompt_tokens": 10,
            "timings": {"decode_tokens_per_sec": 50.0, "prefill_ms": 100, "decode_ms": 400},
        },
        {"pass": True, "given": "42", "correct": "42"},
    )
    assert "PASS" in line
    assert "50tps" in line
    assert "prefill=100ms decode=400ms" in line
    assert "in=10 out=20" in line


def test_format_row_fail_handles_nonmapping_timings():
    # A non-dict timings value must not crash; falls back to wall time.
    line = format_row(2, {"timings": "garbage", "wall_seconds": 1.0}, {"pass": False})
    assert "FAIL" in line
    assert "wall=1.00s" in line
