"""Tests for check_commit_messages.py."""

from __future__ import annotations

import pytest
from check_commit_messages import problems


@pytest.mark.parametrize(
    "message",
    [
        "feat(ds4): add sparse prefill",
        "fix(server): clamp the rollback window",
        "perf(qwen35/gdn): fuse the gate",
        "refactor(server,harness)!: rename engine\n\nBody text.\n",
        "docs(contributing): explain the compact label\n\nCo-Authored-By: X <x@y.z>\n",
        'Revert "feat(ds4): add sparse prefill"',
    ],
)
def test_accepts(message: str) -> None:
    assert problems(message) == []


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("Add sparse prefill", "type(scope): summary"),
        ("feature(ds4): add sparse prefill", "type(scope): summary"),
        ("feat(DS4): add sparse prefill", "lowercase scope"),
        ("feat(ds4):add sparse prefill", "type(scope): summary"),
        ("fixup! feat(ds4): add sparse prefill", "type(scope): summary"),
        ("fix: clamp the rollback window", "required lowercase scope"),
        ("ci(): empty scope", "required lowercase scope"),
        ("fix(server): clamp the rollback window.", "ends with a period"),
        ("fix(server): " + "x" * 100, "at most 100"),
        ("fix(server): clamp\nbody without a blank line", "blank line"),
    ],
)
def test_rejects(message: str, expected: str) -> None:
    assert any(expected in p for p in problems(message))
