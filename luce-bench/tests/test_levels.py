"""Tests for ``lucebench.levels`` — the curated level → area mapping."""

from __future__ import annotations

import pytest

from lucebench.levels import LEVELS, resolve_level


def test_level0_is_smoke_only() -> None:
    assert LEVELS["level0"] == [("smoke", None)]


def test_level1_has_all_caps_set() -> None:
    """level1 is the fast tier — every entry has an n_cap."""
    level1 = LEVELS["level1"]
    assert ("smoke", None) in level1
    assert ("code", 5) in level1
    assert ("gsm8k", 5) in level1
    assert ("agent", 2) in level1
    assert ("longctx", 1) in level1
    # No surprise extras.
    assert {a for a, _ in level1} == {"smoke", "code", "gsm8k", "agent", "longctx"}


def test_level2_has_no_caps() -> None:
    """level2 runs every area in full (n_cap=None on each)."""
    level2 = LEVELS["level2"]
    assert all(cap is None for _, cap in level2), f"level2 had caps: {level2}"
    names = {a for a, _ in level2}
    assert names == {
        "smoke",
        "code",
        "gsm8k",
        "truthfulqa-mc1",
        "hellaswag",
        "agent",
        "longctx",
    }


def test_level3_is_superset_of_level2() -> None:
    """Each lower tier is a strict subset of the next (names only)."""
    l2 = {a for a, _ in LEVELS["level2"]}
    l3 = {a for a, _ in LEVELS["level3"]}
    assert l2.issubset(l3)
    # Extras only in level3:
    assert {"ds4-eval", "agent_recorded", "forge"} <= l3


def test_resolve_level_returns_copy() -> None:
    """``resolve_level`` returns a fresh list — mutation can't poison the registry."""
    result = resolve_level("level1")
    result.append(("ds4-eval", None))
    assert resolve_level("level1") != result
    assert ("ds4-eval", None) not in resolve_level("level1")


def test_resolve_level_unknown_raises() -> None:
    with pytest.raises(ValueError, match="unknown level"):
        resolve_level("level9")


def test_resolve_level_known_keys() -> None:
    for k in ("level0", "level1", "level2", "level3"):
        assert resolve_level(k) == LEVELS[k]
