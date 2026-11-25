"""Card-registry unit tests — normalization, resolution, capability gate.

Pins the contract of ``lucebench.model_cards``:

  * ``normalize_model_card_stem`` corpus (the doc's required corpus:
    ``qwen/qwen3.6-27b:free``, provider prefixes, version/quant suffixes,
    aliases, unknown → its own stem).
  * ``resolve_card`` precedence: props > bundled > none.
  * ``card_is_thinking_capable`` gate: thinking_control / terminator_hint
    → capable; bare card → not.

The bundled-lookup test depends on the build-time force-include having
landed ``qwen3.6-27b.json`` under ``lucebench/_model_cards/``; when that
package data is absent (a source checkout that never built the wheel) the
lookup degrades to ``"none"`` and the bundled-hit assertion is skipped.
"""

from __future__ import annotations

import pytest

from lucebench.model_cards import (
    _bundled_cards,
    card_is_thinking_capable,
    normalize_model_card_stem,
    resolve_card,
)

# ────────────────────────────────────────────────────────────────────
# Normalization corpus — mirrors docs/client-thinking-budget.md
# ────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # OpenRouter route: provider prefix + :free suffix.
        ("qwen/qwen3.6-27b:free", "qwen3.6-27b"),
        # HF org/repo with mixed case.
        ("Qwen/Qwen3.6-27B", "qwen3.6-27b"),
        # Bare HF id.
        ("qwen3.6-27b", "qwen3.6-27b"),
        # Display-name shape: spaces → '-', lowercased.
        ("Qwen3.6 27B", "qwen3.6-27b"),
        # Underscores and tabs → '-'.
        ("Qwen3.6_27B", "qwen3.6-27b"),
        # Other route/quant suffixes after the colon.
        ("qwen/qwen3.6-27b:nitro", "qwen3.6-27b"),
        ("qwen3.6-27b:Q4_K_M", "qwen3.6-27b"),
        # Punctuation that isn't [a-z0-9.-] is dropped.
        ("Gemma-4 (26B)!", "gemma-4-26b"),
        # Unknown model keeps its own normalized stem (no card guess).
        ("mystery-model-7b", "mystery-model-7b"),
        ("Unknown_Model 13B", "unknown-model-13b"),
        # Deep provider path keeps only the final segment.
        ("openrouter/qwen/qwen3.6-27b:free", "qwen3.6-27b"),
        # Non-string / empty.
        ("", ""),
    ],
)
def test_normalize_corpus(raw: str, expected: str) -> None:
    assert normalize_model_card_stem(raw) == expected


def test_normalize_non_string() -> None:
    assert normalize_model_card_stem(None) == ""  # type: ignore[arg-type]


# ────────────────────────────────────────────────────────────────────
# resolve_card precedence: props > bundled > none
# ────────────────────────────────────────────────────────────────────


def _props_card() -> dict:
    return {"name": "From Props", "thinking_terminator_hint": "</think>\n\n"}


def test_resolve_props_beats_bundled() -> None:
    """A /props card wins even when the id also matches a bundled stem."""
    card, source = resolve_card("qwen/qwen3.6-27b:free", _props_card())
    assert source == "props"
    assert card == _props_card()


def test_resolve_bundled_when_no_props() -> None:
    """No props card → bundled registry lookup by normalized stem."""
    card, source = resolve_card("qwen/qwen3.6-27b:free", None)
    if "qwen3.6-27b" in _bundled_cards():
        assert source == "bundled"
        assert isinstance(card, dict)
        # Canonical qwen card carries thinking_control.
        assert "thinking_control" in card
    else:
        # No packaged cards in this checkout — degrades to none.
        assert source == "none"
        assert card is None


def test_resolve_none_for_unknown() -> None:
    card, source = resolve_card("some-unknown-model-99b", None)
    assert source == "none"
    assert card is None


def test_resolve_empty_props_dict_falls_through() -> None:
    """An empty {} props card is not authoritative — fall to bundled/none."""
    card, source = resolve_card("some-unknown-model-99b", {})
    assert source == "none"
    assert card is None


# ────────────────────────────────────────────────────────────────────
# Capability gate
# ────────────────────────────────────────────────────────────────────


def test_capable_with_thinking_control() -> None:
    assert card_is_thinking_capable(
        {"thinking_control": {"think_prompt_token": "/think"}}
    )


def test_capable_with_terminator_hint() -> None:
    assert card_is_thinking_capable({"thinking_terminator_hint": "</think>\n\n"})


def test_not_capable_bare_card() -> None:
    assert not card_is_thinking_capable({"name": "Plain Model", "max_tokens": 4096})


def test_not_capable_none() -> None:
    assert not card_is_thinking_capable(None)


def test_not_capable_empty_thinking_control() -> None:
    assert not card_is_thinking_capable({"thinking_control": {}})
    assert not card_is_thinking_capable({"thinking_terminator_hint": ""})
