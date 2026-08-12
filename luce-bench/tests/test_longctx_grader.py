"""Tests for the longctx area's Risk-prefix grader.

The 2026-05-30 qwen3.6-27b thinking-enabled benchmark exposed the
strict-prefix grader as too brittle: thinking models that emit a
short transition phrase before the required ``Risk:`` line were
graded as failures even though the response contained the required
prefix on a later line. The lenient variant accepts that pattern;
the strict metric is preserved alongside for regression detection.
"""

from __future__ import annotations

from lucebench.areas.longctx import grade_longctx, grade_longctx_case


def _row(content: str) -> dict[str, str]:
    return {"content": content}


def test_pure_prefix_passes_strict_and_lenient() -> None:
    """The instruction-compliant case: response opens with ``Risk:``."""
    g = grade_longctx("", "Risk: The server's tokenization layer leaks under load.")
    assert g["graded_pass"] is True
    assert g["strict_pass"] is True
    assert g["format_pass"] is True


def test_leading_whitespace_is_ok_in_strict() -> None:
    """``Risk:`` with leading whitespace still counts as strict-prefix."""
    g = grade_longctx("", "\n  Risk: details here.")
    assert g["strict_pass"] is True
    assert g["graded_pass"] is True


def test_thinking_preamble_passes_lenient_but_not_strict() -> None:
    """Regression for the qwen3.6 thinking-mode failure: model emits a
    transition phrase, then the required ``Risk:`` line. The lenient
    grader must accept this while strict_pass records the imperfection."""
    content = (
        "Considering the limited time by the user, I have to give the "
        "solution based on the thinking directly now.\n\n\n\n"
        "Risk: The server's reliance on parsing XML tool calls and "
        "emitting them in OpenAI format during SSE streaming creates a "
        "fragile compatibility layer."
    )
    g = grade_longctx("", content)
    assert g["graded_pass"] is True, "lenient grader must accept preamble + Risk: line"
    assert g["strict_pass"] is False, "strict grader must still note the preamble"
    assert g["format_pass"] is True


def test_no_risk_anywhere_fails() -> None:
    """Response mentions risk casually but never as a line-prefix → fail."""
    g = grade_longctx(
        "",
        "The server has some risks but I'd rather discuss the architecture overall.",
    )
    assert g["graded_pass"] is False
    assert g["strict_pass"] is False


def test_too_short_fails_even_with_risk() -> None:
    """An 8-char minimum gates trivial replies."""
    g = grade_longctx("", "Risk:")
    # "Risk:" is too short — fails the nonempty bar.
    assert g["graded_pass"] is False


def test_grade_longctx_case_surfaces_strict_pass() -> None:
    """The case-level wrapper exposes strict_pass for snapshot inspection."""
    content = "Considering ...\n\nRisk: a real risk."
    out = grade_longctx_case({"prompt": ""}, _row(content))
    assert "strict_pass" in out
    assert out["strict_pass"] is False
    assert out["pass"] is True


def test_grade_longctx_case_strict_pass_when_no_preamble() -> None:
    """When the model complies strictly, strict_pass and pass both True."""
    out = grade_longctx_case(
        {"prompt": ""},
        _row("Risk: tokenization layer leaks under load."),
    )
    assert out["strict_pass"] is True
    assert out["pass"] is True
