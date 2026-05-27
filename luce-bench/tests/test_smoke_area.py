"""Unit tests for the `smoke` area — case loader + lenient grader.

No live server. We feed canned ``row["content"]`` strings through
``grade_smoke_case`` to lock in the substring-match + case-folding
semantics. Also covers the preflight helper's failure path against a
deliberately closed port so we don't regress the "no more 92 timeouts"
fix.
"""

from __future__ import annotations

from lucebench.areas import smoke
from lucebench.cli import AREAS, _preflight


def test_smoke_cases_load():
    cases = smoke.load_smoke_cases()
    assert len(cases) == 3
    ids = [c["id"] for c in cases]
    assert ids == ["smoke-arithmetic", "smoke-capital", "smoke-sequence"]
    for c in cases:
        assert c["area"] == "smoke"
        assert c["kind"] == "smoke"
        assert c["prompt"]
        assert c["expected"]


def test_smoke_area_registered():
    """The CLI's AREAS dict must include smoke with the right defaults."""
    assert "smoke" in AREAS
    cfg = AREAS["smoke"]
    assert cfg["default_max_tokens"] == 4096
    assert cfg["default_thinking"] is False
    # Grader and loader are wired up.
    assert callable(cfg["load"])
    assert callable(cfg["grade"])


def test_grade_smoke_arithmetic_pass():
    case = {"id": "smoke-arithmetic", "expected": "4", "prompt": "..."}
    row = {"content": "4"}
    g = smoke.grade_smoke_case(case, row)
    assert g["pass"] is True
    assert g["status"] == "passed"
    assert g["correct"] == "4"


def test_grade_smoke_arithmetic_pass_embedded():
    """Substring match — the model padded its reply but '4' is in there."""
    case = {"id": "smoke-arithmetic", "expected": "4", "prompt": "..."}
    row = {"content": "The answer is 4."}
    g = smoke.grade_smoke_case(case, row)
    assert g["pass"] is True


def test_grade_smoke_capital_case_insensitive():
    """The capital-of-France case folds — 'Paris', 'paris', 'PARIS' all pass."""
    case = {"id": "smoke-capital", "expected": "Paris", "prompt": "..."}
    for content in ("Paris", "paris", "Paris.", "The capital is Paris."):
        g = smoke.grade_smoke_case(case, {"content": content})
        assert g["pass"] is True, f"expected pass for {content!r}"


def test_grade_smoke_matches_reasoning_content():
    """Some servers route the answer to reasoning_content when max_tokens
    trips mid-think; we accept matches in either field."""
    case = {"id": "smoke-arithmetic", "expected": "4", "prompt": "..."}
    row = {"content": "", "reasoning_content": "Let me think... 2+2 = 4"}
    g = smoke.grade_smoke_case(case, row)
    assert g["pass"] is True
    assert g["format_pass"] is True


def test_grade_smoke_sequence_pass():
    case = {"id": "smoke-sequence", "expected": "3", "prompt": "..."}
    row = {"content": "3"}
    g = smoke.grade_smoke_case(case, row)
    assert g["pass"] is True


def test_grade_smoke_fail_wrong_answer():
    case = {"id": "smoke-arithmetic", "expected": "4", "prompt": "..."}
    row = {"content": "five"}
    g = smoke.grade_smoke_case(case, row)
    assert g["pass"] is False
    assert g["status"] == "failed"


def test_grade_smoke_fail_empty_content():
    """Empty / None content is a format_error, not a regular fail."""
    case = {"id": "smoke-arithmetic", "expected": "4", "prompt": "..."}
    for content in ("", None):
        g = smoke.grade_smoke_case(case, {"content": content})
        assert g["pass"] is False
        assert g["status"] == "format_error"
        assert g["format_pass"] is False


def test_preflight_unreachable_fails_fast():
    """Preflight against a closed port returns ok=False quickly."""
    # Port 1 is reserved / closed in practice. Use a tiny timeout so
    # the test stays sub-second even on slow CI.
    ok, lines = _preflight("http://127.0.0.1:1", timeout_s=2)
    assert ok is False
    # First line is the header, second is the liveness check — must be ✗.
    assert lines[0].startswith("[lucebench] preflight")
    assert "✗" in lines[1] or "FAIL" in lines[1]
    assert "liveness" in lines[1]
