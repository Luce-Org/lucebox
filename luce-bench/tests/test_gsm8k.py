"""Unit tests for the `gsm8k` area — fixture loader + numeric grader.

No live server. The fixture has 100 vendored cases (deterministic
``Random(42)`` sample of the upstream test split); we lock in the
schema, a positive grade via the canonical ``#### N`` marker, and the
fallback "last number wins" path. Also covers float-tolerance + the
format_error vs failed split.
"""

from __future__ import annotations

from lucebench.areas import gsm8k
from lucebench.cli import AREAS


def test_gsm8k_cases_load():
    cases = gsm8k.load_gsm8k_cases()
    assert len(cases) == 100, "fixture should have exactly the 100-case Random(42) sample"
    for c in cases:
        assert c["area"] == "gsm8k"
        assert c["source"] == "gsm8k"
        assert c["kind"] == "math-word-problem"
        assert c["id"].startswith("gsm8k-test-")
        assert c["prompt"]
        # Expected must parse as a real number — upstream guarantees this.
        float(c["expected"])
    # IDs must be unique so --case-id can address them deterministically.
    assert len({c["id"] for c in cases}) == len(cases)


def test_gsm8k_area_registered():
    """The CLI's AREAS dict must include gsm8k with the right defaults."""
    assert "gsm8k" in AREAS
    cfg = AREAS["gsm8k"]
    assert cfg["default_max_tokens"] == 2048
    assert cfg["default_thinking"] is False
    assert callable(cfg["load"])
    assert callable(cfg["grade"])


def test_grade_gsm8k_marker_pass():
    """Canonical `#### N` marker → strict pass."""
    case = {"id": "x", "expected": "18", "answer": "18"}
    row = {"content": "Let me work this out... 5 * 2 = 10, then +2 = 12, then 12 / (2/3) = 18.\n#### 18"}
    g = gsm8k.grade_gsm8k_case(case, row)
    assert g["pass"] is True
    assert g["status"] == "passed"
    assert g["given"] == "18"
    assert g["correct"] == "18"
    assert g["format_pass"] is True


def test_grade_gsm8k_marker_with_commas():
    """Comma-separated thousands in the marker must normalise."""
    case = {"id": "x", "expected": "3500", "answer": "3500"}
    row = {"content": "Answer time.\n#### 3,500"}
    g = gsm8k.grade_gsm8k_case(case, row)
    assert g["pass"] is True


def test_grade_gsm8k_no_marker_last_number_fallback():
    """No `####` marker → take the last number in the reply."""
    case = {"id": "x", "expected": "42", "answer": "42"}
    row = {"content": "The cost is 7 dollars per item and there are 6 items so the answer is 42"}
    g = gsm8k.grade_gsm8k_case(case, row)
    assert g["pass"] is True
    assert g["given"] == "42"


def test_grade_gsm8k_float_tolerance():
    """18 vs 18.0 vs 18.00 must all compare equal under the float-tolerance rule."""
    case = {"id": "x", "expected": "18", "answer": "18"}
    for content in ("#### 18", "#### 18.0", "#### 18.00", "The answer is 18.000"):
        g = gsm8k.grade_gsm8k_case(case, {"content": content})
        assert g["pass"] is True, f"expected pass for {content!r}"


def test_grade_gsm8k_wrong_number():
    """Wrong numeric answer → failed (NOT format_error)."""
    case = {"id": "x", "expected": "18", "answer": "18"}
    row = {"content": "#### 17"}
    g = gsm8k.grade_gsm8k_case(case, row)
    assert g["pass"] is False
    assert g["status"] == "failed"
    assert g["format_pass"] is True
    assert g["given"] == "17"


def test_grade_gsm8k_no_number_format_error():
    """Reply with no number anywhere → format_error."""
    case = {"id": "x", "expected": "18", "answer": "18"}
    for content in ("", "I don't know", "let me think about that"):
        g = gsm8k.grade_gsm8k_case(case, {"content": content})
        assert g["pass"] is False
        assert g["status"] == "format_error"
        assert g["format_pass"] is False


def test_grade_gsm8k_marker_beats_later_text():
    """If the model emits `#### N` then keeps talking, the marker wins."""
    case = {"id": "x", "expected": "18", "answer": "18"}
    row = {"content": "Working: 12 / (2/3) = 18.\n#### 18\n\nThough wait, maybe it could also be 19 or 20."}
    g = gsm8k.grade_gsm8k_case(case, row)
    assert g["pass"] is True
    assert g["given"] == "18"


def test_grade_gsm8k_reasoning_content_fallback():
    """Empty content + answer in reasoning_content → still graded."""
    case = {"id": "x", "expected": "18", "answer": "18"}
    row = {"content": "", "reasoning_content": "Thinking... the answer is #### 18"}
    g = gsm8k.grade_gsm8k_case(case, row)
    assert g["pass"] is True


def test_extract_gsm8k_answer_helpers():
    """Direct unit tests of the extractor — both code paths."""
    assert gsm8k.extract_gsm8k_answer("foo bar #### 42") == "42"
    assert gsm8k.extract_gsm8k_answer("#### -3,500") == "-3500"
    assert gsm8k.extract_gsm8k_answer("#### 0.5") == "0.5"
    # Fallback path
    assert gsm8k.extract_gsm8k_answer("first 1 then 2 then 3") == "3"
    assert gsm8k.extract_gsm8k_answer("answer is 1,234,567") == "1234567"
    # Nothing to extract
    assert gsm8k.extract_gsm8k_answer("") is None
    assert gsm8k.extract_gsm8k_answer("nothing numeric here") is None
