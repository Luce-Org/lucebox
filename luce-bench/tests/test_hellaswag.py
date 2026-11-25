"""Unit tests for the `hellaswag` area.

No live server. The fixture has 100 vendored cases (deterministic
``Random(42)`` sample of the upstream validation split). Every
HellaSwag row has exactly 4 endings labelled A/B/C/D, so the schema
and grader checks are tighter than TruthfulQA's variable-N MC.
"""

from __future__ import annotations

from lucebench.areas import hellaswag
from lucebench.cli import AREAS


def test_hellaswag_cases_load():
    cases = hellaswag.load_hellaswag_cases()
    assert len(cases) == 100
    for c in cases:
        assert c["area"] == "hellaswag"
        assert c["kind"] == "multiple-choice"
        assert c["id"].startswith("hellaswag-val-")
        # HellaSwag is always exactly 4 endings.
        assert len(c["choices"]) == 4
        assert c["expected"] in {"A", "B", "C", "D"}
        # Prompt must use "Context:" prefix (not "Question:") and
        # render all four labels.
        assert "Context:" in c["prompt"]
        for letter in "ABCD":
            assert f"{letter}." in c["prompt"]
        assert c["ctx"]  # the sentence prefix carries over
    assert len({c["id"] for c in cases}) == len(cases)


def test_hellaswag_area_registered():
    assert "hellaswag" in AREAS
    cfg = AREAS["hellaswag"]
    assert cfg["default_max_tokens"] == 128
    assert cfg["default_thinking"] is False
    assert callable(cfg["load"])
    assert callable(cfg["grade"])


def test_grade_hellaswag_letter_only_pass():
    case = {"id": "x", "expected": "C", "choices": ["a", "b", "c", "d"]}
    g = hellaswag.grade_hellaswag_case(case, {"content": "C"})
    assert g["pass"] is True
    assert g["status"] == "passed"


def test_grade_hellaswag_canonical_phrasing_pass():
    case = {"id": "x", "expected": "D", "choices": ["a", "b", "c", "d"]}
    g = hellaswag.grade_hellaswag_case(case, {"content": "The answer is D."})
    assert g["pass"] is True
    assert g["given"] == "D"


def test_grade_hellaswag_wrong_letter_fail():
    case = {"id": "x", "expected": "B", "choices": ["a", "b", "c", "d"]}
    g = hellaswag.grade_hellaswag_case(case, {"content": "A"})
    assert g["pass"] is False
    assert g["status"] == "failed"
    assert g["given"] == "A"


def test_grade_hellaswag_no_letter_format_error():
    case = {"id": "x", "expected": "A", "choices": ["a", "b", "c", "d"]}
    g = hellaswag.grade_hellaswag_case(case, {"content": ""})
    assert g["pass"] is False
    assert g["status"] == "format_error"


def test_hellaswag_prompt_includes_ctx():
    """A randomly-picked case must thread its ``ctx`` into the rendered prompt."""
    case = hellaswag.load_hellaswag_cases()[0]
    assert case["ctx"] in case["prompt"]
    for ending in case["choices"]:
        assert ending in case["prompt"]
