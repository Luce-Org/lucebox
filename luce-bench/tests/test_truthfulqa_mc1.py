"""Unit tests for the `truthfulqa-mc1` area + the shared MC helper.

No live server. The fixture has 100 vendored cases (deterministic
``Random(42)`` sample of the upstream validation split). We lock in
the schema, the prompt-rendering (dynamic letter range, 2–13 choices),
and grader behavior on both the "canonical phrasing" and
"last-letter fallback" paths.
"""

from __future__ import annotations

from lucebench.areas import truthfulqa_mc1
from lucebench.areas._mc import (
    build_mc_prompt,
    extract_mc_answer,
    grade_mc_case,
)
from lucebench.cli import AREAS


def test_truthfulqa_mc1_cases_load():
    cases = truthfulqa_mc1.load_truthfulqa_mc1_cases()
    assert len(cases) == 100, "fixture should have exactly the 100-case Random(42) sample"
    for c in cases:
        assert c["area"] == "truthfulqa-mc1"
        assert c["kind"] == "multiple-choice"
        assert c["id"].startswith("truthfulqa-mc1-val-")
        assert c["prompt"]
        assert c["choices"], "every case must carry its choice list"
        # Expected must be a single uppercase letter within the choice range.
        assert len(c["expected"]) == 1
        assert "A" <= c["expected"] <= chr(ord("A") + len(c["choices"]) - 1)
        # Prompt must include the expected letter scaffold + the actual choices.
        assert "A." in c["prompt"]
        assert "Reply with ONLY the letter" in c["prompt"]
    assert len({c["id"] for c in cases}) == len(cases)


def test_truthfulqa_mc1_area_registered():
    assert "truthfulqa-mc1" in AREAS
    cfg = AREAS["truthfulqa-mc1"]
    assert cfg["default_max_tokens"] == 256
    assert cfg["default_thinking"] is False
    assert callable(cfg["load"])
    assert callable(cfg["grade"])


def test_truthfulqa_mc1_prompt_dynamic_letter_range():
    """Prompt must render exactly one label per choice — no extras."""
    prompt = build_mc_prompt("test?", ["alpha", "beta"])
    assert "A. alpha" in prompt
    assert "B. beta" in prompt
    assert "C." not in prompt  # only 2 choices → no C label
    # 5-choice case
    prompt5 = build_mc_prompt("test?", ["a", "b", "c", "d", "e"])
    for letter in "ABCDE":
        assert f"{letter}." in prompt5
    assert "F." not in prompt5


def test_grade_truthfulqa_mc1_letter_only_pass():
    case = {"id": "x", "expected": "B", "choices": ["wrong", "right", "wrong"]}
    g = truthfulqa_mc1.grade_truthfulqa_mc1_case(case, {"content": "B"})
    assert g["pass"] is True
    assert g["status"] == "passed"
    assert g["given"] == "B"


def test_grade_truthfulqa_mc1_canonical_phrasing_pass():
    case = {"id": "x", "expected": "C", "choices": ["a", "b", "c", "d"]}
    for content in (
        "Answer: C",
        "The answer is C.",
        "the answer is (C)",
        "Final answer: C",
    ):
        g = grade_mc_case(case, {"content": content})
        assert g["pass"] is True, f"expected pass for {content!r}"


def test_grade_truthfulqa_mc1_wrong_letter_fail():
    case = {"id": "x", "expected": "B", "choices": ["a", "b", "c"]}
    g = grade_mc_case(case, {"content": "A"})
    assert g["pass"] is False
    assert g["status"] == "failed"
    assert g["given"] == "A"


def test_grade_truthfulqa_mc1_out_of_range_letter_ignored():
    """Model emits 'F' but the case only has 3 choices (A/B/C) — must not match."""
    case = {"id": "x", "expected": "C", "choices": ["a", "b", "c"]}
    g = grade_mc_case(case, {"content": "F"})
    # F isn't in range — grader returns nothing matched.
    assert g["pass"] is False
    assert g["status"] == "format_error"


def test_grade_truthfulqa_mc1_no_letter_format_error():
    case = {"id": "x", "expected": "A", "choices": ["a", "b"]}
    g = grade_mc_case(case, {"content": ""})
    assert g["pass"] is False
    assert g["status"] == "format_error"
    assert g["format_pass"] is False


def test_extract_mc_answer_strips_think_block():
    """Letters inside `<think>...</think>` must NOT influence the answer."""
    # Model thinks 'B' but answers 'C' — should grade as C.
    text = "<think>I'm leaning B but actually...</think>\nAnswer: C"
    assert extract_mc_answer(text, nchoices=4) == "C"


def test_extract_mc_answer_last_letter_fallback():
    """No canonical phrasing → last in-range letter wins."""
    text = "I think it's A, but on reflection B makes more sense."
    assert extract_mc_answer(text, nchoices=4) == "B"


def test_extract_mc_answer_respects_range():
    text = "The answer is Z."
    # nchoices=3 means only A..C valid — Z is out of range.
    assert extract_mc_answer(text, nchoices=3) is None
