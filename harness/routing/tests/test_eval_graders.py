"""Deterministic graders: what counts as a correct answer for each prompt type."""

import pytest
from eval.graders import grade_deterministic

NUMERIC = {"type": "numeric", "answer": 1234.0, "tolerance": 1e-4}
MATH_FRAC = {"type": "math", "answer": "\\frac{1}{2}"}
MATH_TUPLE = {"type": "math", "answer": "\\left( 3, \\frac{\\pi}{2} \\right)"}
CHOICE = {"type": "choice", "answer": "C"}
LABEL = {"type": "label", "answer": "positive", "labels": ["positive", "negative", "neutral"]}
JSON = {"type": "json_keys", "keys": ["name", "year"], "values": {"year": 1999}}


@pytest.mark.parametrize("text,grader,expected", [
    # numeric: boxed wins over later numbers; thousands separators; last-number fallback
    ("so the total is \\boxed{1,234} (checked 3 times)", NUMERIC, True),
    ("Step 1: 12 * 100 = 1200. Step 2: 1200 + 34 = 1234", NUMERIC, True),
    ("The answer is $1234.00.", NUMERIC, True),
    ("\\boxed{1243}", NUMERIC, False),
    # reasoning is ignored: only the post-</think> answer is graded
    ("<think>maybe \\boxed{1234}</think>The answer is \\boxed{99}", NUMERIC, False),
    ("<think>hmm 99</think>\\boxed{1234}", NUMERIC, True),
    # math: LaTeX-equivalent forms
    ("\\boxed{\\dfrac12}", MATH_FRAC, True),
    ("\\boxed{0.5}", MATH_FRAC, True),
    ("\\boxed{\\frac{1}{3}}", MATH_FRAC, False),
    ("thus \\boxed{(3, \\frac{\\pi}{2})}", MATH_TUPLE, True),
    ("\\boxed{(3, \\pi)}", MATH_TUPLE, False),
    # choice letter extraction
    ("Reasoning...\nThe answer is (C).", CHOICE, True),
    ("Answer: **C**", CHOICE, True),
    ("The answer is (B)", CHOICE, False),
    ("I cannot decide.", CHOICE, False),
    # label: exactly one label, the right one
    ("Positive", LABEL, True),
    ("positive or negative, hard to say", LABEL, False),
    ("negative", LABEL, False),
    # json_keys: fenced or bare JSON, required keys and pinned values
    ('```json\n{"name": "X", "year": 1999}\n```', JSON, True),
    ('Sure! {"name": "X", "year": "1999"} hope that helps', JSON, True),
    ('{"name": "X"}', JSON, False),
    ('{"name": "X", "year": 2001}', JSON, False),
    ("no json here", JSON, False),
])
def test_deterministic_graders(text, grader, expected):
    assert grade_deterministic(text, grader).correct is expected


HUMANEVAL_STYLE = {
    "type": "python_tests",
    "entry_point": "add",
    "prompt_prefix": "def add(a, b):\n    \"\"\"Add two numbers.\"\"\"\n",
    "tests": "def check(candidate):\n    assert candidate(2, 3) == 5\n    assert candidate(-1, 1) == 0\n",
}
MBPP_STYLE = {"type": "python_tests", "entry_point": None,
              "tests": "assert square(3) == 9\nassert square(0) == 0"}


@pytest.mark.parametrize("text,grader,expected", [
    ("```python\ndef add(a, b):\n    return a + b\n```", HUMANEVAL_STYLE, True),
    ("```python\ndef add(a, b):\n    return a - b\n```", HUMANEVAL_STYLE, False),
    # body-only answer is spliced under the prompt's signature
    ("```python\n    return a + b\n```", HUMANEVAL_STYLE, True),
    # the longest fenced block is the solution, the usage example is ignored
    ("```python\nsquare(2)\n```\n```python\ndef square(x):\n    return x * x\n```", MBPP_STYLE, True),
    # truncated (unterminated) fence still runs
    ("```python\ndef square(x):\n    return x * x\n", MBPP_STYLE, True),
    ("```python\ndef square(x):\n    return x + x\n```", MBPP_STYLE, False),
    ("```python\ndef square(x) return\n```", MBPP_STYLE, False),
])
def test_python_tests_grader(text, grader, expected):
    assert grade_deterministic(text, grader).correct is expected


def test_python_tests_timeout_and_no_network():
    hang = {**MBPP_STYLE, "timeout": 1}
    r = grade_deterministic("```python\ndef square(x):\n    while True: pass\n```", hang)
    assert (r.correct, r.detail["stderr_tail"]) == (False, "timeout")

    net = {"type": "python_tests", "entry_point": None,
           "tests": "import socket\nsocket.create_connection(('127.0.0.1', 9), timeout=1)"}
    r = grade_deterministic("```python\npass\n```", net)
    assert not r.correct and "network disabled" in r.detail["stderr_tail"]
