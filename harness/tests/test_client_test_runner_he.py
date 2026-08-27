#!/usr/bin/env python3

import importlib.util
import sys
import unittest
from pathlib import Path


HARNESS_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HARNESS_DIR))
SPEC = importlib.util.spec_from_file_location(
    "client_test_runner", HARNESS_DIR / "client_test_runner.py")
assert SPEC and SPEC.loader
RUNNER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = RUNNER
SPEC.loader.exec_module(RUNNER)


PROMPT = """Complete the following Python function.

def add_one(value: int) -> int:
    \"\"\"Return the value plus one.\"\"\"
"""
GOLD_TEST = """
def check(candidate):
    assert candidate(1) == 2
    assert candidate(-1) == 0
"""


class HumanEvalScoringTests(unittest.TestCase):
    def test_scores_indented_fenced_function_body(self) -> None:
        correct, detail = RUNNER._score_he_response(
            "```python\n    return value + 1\n```",
            "add_one", GOLD_TEST, PROMPT)
        self.assertTrue(correct, detail)

    def test_scores_bare_indented_function_body(self) -> None:
        correct, detail = RUNNER._score_he_response(
            "    return value + 1", "add_one", GOLD_TEST, PROMPT)
        self.assertTrue(correct, detail)

    def test_scores_complete_function(self) -> None:
        correct, detail = RUNNER._score_he_response(
            "def add_one(value: int) -> int:\n    return value + 1",
            "add_one", GOLD_TEST, PROMPT)
        self.assertTrue(correct, detail)

    def test_rejects_incorrect_body(self) -> None:
        correct, _ = RUNNER._score_he_response(
            "    return value - 1", "add_one", GOLD_TEST, PROMPT)
        self.assertFalse(correct)


if __name__ == "__main__":
    unittest.main()
