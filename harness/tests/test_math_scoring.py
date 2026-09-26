from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
HARNESS_DIR = REPO_ROOT / "harness"
sys.path.insert(0, str(HARNESS_DIR))

import client_test_runner as ctr  # noqa: E402
import math_scoring  # noqa: E402


class MathScoringTest(unittest.TestCase):
    def test_boxed_extraction(self) -> None:
        self.assertEqual(
            math_scoring._extract_boxed(r"so the domain is \boxed{[2, 5)}"),
            "[2, 5)",
        )

    def test_fraction_equivalence(self) -> None:
        # \frac{20}{3} and \dfrac{1}{2} must equal their plain forms.
        self.assertTrue(math_scoring._math_equiv(r"\frac{20}{3}", "20/3"))
        self.assertTrue(math_scoring._math_equiv(r"\dfrac{1}{2}", "1/2"))
        self.assertTrue(math_scoring._math_equiv(r"\frac{5\sqrt{5}}{3}", r"5\sqrt{5}/3"))

    def test_interval_spacing(self) -> None:
        self.assertTrue(math_scoring._math_equiv("[2, 5)", "[2,5)"))
        self.assertTrue(math_scoring._math_equiv("[2, 5)", "[2,5)"))

    def test_numeric(self) -> None:
        self.assertTrue(math_scoring._math_equiv("24", "24"))
        self.assertTrue(math_scoring._math_equiv("$18", "18"))
        self.assertFalse(math_scoring._math_equiv("24", "25"))


class GsmScoringTest(unittest.TestCase):
    def _score(self, text: str, gold: str) -> bool:
        return ctr._score_gsm_response(text, gold)[0]

    def test_answer_marker_number(self) -> None:
        # Regression: the old extractor grabbed the intermediate "20".
        text = "Add the sheep from all three cities:\n$$20 + 80 + 160 = 260$$\n\n**Answer:** 260"
        self.assertTrue(self._score(text, "260"))

    def test_answer_marker_with_words(self) -> None:
        text = "**Answer:**\nIt takes **160 minutes** (or 2 hours and 40 minutes)."
        self.assertTrue(self._score(text, "160"))

    def test_answer_marker_miles(self) -> None:
        text = "**Answer:**\nJohn is **120 miles** from home when the 4 hours are up."
        self.assertTrue(self._score(text, "120"))

    def test_boxed_takes_priority(self) -> None:
        self.assertTrue(self._score("we get 40 then\n\\boxed{18}", "18"))

    def test_hash_marker(self) -> None:
        self.assertTrue(self._score("The total is 3\n#### 3", "3"))

    def test_wrong(self) -> None:
        self.assertFalse(self._score("**Answer:** 20", "260"))


if __name__ == "__main__":
    unittest.main()
