#!/usr/bin/env python3
"""Regression tests for shared math scoring and answer equivalence."""

from __future__ import annotations

import unittest

from harness.math_scoring import _extract_boxed, _math_equiv, _normalize_math


class MathEquivTests(unittest.TestCase):
    def test_equivalent_fractions(self) -> None:
        self.assertTrue(_math_equiv(r"\frac{1}{2}", r"\frac{2}{4}"))
        self.assertTrue(_math_equiv(r"\frac{6}{8}", r"\frac{3}{4}"))
        self.assertFalse(_math_equiv(r"\frac{1}{2}", r"\frac{1}{3}"))

    def test_fraction_vs_decimal(self) -> None:
        self.assertTrue(_math_equiv(r"\frac{1}{2}", "0.5"))
        self.assertTrue(_math_equiv("0.5", r"\frac{1}{2}"))
        self.assertTrue(_math_equiv(r"\frac{4}{2}", "2"))
        self.assertTrue(_math_equiv(r"\frac{4}{2}", "2.0"))
        self.assertTrue(_math_equiv("3", "3.0"))

    def test_positive_and_negative_fractions(self) -> None:
        self.assertTrue(_math_equiv(r"-\frac{1}{2}", "-0.5"))
        self.assertTrue(_math_equiv("-0.5", r"-\frac{1}{2}"))
        self.assertTrue(_math_equiv(r"-\frac{1}{2}", r"-\frac{2}{4}"))
        self.assertFalse(_math_equiv(r"\frac{1}{2}", r"-\frac{1}{2}"))

    def test_negative_numerators_and_denominators(self) -> None:
        self.assertTrue(_math_equiv(r"\frac{-1}{2}", "-0.5"))
        self.assertTrue(_math_equiv(r"\frac{1}{-2}", "-0.5"))
        self.assertTrue(_math_equiv(r"\frac{-1}{-2}", "0.5"))
        self.assertTrue(_math_equiv(r"\frac{-1}{2}", r"\frac{1}{-2}"))

    def test_mixed_fractions(self) -> None:
        self.assertTrue(_math_equiv(r"1\frac{1}{2}", r"1\frac{2}{4}"))
        self.assertTrue(_math_equiv(r"1\frac{1}{2}", "1.5"))
        self.assertTrue(_math_equiv("1.5", r"1\frac{1}{2}"))

    def test_improper_vs_mixed_fractions(self) -> None:
        self.assertTrue(_math_equiv(r"\frac{3}{2}", r"1\frac{1}{2}"))
        self.assertTrue(_math_equiv(r"1\frac{1}{2}", r"\frac{3}{2}"))

    def test_negative_mixed_fractions(self) -> None:
        self.assertTrue(_math_equiv(r"-1\frac{1}{2}", "-1.5"))
        self.assertTrue(_math_equiv(r"-1\frac{1}{2}", r"-1\frac{2}{4}"))
        self.assertTrue(_math_equiv(r"-1\frac{1}{2}", r"-\frac{3}{2}"))

    def test_commas_in_numeric_and_fraction_values(self) -> None:
        self.assertTrue(_math_equiv("1,000", "1000"))
        self.assertTrue(_math_equiv(r"\frac{1,000}{2}", "500"))
        self.assertTrue(_math_equiv("1,000.5", "1000.5"))

    def test_whitespace_variations(self) -> None:
        self.assertTrue(_math_equiv(r"\frac { 1 } { 2 }", "0.5"))
        self.assertTrue(_math_equiv(r"\frac{1}{2 }", r"\frac{1}{2}"))
        self.assertTrue(_math_equiv(r"  \frac{1}{2}  ", "0.5"))

    def test_zero_denominators(self) -> None:
        self.assertFalse(_math_equiv(r"\frac{1}{0}", "1"))
        self.assertFalse(_math_equiv(r"1\frac{1}{0}", "1.5"))
        self.assertTrue(_math_equiv(r"\frac{1}{0}", r"\frac{1}{0}"))

    def test_malformed_fractions(self) -> None:
        self.assertFalse(_math_equiv(r"\frac{1}", "1"))
        self.assertFalse(_math_equiv(r"\frac{}{}", "0"))

    def test_floating_point_tolerance(self) -> None:
        self.assertTrue(_math_equiv("1.0000001", "1.0000002"))
        self.assertFalse(_math_equiv("1.000001", "1.000003"))

    def test_none_and_empty_inputs(self) -> None:
        self.assertFalse(_math_equiv(None, "1"))
        self.assertFalse(_math_equiv("1", None))
        self.assertFalse(_math_equiv(None, None))
        self.assertTrue(_math_equiv("", ""))
        self.assertTrue(_math_equiv("   ", ""))
        self.assertFalse(_math_equiv("", "0"))

    def test_symbolic_expressions(self) -> None:
        self.assertTrue(_math_equiv("x + 1", "x + 1"))
        self.assertTrue(_math_equiv(r"\sqrt{2}", r"\sqrt{2}"))
        self.assertFalse(_math_equiv("x + 1", "1 + x"))
        self.assertFalse(_math_equiv(r"\sqrt{4}", "2"))
        self.assertTrue(_math_equiv(r"\frac{a}{b}", r"\frac{a}{b}"))
        self.assertFalse(_math_equiv(r"\frac{a}{b}", r"\frac{c}{d}"))

    def test_extract_boxed_and_normalize(self) -> None:
        self.assertEqual(_extract_boxed(r"Final answer: \boxed{42}"), "42")
        self.assertEqual(_extract_boxed(r"\boxed{\frac{1}{2}}"), r"\frac{1}{2}")
        self.assertEqual(_extract_boxed(r"Step \boxed{1} then \boxed{2}"), "2")
        self.assertIsNone(_extract_boxed("No boxed here"))
        self.assertEqual(_normalize_math("$18$"), "18")
        self.assertEqual(_normalize_math(r"\text{cm}"), "cm")
        self.assertEqual(_normalize_math(r"\tfrac{1}{2}"), r"\frac{1}{2}")
        self.assertEqual(_normalize_math("50 dollars"), "50")


if __name__ == "__main__":
    unittest.main()
