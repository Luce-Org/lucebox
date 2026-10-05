#!/usr/bin/env python3
"""Unit coverage for helpers used by the parallel-serving integration test."""

import re
import unittest

from test_server_parallel import make_math_prompts


class MathPromptTest(unittest.TestCase):
    def test_64_stream_answers_are_unique_and_disjoint_from_operands(self):
        prompts = make_math_prompts(64)
        answers = [int(answer) for _, answer in prompts]
        operands = []

        for prompt, answer in prompts:
            match = re.fullmatch(
                r"What is ([0-9]+)\+([0-9]+)\? Answer with just the number\.",
                prompt,
            )
            self.assertIsNotNone(match)
            a, b = (int(value) for value in match.groups())
            self.assertEqual(a + b, int(answer))
            operands.extend((a, b))

        self.assertEqual(len(answers), len(set(answers)))
        self.assertTrue(all(100 <= answer <= 999 for answer in answers))
        self.assertTrue(all(10 <= operand <= 99 for operand in operands))
        self.assertTrue(set(answers).isdisjoint(operands))

    def test_rejects_counts_outside_supported_concurrency(self):
        for count in (-1, 65):
            with self.subTest(count=count):
                with self.assertRaises(ValueError):
                    make_math_prompts(count)


if __name__ == "__main__":
    unittest.main()
