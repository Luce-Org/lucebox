#!/usr/bin/env python3
"""Input-validation checks for the canonical concurrency runner."""

from __future__ import annotations

import os
import subprocess
import tempfile
import unittest
from pathlib import Path

RUNNER = Path(__file__).with_name("run_qwen36_canonical_concurrency.sh")


class CanonicalRunnerPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        # A regression that lets a value through must not write into the repo.
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.out = Path(self.temp.name) / "out"

    def test_policy_rejects_out_of_range_values(self) -> None:
        huge = ("18446744073709551616", "999999999999999999999999999999999999")
        cases = (
            ("PREFILL_FIRST_BURST_STEPS", ("-1", "1025", *huge),
             "PREFILL_FIRST_BURST_STEPS must be an integer in range 0..1024"),
            ("IDLE_PREFILL_TOKENS", ("0", "16385", *huge),
             "IDLE_PREFILL_TOKENS must be an integer in range 1..16384"),
        )
        for variable, values, message in cases:
            for value in values:
                env = os.environ.copy()
                env.update(
                    MODEL="/dev/null",
                    SERVER_BIN="/bin/true",
                    OUT=str(self.out),
                    **{variable: value},
                )
                result = subprocess.run(
                    [str(RUNNER)], env=env, text=True, capture_output=True, check=False
                )
                with self.subTest(variable=variable, value=value):
                    self.assertEqual(result.returncode, 2)
                    self.assertIn(message, result.stderr)

    def test_policy_range_edges_are_accepted(self) -> None:
        # Each accepted edge must pass its own guard and stop at a later one,
        # so a shifted or over-tight cap changes which message is printed.
        cases = (
            ({"PREFILL_FIRST_BURST_STEPS": "0", "IDLE_PREFILL_TOKENS": "0"}, "IDLE_PREFILL_TOKENS"),
            ({"PREFILL_FIRST_BURST_STEPS": "1024", "IDLE_PREFILL_TOKENS": "0"}, "IDLE_PREFILL_TOKENS"),
            ({"IDLE_PREFILL_TOKENS": "1", "SLOTS": "8"}, "SLOTS"),
            ({"IDLE_PREFILL_TOKENS": "16384", "SLOTS": "8"}, "SLOTS"),
        )
        for overrides, later_guard in cases:
            env = os.environ.copy()
            env.update(
                MODEL="/dev/null", SERVER_BIN="/bin/true", OUT=str(self.out), **overrides
            )
            result = subprocess.run(
                [str(RUNNER)], env=env, text=True, capture_output=True, check=False
            )
            with self.subTest(**overrides):
                self.assertEqual(result.returncode, 2)
                self.assertTrue(
                    result.stderr.startswith(f"{later_guard} must be"), result.stderr
                )


if __name__ == "__main__":
    unittest.main()
