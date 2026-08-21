#!/usr/bin/env python3
"""Focused policy/provenance checks for the canonical concurrency runner."""

from __future__ import annotations

import os
import re
import subprocess
import unittest
from pathlib import Path

RUNNER = Path(__file__).with_name("run_qwen36_canonical_concurrency.sh")


class CanonicalRunnerPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.script = RUNNER.read_text(encoding="utf-8")

    def test_prefill_first_policy_defaults_off_and_rejects_out_of_range_values(self) -> None:
        self.assertIn(
            'PREFILL_FIRST_BURST_STEPS="${PREFILL_FIRST_BURST_STEPS:-0}"',
            self.script,
        )

        for value in (
            "-1",
            "1025",
            "18446744073709551616",
            "999999999999999999999999999999999999",
        ):
            env = os.environ.copy()
            env.update(
                MODEL="/dev/null",
                SERVER_BIN="/bin/true",
                PREFILL_FIRST_BURST_STEPS=value,
            )
            result = subprocess.run(
                [str(RUNNER)], env=env, text=True, capture_output=True, check=False
            )
            with self.subTest(value=value):
                self.assertEqual(result.returncode, 2)
                self.assertIn(
                    "PREFILL_FIRST_BURST_STEPS must be an integer in range 0..1024",
                    result.stderr,
                )
        self.assertIn(
            '(( 10#$PREFILL_FIRST_BURST_STEPS > 1024 ))',
            self.script,
        )

    def test_idle_prefill_budget_rejects_zero_and_above_effective_cap(self) -> None:
        self.assertIn(
            'IDLE_PREFILL_TOKENS="${IDLE_PREFILL_TOKENS:-4096}"',
            self.script,
        )
        for value in (
            "0",
            "16385",
            "18446744073709551616",
            "999999999999999999999999999999999999",
        ):
            env = os.environ.copy()
            env.update(
                MODEL="/dev/null",
                SERVER_BIN="/bin/true",
                IDLE_PREFILL_TOKENS=value,
            )
            result = subprocess.run(
                [str(RUNNER)], env=env, text=True, capture_output=True, check=False
            )
            with self.subTest(value=value):
                self.assertEqual(result.returncode, 2)
                self.assertIn(
                    "IDLE_PREFILL_TOKENS must be an integer in range 1..16384",
                    result.stderr,
                )
        self.assertIn(
            '(( 10#$IDLE_PREFILL_TOKENS > 16384 ))',
            self.script,
        )

    def test_policy_is_inside_ar_and_ddtree_launch_arrays_before_command(self) -> None:
        launch_blocks = re.findall(
            r'launch=\(env (?P<body>.*?)"\$\{command\[@\]\}"\)',
            self.script,
            flags=re.DOTALL,
        )
        self.assertEqual(len(launch_blocks), 2)
        self.assertEqual(
            sum("DFLASH_DDTREE_ADAPTIVE" in block for block in launch_blocks), 1
        )
        assignment = (
            'DFLASH_PREFILL_FIRST_BURST_STEPS="$PREFILL_FIRST_BURST_STEPS"'
        )
        idle_assignment = (
            'DFLASH_IDLE_PREFILL_TOKENS="$IDLE_PREFILL_TOKENS"'
        )
        for block in launch_blocks:
            with self.subTest(ddtree="DFLASH_DDTREE_ADAPTIVE" in block):
                self.assertIn('ROCR_VISIBLE_DEVICES="$GPU_DEVICE"', block)
                self.assertIn(assignment, block)
                self.assertIn(idle_assignment, block)
                self.assertLess(block.index(assignment), block.index("stdbuf"))
                self.assertLess(block.index(idle_assignment), block.index("stdbuf"))

    def test_metadata_records_the_exact_resolved_policy_value(self) -> None:
        self.assertIn(
            "slots,prefill_first_burst_steps,expected_gpu_arch,"
            "idle_prefill_tokens=sys.argv[1:]",
            self.script,
        )
        self.assertIn(
            '"prefill_first_burst_steps":int(prefill_first_burst_steps)',
            self.script,
        )
        self.assertIn(
            '"$GPU_DEVICE" "$SLOTS" "$PREFILL_FIRST_BURST_STEPS"',
            self.script,
        )

        self.assertIn('"idle_prefill_tokens":int(idle_prefill_tokens)', self.script)
        self.assertIn(
            '"$EXPECTED_GPU_ARCH" "$IDLE_PREFILL_TOKENS"', self.script
        )

    def test_expected_arch_is_optional_recorded_and_checked_after_health(self) -> None:
        self.assertIn('EXPECTED_GPU_ARCH="${EXPECTED_GPU_ARCH:-}"', self.script)
        self.assertIn(
            '[[ -z "$EXPECTED_GPU_ARCH" || "$EXPECTED_GPU_ARCH" =~ '
            '^gfx[0-9a-f]+$ ]]',
            self.script,
        )
        self.assertIn('"expected_gpu_arch":expected_gpu_arch or None', self.script)
        self.assertIn(
            '"$PREFILL_FIRST_BURST_STEPS" "$EXPECTED_GPU_ARCH"', self.script
        )
        health = self.script.index("if ! wait_health")
        literal_check = self.script.index(
            'grep -F -- "$EXPECTED_GPU_ARCH" "$case_dir/server.log"', health
        )
        client = self.script.index("client_common=(", literal_check)
        self.assertLess(health, literal_check)
        self.assertLess(literal_check, client)
        guard = self.script[literal_check:client]
        self.assertIn('> "$case_dir/gpu-identity.txt"', guard)
        self.assertLess(guard.index("stop_server"), guard.index("return 1"))


if __name__ == "__main__":
    unittest.main()
