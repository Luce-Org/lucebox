"""Tests for server/scripts/bench_ds4_decode.py run validation."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "bench_ds4_decode.py"
SPEC = importlib.util.spec_from_file_location("bench_ds4_decode", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
bench_ds4_decode = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(bench_ds4_decode)


class ControlledDecodeBenchmarkTests(unittest.TestCase):
    @staticmethod
    def valid_run(**overrides: object) -> dict:
        run = {
            "completion_tokens": 512,
            "output_matches_prompt": True,
            "cache_hit": False,
            "cached_prefix_tokens": 0,
            "decode_tokens_per_second": 47.0,
            "spec_decode_ran": True,
            "accept_rate": 1.0,
        }
        run.update(overrides)
        return run

    def test_speculative_run_is_accepted(self) -> None:
        bench_ds4_decode.validate_run(self.valid_run(), max_tokens=512)

    def test_autoregressive_fallback_is_rejected(self) -> None:
        # An explicit autoregressive fallback and a missing signal both fail.
        for overrides in ({"spec_decode_ran": False, "accept_rate": 0.0},
                          {"spec_decode_ran": None}):
            with self.subTest(**overrides):
                with self.assertRaisesRegex(RuntimeError, "speculative decode did not run"):
                    bench_ds4_decode.validate_run(
                        self.valid_run(**overrides), max_tokens=512,
                    )


if __name__ == "__main__":
    unittest.main()
