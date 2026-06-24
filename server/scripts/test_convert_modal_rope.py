#!/usr/bin/env python3
"""
Pure-Python unit tests for resolve_rope_theta().

No torch, no gguf, no GPU needed.  Imports only the pure helper from
convert_modal_dflash_to_gguf, making this safe to run offline.
"""

import sys
import unittest

# Import only the pure function — no IO, no gguf, no safetensors needed.
from convert_modal_dflash_to_gguf import resolve_rope_theta


class TestResolveRopeTheta(unittest.TestCase):

    # ── happy-path cases ──────────────────────────────────────────────

    def test_rope_parameters_scope_wins(self):
        """35B-a3b layout: theta nested under rope_parameters."""
        cfg = {"rope_parameters": {"rope_theta": 10_000_000}}
        self.assertEqual(resolve_rope_theta(cfg), 10_000_000.0)

    def test_rope_parameters_beats_top_level(self):
        """When both scopes present, rope_parameters takes priority."""
        cfg = {
            "rope_theta": 1_000_000,
            "rope_parameters": {"rope_theta": 10_000_000},
        }
        self.assertEqual(resolve_rope_theta(cfg), 10_000_000.0)

    def test_top_level_fallback(self):
        """27B legacy layout: theta at top level, no rope_parameters."""
        cfg = {"rope_theta": 10_000_000}
        self.assertEqual(resolve_rope_theta(cfg), 10_000_000.0)

    def test_genuine_1m_top_level(self):
        """gemma4 case: real top-level theta=1_000_000 must resolve correctly."""
        cfg = {"rope_theta": 1_000_000}
        self.assertEqual(resolve_rope_theta(cfg), 1_000_000.0)

    def test_genuine_1m_in_rope_parameters(self):
        """theta=1_000_000 inside rope_parameters must also resolve correctly."""
        cfg = {"rope_parameters": {"rope_theta": 1_000_000}}
        self.assertEqual(resolve_rope_theta(cfg), 1_000_000.0)

    def test_returns_float(self):
        """Result is always float regardless of source type (int vs float)."""
        cfg = {"rope_theta": 500_000}
        result = resolve_rope_theta(cfg)
        self.assertIsInstance(result, float)

    # ── fail-loud cases ───────────────────────────────────────────────

    def test_neither_scope_exits(self):
        """No rope_theta anywhere → must raise SystemExit (NOT silently return 1M)."""
        cfg = {}
        with self.assertRaises(SystemExit):
            resolve_rope_theta(cfg)

    def test_empty_rope_parameters_exits(self):
        """rope_parameters present but empty → must raise SystemExit."""
        cfg = {"rope_parameters": {}}
        with self.assertRaises(SystemExit):
            resolve_rope_theta(cfg)

    def test_none_top_level_exits(self):
        """Explicit null at top level, no rope_parameters → must raise SystemExit."""
        cfg = {"rope_theta": None}
        with self.assertRaises(SystemExit):
            resolve_rope_theta(cfg)

    def test_none_in_rope_parameters_falls_back_to_top_level(self):
        """Null inside rope_parameters → fall through to top-level."""
        cfg = {"rope_parameters": {"rope_theta": None}, "rope_theta": 10_000_000}
        self.assertEqual(resolve_rope_theta(cfg), 10_000_000.0)

    def test_none_everywhere_exits(self):
        """Null in both scopes → must raise SystemExit."""
        cfg = {"rope_parameters": {"rope_theta": None}, "rope_theta": None}
        with self.assertRaises(SystemExit):
            resolve_rope_theta(cfg)

    def test_silent_1m_default_is_gone(self):
        """Critical regression guard: empty config must NOT return 1_000_000."""
        cfg = {}
        try:
            result = resolve_rope_theta(cfg)
            # If we reach here, the function returned instead of exiting.
            self.fail(
                f"resolve_rope_theta({{}}) returned {result!r} "
                f"instead of raising SystemExit — silent-1M default is back"
            )
        except SystemExit:
            pass  # correct — loud failure, not silent wrong value


if __name__ == "__main__":
    unittest.main(verbosity=2)
