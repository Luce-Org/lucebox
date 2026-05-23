"""Tests for typed metrics parser (seed #5).

Verifies that the BanditRunMetrics parser:
  - Returns None (not "N/A") for missing accept_rate, wall, tokens, session_id
  - Parses numeric fields correctly when present
  - Handles a log fixture with incomplete rows (Day-4-v2 pattern)
"""

from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

HARNESS_DIR = Path(__file__).resolve().parent.parent
if str(HARNESS_DIR.parent) not in sys.path:
    sys.path.insert(0, str(HARNESS_DIR.parent))

from harness.metrics_parser import BanditRunMetrics, parse_bandit_log_line, parse_bandit_log


# A Day-4-v2-style log line with all fields present
FULL_LOG_LINE = json.dumps({
    "session_id": "sess-abc123",
    "accept_rate": 0.42,
    "wall_s": 18.5,
    "tokens": 312,
    "client": "hermes",
    "condition": "C_bandit",
})

# A log line missing accept_rate (the Day-4-v2 "N/A" scenario)
MISSING_ACCEPT_RATE_LINE = json.dumps({
    "session_id": "sess-def456",
    "wall_s": 22.1,
    "tokens": 280,
    "client": "hermes",
    "condition": "C_bandit",
})

# A log line missing everything except session_id
MINIMAL_LINE = json.dumps({
    "session_id": "sess-min-001",
})

# A non-JSON line (should be skipped gracefully)
JUNK_LINE = "2026-05-23 INFO server started on port 18080"


class TestBanditRunMetricsParser(unittest.TestCase):

    def test_full_line_parses_correctly(self):
        """All fields present → typed values, no 'N/A' strings."""
        m = parse_bandit_log_line(FULL_LOG_LINE)
        self.assertIsNotNone(m)
        self.assertEqual(m.session_id, "sess-abc123")
        self.assertAlmostEqual(m.accept_rate, 0.42)
        self.assertAlmostEqual(m.wall_s, 18.5)
        self.assertEqual(m.tokens, 312)
        self.assertEqual(m.client, "hermes")
        # No "N/A" strings leaked into typed fields
        self.assertNotEqual(m.accept_rate, "N/A")

    def test_metrics_parser_handles_missing_accept_rate_field(self):
        """Missing accept_rate → None, not 'N/A' string (seed #5)."""
        m = parse_bandit_log_line(MISSING_ACCEPT_RATE_LINE)
        self.assertIsNotNone(m)
        self.assertIsNone(m.accept_rate, msg="accept_rate must be None when absent, not 'N/A'")
        self.assertAlmostEqual(m.wall_s, 22.1)
        self.assertEqual(m.tokens, 280)

    def test_minimal_line_has_none_for_missing_fields(self):
        """Minimal line: all optional fields are None."""
        m = parse_bandit_log_line(MINIMAL_LINE)
        self.assertIsNotNone(m)
        self.assertIsNone(m.accept_rate)
        self.assertIsNone(m.wall_s)
        self.assertIsNone(m.tokens)
        self.assertIsNone(m.client)

    def test_junk_line_returns_none(self):
        """Non-JSON lines return None gracefully."""
        m = parse_bandit_log_line(JUNK_LINE)
        self.assertIsNone(m)

    def test_parse_bandit_log_multi_line(self):
        """parse_bandit_log processes multiple lines, skips junk."""
        lines = [
            FULL_LOG_LINE,
            MISSING_ACCEPT_RATE_LINE,
            JUNK_LINE,
            MINIMAL_LINE,
        ]
        results = parse_bandit_log("\n".join(lines))
        # 3 valid JSON lines, 1 junk
        self.assertEqual(len(results), 3)
        # accept_rate correctly None on the second result
        self.assertIsNone(results[1].accept_rate)
        # First result has numeric accept_rate
        self.assertAlmostEqual(results[0].accept_rate, 0.42)

    def test_bandit_run_metrics_fields(self):
        """BanditRunMetrics has the expected typed fields."""
        m = BanditRunMetrics(
            session_id="s1",
            accept_rate=0.5,
            wall_s=10.0,
            tokens=100,
            client="claude_code",
            condition="C_bandit",
        )
        self.assertIsInstance(m.session_id, str)
        self.assertIsInstance(m.accept_rate, float)
        self.assertIsInstance(m.wall_s, float)
        self.assertIsInstance(m.tokens, int)


if __name__ == "__main__":
    unittest.main()
