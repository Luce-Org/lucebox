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

from harness.metrics_parser import (
    BanditRunMetrics,
    BanditTurnRecord,
    parse_bandit_log_line,
    parse_bandit_log,
    parse_spec_decode_line,
    extract_accept_rate_from_log,
    parse_bandit_session_from_log,
)


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


class TestSpecDecodeParser(unittest.TestCase):
    """Tests for the [spec-decode] plain-text log line parser."""

    def test_spec_decode_line_parses_accept_rate(self):
        """[spec-decode] line with accepted=8/10 → accept_rate=0.8."""
        line = "[spec-decode] tokens=312 time=18.50 s speed=16.9 tok/s steps=10 accepted=8/10"
        m = parse_spec_decode_line(line)
        self.assertIsNotNone(m)
        self.assertAlmostEqual(m.accept_rate, 0.8)

    def test_spec_decode_line_full_acceptance(self):
        """accepted=5/5 → accept_rate=1.0."""
        line = "[spec-decode] tokens=50 time=2.1 s speed=23.8 tok/s steps=5 accepted=5/5"
        m = parse_spec_decode_line(line)
        self.assertIsNotNone(m)
        self.assertAlmostEqual(m.accept_rate, 1.0)

    def test_spec_decode_line_zero_steps_returns_none(self):
        """accepted=0/0 (degenerate) → None rather than division by zero."""
        line = "[spec-decode] tokens=0 time=0.0 s speed=0 tok/s steps=0 accepted=0/0"
        m = parse_spec_decode_line(line)
        self.assertIsNone(m)

    def test_spec_decode_non_matching_line_returns_none(self):
        """Non-[spec-decode] line → None."""
        line = "2026-05-23 INFO prefill done tokens=100"
        m = parse_spec_decode_line(line)
        self.assertIsNone(m)


class TestExtractAcceptRateFromLog(unittest.TestCase):
    """Tests for extract_accept_rate_from_log (Blocker #6 wiring helper)."""

    def test_extracts_from_pflash_bandit_json_line(self):
        """[pflash-bandit] JSON line → accept_rate returned."""
        log = (
            '2026-05-23 INFO startup\n'
            '[pflash-bandit] {"accept_rate": 0.55, "session_id": "s1"}\n'
            '2026-05-23 INFO done\n'
        )
        rate = extract_accept_rate_from_log(log)
        self.assertIsNotNone(rate)
        self.assertAlmostEqual(rate, 0.55)

    def test_uses_last_pflash_bandit_line(self):
        """Multiple [pflash-bandit] lines → last one wins (converged state)."""
        log = (
            '[pflash-bandit] {"accept_rate": 0.30}\n'
            '[pflash-bandit] {"accept_rate": 0.45}\n'
            '[pflash-bandit] {"accept_rate": 0.60}\n'
        )
        rate = extract_accept_rate_from_log(log)
        self.assertAlmostEqual(rate, 0.60)

    def test_parses_plain_text_pflash_bandit_accept_line(self):
        """Plain-text [pflash-bandit] accept=... lines → accept_rate returned."""
        log = (
            '[pflash-bandit] session=s1 turn=1 keep=0.1000->0.2000 ema=0.123 accept=0.347\n'
            '[pflash-bandit] session=s1 turn=2 keep=0.2000->0.3000 ema=0.456 accept=0.812\n'
        )
        rate = extract_accept_rate_from_log(log)
        self.assertIsNotNone(rate)
        self.assertAlmostEqual(rate, 0.812)

    def test_falls_back_to_spec_decode_when_no_bandit(self):
        """No [pflash-bandit] lines → fall back to [spec-decode]."""
        log = (
            '2026-05-23 INFO startup\n'
            '[spec-decode] tokens=200 time=10.0 s speed=20.0 tok/s steps=5 accepted=4/5\n'
        )
        rate = extract_accept_rate_from_log(log)
        self.assertIsNotNone(rate)
        self.assertAlmostEqual(rate, 0.8)

    def test_returns_none_when_no_matching_lines(self):
        """Log with no [pflash-bandit] or [spec-decode] → None."""
        log = "2026-05-23 INFO server started\n2026-05-23 INFO request received\n"
        rate = extract_accept_rate_from_log(log)
        self.assertIsNone(rate)

    def test_bandit_preferred_over_spec_decode(self):
        """When both present, [pflash-bandit] takes priority."""
        log = (
            '[spec-decode] tokens=100 time=5.0 s speed=20.0 tok/s steps=5 accepted=2/5\n'
            '[pflash-bandit] {"accept_rate": 0.75}\n'
        )
        rate = extract_accept_rate_from_log(log)
        self.assertAlmostEqual(rate, 0.75)


class TestParseBanditSessionFromLog(unittest.TestCase):
    """Tests for parse_bandit_session_from_log."""

    def _make_log(self, turns: list[dict]) -> str:
        lines = []
        for t in turns:
            keep_before = t.get("keep_before", 0.10)
            keep_after = t.get("keep_after", 0.12)
            ema = t.get("ema", 0.25)
            accept = t.get("accept", 0.35)
            turn = t.get("turn", 1)
            session = t.get("session", "s1")
            lines.append(
                f"[pflash-bandit] session={session} turn={turn} "
                f"keep={keep_before:.4f}->{keep_after:.4f} "
                f"ema={ema:.3f} accept={accept:.3f}"
            )
        return "\n".join(lines) + "\n"

    def test_parses_single_turn(self):
        log = self._make_log([{"turn": 1, "keep_before": 0.10, "keep_after": 0.12,
                                "ema": 0.25, "accept": 0.35}])
        records = parse_bandit_session_from_log(log, session_id="s1")
        self.assertEqual(len(records), 1)
        r = records[0]
        self.assertEqual(r.turn, 1)
        self.assertAlmostEqual(r.keep_before, 0.10, places=4)
        self.assertAlmostEqual(r.keep_after, 0.12, places=4)
        self.assertAlmostEqual(r.ema, 0.25, places=3)
        self.assertAlmostEqual(r.accept_rate, 0.35, places=3)

    def test_parses_five_turns(self):
        turns = [
            {"turn": 1, "keep_before": 0.10, "keep_after": 0.12, "ema": 0.20, "accept": 0.40},
            {"turn": 2, "keep_before": 0.12, "keep_after": 0.15, "ema": 0.30, "accept": 0.55},
            {"turn": 3, "keep_before": 0.15, "keep_after": 0.18, "ema": 0.38, "accept": 0.62},
            {"turn": 4, "keep_before": 0.18, "keep_after": 0.20, "ema": 0.44, "accept": 0.70},
            {"turn": 5, "keep_before": 0.20, "keep_after": 0.22, "ema": 0.50, "accept": 0.75},
        ]
        log = self._make_log(turns)
        records = parse_bandit_session_from_log(log, session_id="s1")
        self.assertEqual(len(records), 5)
        # keep_after changes across turns
        keep_afters = [r.keep_after for r in records]
        self.assertGreater(len(set(keep_afters)), 1, msg="keep_after must vary across turns")

    def test_filters_by_session_id(self):
        turns_s1 = [{"turn": 1, "session": "s1", "keep_before": 0.10, "keep_after": 0.12,
                      "ema": 0.25, "accept": 0.35}]
        turns_s2 = [{"turn": 1, "session": "s2", "keep_before": 0.10, "keep_after": 0.11,
                      "ema": 0.20, "accept": 0.30}]
        log = self._make_log(turns_s1) + self._make_log(turns_s2)
        records = parse_bandit_session_from_log(log, session_id="s1")
        self.assertEqual(len(records), 1)
        self.assertAlmostEqual(records[0].keep_after, 0.12, places=4)

    def test_session_id_none_returns_all(self):
        turns = [
            {"turn": 1, "session": "a", "keep_before": 0.10, "keep_after": 0.11,
             "ema": 0.25, "accept": 0.30},
            {"turn": 2, "session": "b", "keep_before": 0.15, "keep_after": 0.16,
             "ema": 0.30, "accept": 0.40},
        ]
        log = self._make_log(turns)
        records = parse_bandit_session_from_log(log, session_id=None)
        self.assertEqual(len(records), 2)

    def test_empty_log_returns_empty(self):
        records = parse_bandit_session_from_log("no bandit lines here\n", session_id="s1")
        self.assertEqual(records, [])


if __name__ == "__main__":
    unittest.main()
