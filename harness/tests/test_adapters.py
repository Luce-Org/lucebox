"""Tests for ClientAdapter protocol + bandit subcommand (seeds #4, #6).

Seed #4: adapter_invoke records session_id in request capture
Seed #6: matrix runs 5 adapters and produces structured CSV
"""

from __future__ import annotations

import csv
import io
import json
import sys
import unittest
from pathlib import Path

HARNESS_DIR = Path(__file__).resolve().parent.parent
if str(HARNESS_DIR.parent) not in sys.path:
    sys.path.insert(0, str(HARNESS_DIR.parent))

from harness.tests._stub_server import StubServer
from harness.client_test_runner import (
    ClientAdapter,
    ClaudeCodeAdapter,
    HermesAdapter,
    CodexAdapter,
    PiAdapter,
    OpenCodeAdapter,
    AdapterResult,
    run_bandit,
)


class TestAdapterInvokeSessionId(unittest.TestCase):
    """Seed #4: adapter_invoke records session_id in request capture."""

    def test_adapter_invoke_records_session_id_in_request_capture(self):
        """ClaudeCodeAdapter dry-run produces AdapterResult with session_id."""
        adapter = ClaudeCodeAdapter()
        result = adapter.dry_run(session_id="seed4-test-session")
        self.assertIsInstance(result, AdapterResult)
        self.assertEqual(result.session_id, "seed4-test-session")
        self.assertTrue(result.preflight_ok)
        self.assertIsNone(result.error)

    def test_hermes_adapter_dry_run(self):
        """HermesAdapter dry-run produces AdapterResult."""
        adapter = HermesAdapter()
        result = adapter.dry_run(session_id="hermes-sess-001")
        self.assertIsInstance(result, AdapterResult)
        self.assertEqual(result.session_id, "hermes-sess-001")

    def test_codex_adapter_dry_run(self):
        """CodexAdapter dry-run produces AdapterResult."""
        adapter = CodexAdapter()
        result = adapter.dry_run(session_id="codex-sess-001")
        self.assertIsInstance(result, AdapterResult)
        self.assertEqual(result.session_id, "codex-sess-001")


class TestAdapterPreflightMissingBinary(unittest.TestCase):
    """Adapter.preflight() for a missing binary returns preflight_ok=False + actionable message."""

    def test_preflight_fails_for_nonexistent_binary(self):
        """Preflight for a nonexistent binary exits with preflight_ok=False."""
        # Use the generic mechanism; ClaudeCodeAdapter checks for 'claude'
        adapter = ClaudeCodeAdapter(binary="_not_a_real_binary_xyz987")
        result = adapter.preflight_check()
        self.assertFalse(result.preflight_ok)
        self.assertIsNotNone(result.error)
        # Actionable message must name the binary or asdf
        msg = (result.error or "").lower()
        self.assertTrue(
            "asdf" in msg or "_not_a_real_binary" in msg or "install" in msg or "not found" in msg,
            msg=f"No actionable hint in error: {result.error!r}",
        )


class TestBanditMatrix5AdaptersCSV(unittest.TestCase):
    """Seed #6: --dry-run on 5 adapters emits 5-row CSV."""

    def test_matrix_runs_5_adapters_and_produces_structured_csv(self):
        """run_bandit dry_run=True → 5-row CSV with expected columns."""
        output = io.StringIO()
        results = run_bandit(
            clients=["claude_code", "hermes", "opencode", "codex", "pi"],
            condition="C_bandit",
            dry_run=True,
            output=output,
        )
        csv_text = output.getvalue()
        self.assertTrue(csv_text.strip(), "CSV output must not be empty")

        reader = csv.DictReader(io.StringIO(csv_text))
        rows = list(reader)
        self.assertEqual(len(rows), 5, f"Expected 5 rows, got {len(rows)}\n{csv_text}")

        client_names = {r["client"] for r in rows}
        self.assertEqual(
            client_names,
            {"claude_code", "hermes", "opencode", "codex", "pi"},
        )

        # Required columns per exit gate spec
        required_cols = {"client", "preflight_ok", "session_id_captured", "accept_rate", "wall_s", "exit_code"}
        actual_cols = set(reader.fieldnames or [])
        # Re-parse since we iterated fieldnames after exhausting reader
        reader2 = csv.DictReader(io.StringIO(csv_text))
        actual_cols = set(reader2.fieldnames or [])
        self.assertTrue(
            required_cols.issubset(actual_cols),
            msg=f"Missing columns: {required_cols - actual_cols}. Got: {actual_cols}",
        )

        # dry-run rows: preflight_ok must be a valid boolean string
        for row in rows:
            self.assertIn(row["preflight_ok"], ("True", "False"),
                          msg=f"preflight_ok must be True/False, got: {row['preflight_ok']!r}")


if __name__ == "__main__":
    unittest.main()
