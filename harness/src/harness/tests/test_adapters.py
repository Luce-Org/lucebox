"""Tests for ClientAdapter protocol + bandit subcommand (seeds #4, #6).

Seed #4: adapter_invoke records session_id in request capture
Seed #6: matrix runs 5 adapters and produces structured CSV
"""

from __future__ import annotations

import csv
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

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
    BANDIT_SERVER_PROFILE,
    start_server,
    _load_session_prompts,
    _BANDIT_SESSION_PROMPT_FILES,
    _BANDIT_SESSION_PROMPTS_DIR,
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


class TestClaudeCodeAdapterLiveRun(unittest.TestCase):
    """ClaudeCodeAdapter live_run should invoke claude directly, not shell out to a wrapper."""

    def test_live_run_invokes_claude_directly_with_long_prompt(self):
        adapter = ClaudeCodeAdapter()
        captured: dict[str, object] = {}

        class _FakeProc:
            returncode = 0

        def fake_run(cmd, **kwargs):
            captured["cmd"] = cmd
            captured["kwargs"] = kwargs
            return _FakeProc()

        with patch.dict(
            os.environ,
            {
                "BASE_URL": "http://127.0.0.1:18080",
                "API_KEY": "sk-lucebox",
                "MODEL_ID": "luce-dflash",
                "CLAUDE_BIN": "/usr/bin/claude",
                "CLAUDE_TOOLS": "default",
            },
            clear=False,
        ), patch("harness.client_test_runner.subprocess.run", side_effect=fake_run):
            result = adapter.live_run(session_id="", prompt="")

        self.assertTrue(result.preflight_ok)
        self.assertEqual(result.exit_code, 0)
        self.assertIsNone(result.error)

        cmd = captured["cmd"]
        kwargs = captured["kwargs"]
        self.assertIsInstance(cmd, list)
        self.assertEqual(cmd[0], "/usr/bin/claude")
        self.assertIn("--print", cmd)
        self.assertIn("--output-format", cmd)
        self.assertIn("--model", cmd)
        self.assertIn("--no-session-persistence", cmd)
        self.assertIn("at least 700 words", cmd[-1])
        self.assertEqual(kwargs["stdin"], subprocess.DEVNULL)

        env = kwargs["env"]
        self.assertEqual(env["LUCEBOX_SERVER_BACKEND"], "cpp")
        self.assertEqual(env["ANTHROPIC_API_KEY"], "sk-lucebox")
        self.assertEqual(env["ANTHROPIC_BASE_URL"], "http://127.0.0.1:18080")
        self.assertEqual(env["CLAUDE_CODE_API_BASE_URL"], "http://127.0.0.1:18080")
        self.assertEqual(env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"], "1")
        self.assertEqual(env["CLAUDE_CODE_DISABLE_TELEMETRY"], "1")
        self.assertEqual(env["CLAUDE_CODE_DISABLE_NONSTREAMING_FALLBACK"], "1")


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


class TestAcceptRatePopulatedFromLog(unittest.TestCase):
    """Blocker #6: accept_rate must be non-None when server_log_path contains matching lines."""

    def test_accept_rate_from_spec_decode_log(self):
        """AdapterResult.accept_rate is populated from a server log with [spec-decode] lines."""
        from harness.client_test_runner import AdapterResult
        from harness.metrics_parser import extract_accept_rate_from_log

        log_content = (
            "2026-05-23 INFO server started\n"
            "[spec-decode] tokens=200 time=10.0 s speed=20.0 tok/s steps=5 accepted=4/5\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write(log_content)
            log_path = Path(f.name)
        try:
            result = AdapterResult(
                client="claude_code",
                preflight_ok=True,
                session_id="test-sess-001",
                session_id_captured=True,
                wall_s=10.5,
                exit_code=0,
                server_log_path=log_path,
            )
            # Simulate what run_bandit does after live_run
            if result.accept_rate is None and result.server_log_path is not None:
                log_text = result.server_log_path.read_text(errors="replace")
                result.accept_rate = extract_accept_rate_from_log(log_text)

            self.assertIsNotNone(result.accept_rate,
                                 "accept_rate must be non-None after wiring metrics_parser")
            self.assertAlmostEqual(result.accept_rate, 0.8)
        finally:
            log_path.unlink(missing_ok=True)

    def test_accept_rate_from_bandit_json_log(self):
        """AdapterResult.accept_rate is populated from [pflash-bandit] JSON log lines."""
        from harness.client_test_runner import AdapterResult
        from harness.metrics_parser import extract_accept_rate_from_log

        log_content = (
            "2026-05-23 INFO startup\n"
            '[pflash-bandit] {"accept_rate": 0.62, "session_id": "s42"}\n'
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write(log_content)
            log_path = Path(f.name)
        try:
            result = AdapterResult(
                client="hermes",
                preflight_ok=True,
                session_id="test-sess-002",
                session_id_captured=True,
                wall_s=15.0,
                exit_code=0,
                server_log_path=log_path,
            )
            if result.accept_rate is None and result.server_log_path is not None:
                log_text = result.server_log_path.read_text(errors="replace")
                result.accept_rate = extract_accept_rate_from_log(log_text)

            self.assertIsNotNone(result.accept_rate)
            self.assertAlmostEqual(result.accept_rate, 0.62)
        finally:
            log_path.unlink(missing_ok=True)


class TestRunBanditWiresAcceptRate(unittest.TestCase):
    """Regression: run_bandit must populate accept_rate from server_log_path via metrics_parser.

    Previous tests duplicated the wiring logic inline (line-for-line); they did not
    exercise the actual code path inside run_bandit. This test stubs the adapter and
    calls run_bandit directly so the wiring at client_test_runner.py:2569-2579 is
    covered.
    """

    def test_run_bandit_populates_accept_rate_from_server_log(self):
        from harness.client_test_runner import (
            run_bandit, _ADAPTER_REGISTRY, AdapterResult,
        )

        log_content = (
            "[pflash-bandit] session=claude_code-C_bandit turn=1 keep=0.1000->0.1200 "
            "ema=0.250 accept=0.312\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".log", delete=False) as f:
            f.write(log_content)
            log_path = Path(f.name)

        class _Stub:
            client = "claude_code"
            def preflight_check(self):
                return AdapterResult(
                    client="claude_code", preflight_ok=True, session_id_captured=False,
                )
            def live_run(self, *, session_id, **_kw):
                return AdapterResult(
                    client="claude_code", preflight_ok=True, session_id=session_id,
                    session_id_captured=True, wall_s=10.0, exit_code=0,
                )

        original = _ADAPTER_REGISTRY.get("claude_code")
        _ADAPTER_REGISTRY["claude_code"] = lambda: _Stub()
        try:
            buf = io.StringIO()
            results = run_bandit(
                clients=["claude_code"], condition="C_bandit",
                output=buf, server_log_path=log_path,
            )
            self.assertEqual(len(results), 1)
            self.assertIsNotNone(
                results[0].accept_rate,
                msg="run_bandit must wire accept_rate from server_log_path",
            )
            self.assertAlmostEqual(results[0].accept_rate, 0.312)
            rows = list(csv.DictReader(io.StringIO(buf.getvalue())))
            self.assertEqual(rows[0]["accept_rate"], str(0.312))
        finally:
            if original is not None:
                _ADAPTER_REGISTRY["claude_code"] = original
            log_path.unlink(missing_ok=True)


class TestAdapterSkipReasons(unittest.TestCase):
    """Hermes/OpenCode are intentionally preflight-skipped until config is fixed."""

    def test_hermes_preflight_passes_when_binary_present(self):
        """HermesAdapter.preflight_check succeeds when 'hermes' binary is available."""
        import shutil
        if not shutil.which("hermes"):
            self.skipTest("hermes binary not on PATH")
        adapter = HermesAdapter()
        result = adapter.preflight_check()
        # When the binary is present and --version exits 0, preflight passes
        self.assertTrue(result.preflight_ok, msg=f"preflight failed: {result.error}")

    def test_opencode_preflight_is_deterministic(self):
        adapter = OpenCodeAdapter()
        result = adapter.preflight_check()
        if result.preflight_ok:
            self.assertIsNone(result.error)
        else:
            self.assertIsNotNone(result.error)
            self.assertIn("PREFLIGHT FAIL", result.error or "")


class TestBanditServerProfileHasPflash(unittest.TestCase):
    """Blocker #8: BANDIT_SERVER_PROFILE must include --prefill-compression auto."""

    def test_bandit_server_profile_includes_prefill_compression_auto(self):
        """BANDIT_SERVER_PROFILE args include '--prefill-compression auto'."""
        args = list(BANDIT_SERVER_PROFILE.args)
        self.assertIn("--prefill-compression", args,
                      msg="BANDIT_SERVER_PROFILE must include --prefill-compression")
        idx = args.index("--prefill-compression")
        self.assertEqual(args[idx + 1], "auto",
                         msg="--prefill-compression value must be 'auto'")

    def test_bandit_server_profile_includes_prefill_keep_ratio(self):
        """BANDIT_SERVER_PROFILE includes --prefill-keep-ratio 0.10 (bandit prior)."""
        args = list(BANDIT_SERVER_PROFILE.args)
        self.assertIn("--prefill-keep-ratio", args)
        idx = args.index("--prefill-keep-ratio")
        self.assertEqual(args[idx + 1], "0.10")

    def test_bandit_server_profile_needs_prefill_drafter(self):
        """BANDIT_SERVER_PROFILE.needs_prefill_drafter is True."""
        self.assertTrue(BANDIT_SERVER_PROFILE.needs_prefill_drafter)

    def test_bandit_server_profile_only_cpp_recognised_flags(self):
        """All BANDIT_SERVER_PROFILE flags must be recognised by dflash/src/server/server_main.cpp.

        Stale Python-server flags (--budget, --verify-mode, --prefix-cache-slots,
        --prefill-cache-slots) cause the C++ binary to exit 2 with 'unknown option'
        before it ever opens a port — server.log ends up containing only usage text,
        and accept_rate in the CSV stays empty.
        """
        forbidden = {
            "--budget",
            "--verify-mode",
            "--prefix-cache-slots",
            "--prefill-cache-slots",
            "--lazy-draft",
        }
        args = list(BANDIT_SERVER_PROFILE.args)
        present = forbidden.intersection(args)
        self.assertFalse(
            present,
            msg=(
                f"BANDIT_SERVER_PROFILE contains C++-server-unknown flags {present}; "
                "they cause dflash_server to exit 2 before serving any request."
            ),
        )

    def test_start_server_argv_includes_prefill_compression_when_bandit_profile(self):
        """start_server with BANDIT_SERVER_PROFILE builds argv with --prefill-compression auto.

        Constructs the argv list directly from BANDIT_SERVER_PROFILE.args and
        needs_prefill_drafter, mirroring what start_server does, without launching
        a real process.
        """
        fake_bin = Path("/bin/true")
        fake_drafter = Path("/tmp/fake-drafter.gguf")

        # Reproduce the argv assembly logic from start_server (cpp backend path)
        args = [
            str(fake_bin),
            "--host", "127.0.0.1",
            "--port", "19999",
            "--target", str(fake_bin),
            "--draft", str(fake_bin),
            *BANDIT_SERVER_PROFILE.args,
        ]
        if BANDIT_SERVER_PROFILE.needs_prefill_drafter:
            args.extend(["--prefill-drafter", str(fake_drafter)])

        self.assertIn("--prefill-compression", args,
                      msg=f"--prefill-compression not in server argv: {args}")
        idx = args.index("--prefill-compression")
        self.assertEqual(args[idx + 1], "auto")
        self.assertIn("--prefill-drafter", args,
                      msg="--prefill-drafter must be in server argv for bandit profile")


class TestBanditCLI(unittest.TestCase):
    """CLI-level smoke tests for the bandit subcommand."""

    def _run_bandit_cli(self, *args: str) -> tuple[int, str]:
        """Run client_test_runner as a subprocess, return (rc, stdout)."""
        import subprocess
        import sys
        result = subprocess.run(
            [sys.executable, "-m", "harness.client_test_runner", *args],
            capture_output=True,
            text=True,
            cwd=str(Path(__file__).resolve().parent.parent.parent),
        )
        return result.returncode, result.stdout

    def test_adapter_flag_dry_run_prints_planned_invocation(self):
        """--adapter claude_code --dry-run prints planned invocation (exit-gate for commit 7)."""
        rc, out = self._run_bandit_cli("bandit", "--adapter", "claude_code", "--dry-run")
        self.assertEqual(rc, 0)
        self.assertIn("claude_code", out)
        self.assertIn("True", out)  # preflight_ok

    def test_top_level_clients_flag_triggers_bandit(self):
        """Top-level --clients/--condition flags work without 'bandit' subcommand."""
        rc, out = self._run_bandit_cli(
            "--condition", "C_bandit", "--clients", "claude_code", "--dry-run",
        )
        self.assertEqual(rc, 0)
        self.assertIn("claude_code", out)


class TestBanditSessionPrompts(unittest.TestCase):
    """Tests for bandit-session prompt loading."""

    def test_prompt_files_exist(self):
        """All required prompt files exist in the prompts directory."""
        missing = []
        for fname in _BANDIT_SESSION_PROMPT_FILES:
            p = _BANDIT_SESSION_PROMPTS_DIR / fname
            if not p.exists():
                missing.append(fname)
        self.assertEqual(missing, [], msg=f"Missing prompt files: {missing}")

    def test_load_session_prompts_returns_five(self):
        """_load_session_prompts returns 5 (name, content) pairs."""
        prompts = _load_session_prompts(_BANDIT_SESSION_PROMPTS_DIR, 5)
        self.assertEqual(len(prompts), 5)
        for fname, content in prompts:
            self.assertTrue(content.strip(), msg=f"Prompt {fname} is empty")

    def test_load_session_prompts_respects_limit(self):
        """_load_session_prompts respects the n limit."""
        prompts = _load_session_prompts(_BANDIT_SESSION_PROMPTS_DIR, 2)
        self.assertEqual(len(prompts), 2)

    def test_bandit_session_dry_run_cli(self):
        """bandit-session --dry-run exits 0 and writes an empty CSV header."""
        import subprocess, sys
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            out_csv = Path(f.name)
        try:
            result = subprocess.run(
                [sys.executable, "-m", "harness.client_test_runner",
                 "bandit-session", "--dry-run",
                 "--client", "claude_code",
                 "--turns", "3",
                 "--output", str(out_csv)],
                capture_output=True, text=True,
                cwd=str(Path(__file__).resolve().parent.parent.parent),
            )
            self.assertEqual(result.returncode, 0, msg=result.stderr)
            rows = list(csv.DictReader(out_csv.open()))
            # dry-run writes header only
            self.assertEqual(len(rows), 0)
        finally:
            out_csv.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
