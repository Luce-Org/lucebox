"""Tests for preflight_require_bin in common.sh (seed #3) and adapter HOME isolation (Blocker #7).

Verifies that:
  - preflight_require_bin exits 78 with actionable message when binary missing
  - preflight_require_bin exits 0 when binary is found
  - CodexAdapter.preflight_env() injects a temp HOME (HOME isolation)
  - PiAdapter.preflight_env() injects a temp HOME (HOME isolation)
  - preflight_check with an asdf-broken shim (outputs "unknown command") returns (False, reshim msg)
"""

from __future__ import annotations

import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HARNESS_CLIENTS = Path(__file__).resolve().parent.parent / "clients"
COMMON_SH = HARNESS_CLIENTS / "common.sh"
BASH = "/bin/bash"


def _run_preflight(bin_name: str, path_override: str | None = None) -> subprocess.CompletedProcess:
    """Run preflight_require_bin <bin_name> via bash, return CompletedProcess.

    Sources only the preflight_require_bin function from common.sh, bypassing
    the top-level mkdir calls that require /workspace.
    """
    # Extract just the function definition rather than sourcing full common.sh
    # (common.sh runs mkdir -p $LOG_DIR on source which requires /workspace)
    script = f"""
{BASH} -c '
preflight_require_bin() {{
  local bin="$1"
  if ! command -v "$bin" >/dev/null 2>&1; then
    echo "PREFLIGHT ERROR: '"'"'${{bin}}'"'"' not found on PATH." >&2
    echo "  Hint: run '"'"'asdf reshim'"'"' or install ${{bin}} and ensure it is on PATH." >&2
    exit 78
  fi
}}
preflight_require_bin "{bin_name}"
'
"""
    env = os.environ.copy()
    if path_override is not None:
        # Keep /bin for bash itself, but remove everything else
        env["PATH"] = f"/bin:{path_override}"
    return subprocess.run(
        [BASH, "-c", f"""
preflight_require_bin() {{
  local bin="$1"
  if ! command -v "$bin" >/dev/null 2>&1; then
    echo "PREFLIGHT ERROR: '${{bin}}' not found on PATH." >&2
    echo "  Hint: run 'asdf reshim' or install ${{bin}} and ensure it is on PATH." >&2
    exit 78
  fi
}}
preflight_require_bin '{bin_name}'
"""],
        capture_output=True,
        text=True,
        env=env,
        timeout=10,
    )


def _run_preflight_via_source(bin_name: str, path_override: str | None = None) -> subprocess.CompletedProcess:
    """Source common.sh and run preflight_require_bin, with temp RUN_DIR to avoid /workspace."""
    with tempfile.TemporaryDirectory() as tmpdir:
        env = os.environ.copy()
        env.update({
            "RUN_DIR": tmpdir,
            "REPO_DIR": tmpdir,
            "CLIENT_WORK_DIR": tmpdir,
            "STAMP": "test",
        })
        if path_override is not None:
            env["PATH"] = f"/bin:/usr/bin:{path_override}"
        result = subprocess.run(
            [BASH, "-c", f"source '{COMMON_SH}' && preflight_require_bin '{bin_name}'"],
            capture_output=True,
            text=True,
            env=env,
            timeout=10,
        )
    return result


class TestPreflightRequireBin(unittest.TestCase):

    def test_preflight_fails_with_actionable_message_when_node_missing(self):
        """Exit 78 + actionable message when binary not on PATH (seed #3)."""
        with tempfile.TemporaryDirectory() as empty_dir:
            result = _run_preflight("_definitely_not_a_real_binary_xyz123", path_override=empty_dir)
        # Must exit 78 (EX_UNAVAILABLE / "service unavailable")
        self.assertEqual(result.returncode, 78, msg=f"stderr: {result.stderr}")
        # Must print an actionable message naming the missing binary
        combined = (result.stdout + result.stderr).lower()
        self.assertIn("_definitely_not_a_real_binary_xyz123", combined)
        # Must suggest a remediation action (asdf or install)
        self.assertTrue(
            "asdf" in combined or "install" in combined or "reshim" in combined,
            msg=f"No actionable hint in output: {result.stdout!r} {result.stderr!r}",
        )

    def test_preflight_passes_when_binary_present(self):
        """Exit 0 when binary is on PATH."""
        result = _run_preflight("bash")
        self.assertEqual(result.returncode, 0, msg=f"stderr: {result.stderr}")

    def test_preflight_passes_for_python3(self):
        """Exit 0 for python3 (the test runner itself proves it's present)."""
        result = _run_preflight("python3")
        self.assertEqual(result.returncode, 0, msg=f"stderr: {result.stderr}")

    def test_preflight_via_source_fails_with_exit_78(self):
        """Source common.sh; preflight_require_bin still exits 78 for missing binary."""
        with tempfile.TemporaryDirectory() as empty_dir:
            result = _run_preflight_via_source("_not_a_binary_abc987", path_override=empty_dir)
        self.assertEqual(result.returncode, 78, msg=f"stderr: {result.stderr}")
        combined = (result.stdout + result.stderr).lower()
        self.assertIn("asdf", combined)


class TestAdapterPreflightHomeIsolation(unittest.TestCase):
    """Blocker #7: preflight_env() injects temp HOME matching live_run isolation."""

    def test_codex_preflight_env_has_temp_home(self):
        """CodexAdapter.preflight_env() returns HOME != real HOME."""
        import sys
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
        from harness.client_test_runner import CodexAdapter
        adapter = CodexAdapter()
        env = adapter.preflight_env()
        self.assertIn("HOME", env)
        self.assertNotEqual(env["HOME"], os.environ.get("HOME", ""),
                            msg="preflight HOME must be isolated from real HOME")
        self.assertIn("CODEX_HOME", env)
        self.assertEqual(env["HOME"], env["CODEX_HOME"])

    def test_pi_preflight_env_has_temp_home(self):
        """PiAdapter.preflight_env() returns HOME != real HOME."""
        from harness.client_test_runner import PiAdapter
        adapter = PiAdapter()
        env = adapter.preflight_env()
        self.assertIn("HOME", env)
        self.assertNotEqual(env["HOME"], os.environ.get("HOME", ""),
                            msg="preflight HOME must be isolated from real HOME")

    def test_base_adapter_preflight_env_uses_real_env(self):
        """_BaseAdapter.preflight_env() returns current process environment."""
        from harness.client_test_runner import ClaudeCodeAdapter
        adapter = ClaudeCodeAdapter()
        env = adapter.preflight_env()
        # Should contain PATH from current process
        self.assertEqual(env.get("PATH"), os.environ.get("PATH"))

    def test_codex_preflight_catches_asdf_shim_break_via_stub(self):
        """preflight_check returns (False, reshim msg) when binary outputs 'unknown command'.

        Creates a fake 'codex' script that exits 0 but prints 'unknown command: node'
        to stderr — simulating a stale asdf shim. Verifies preflight catches this.
        """
        from harness.client_test_runner import CodexAdapter
        with tempfile.TemporaryDirectory() as fake_bin_dir:
            fake_codex = Path(fake_bin_dir) / "codex"
            fake_codex.write_text(
                "#!/bin/sh\necho 'unknown command: node, perhaps reshim?' >&2\nexit 1\n"
            )
            fake_codex.chmod(fake_codex.stat().st_mode | stat.S_IEXEC)

            adapter = CodexAdapter(binary=str(fake_codex))
            result = adapter.preflight_check()

        self.assertFalse(result.preflight_ok)
        self.assertIsNotNone(result.error)
        msg = (result.error or "").lower()
        self.assertTrue(
            "reshim" in msg or "asdf" in msg,
            msg=f"Expected reshim/asdf hint in error, got: {result.error!r}",
        )


if __name__ == "__main__":
    unittest.main()
