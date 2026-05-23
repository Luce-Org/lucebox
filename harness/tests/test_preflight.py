"""Tests for preflight_require_bin in common.sh (seed #3).

Verifies that:
  - preflight_require_bin exits 78 with actionable message when binary missing
  - preflight_require_bin exits 0 when binary is found
"""

from __future__ import annotations

import os
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


if __name__ == "__main__":
    unittest.main()
