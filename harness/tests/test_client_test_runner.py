#!/usr/bin/env python3
"""Regression tests for client harness installation reporting."""

from __future__ import annotations

import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest import mock

from harness import client_test_runner as runner


class OmpInstallTests(unittest.TestCase):
    def test_download_failure_returns_structured_install_result(self) -> None:
        spec = runner.CLIENTS["omp"]

        with tempfile.TemporaryDirectory() as temp_dir:
            with (
                mock.patch.object(
                    runner.urllib.request,
                    "urlopen",
                    side_effect=urllib.error.URLError("offline"),
                ) as urlopen,
                mock.patch.object(runner, "run_cmd") as run_cmd,
            ):
                result = runner.install_client(Path(temp_dir), spec)

        urlopen.assert_called_once_with(spec.package, timeout=60)
        run_cmd.assert_not_called()
        self.assertFalse(result["ok"])
        self.assertEqual(result["returncode"], 1)
        self.assertEqual(result["cmd"], ["download", spec.package])
        self.assertIn("offline", result["output_tail"])
        self.assertEqual(result["client"], "omp")
        self.assertEqual(result["installer"], "omp")
        self.assertFalse(result["binary_exists"])
        self.assertFalse(result["version"]["ok"])


if __name__ == "__main__":
    unittest.main()
