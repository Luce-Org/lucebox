#!/usr/bin/env python3
"""End-to-end GPU-evidence checks for the paired synthetic runner."""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNNER = HERE / "run_qwen36_concurrency.sh"


FAKE_SERVER = """\
#!/usr/bin/env python3
import os
import signal
import sys
import time

if "--version" in sys.argv[1:]:
    print("version: 1 (4cb22cd)")
    raise SystemExit(0)
if "--list-devices" in sys.argv[1:]:
    print(os.environ.get("FAKE_LIST_DEVICES", "Available devices:\\n  ROCm0: Radeon 8060S Graphics"))
    raise SystemExit(0)

line = os.environ.get("FAKE_SERVER_LOG", "")
if line:
    print(line, flush=True)
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
while True:
    time.sleep(60)
"""


class SyntheticRunnerGpuProofTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.bin_dir = self.root / "bin"
        self.bin_dir.mkdir()
        self.model = self.root / "model.gguf"
        self.model.write_bytes(b"fake model")
        self.noop = self.root / "noop.py"
        self.noop.write_text("raise SystemExit(0)\n", encoding="utf-8")
        self.luce = self._write_executable("luce-server", FAKE_SERVER)
        self.llama = self._write_executable("llama-server", FAKE_SERVER)
        self._write_executable("curl", "#!/bin/sh\nsleep 0.1\nexit 0\n")
        self._write_executable(
            "rocminfo", "#!/bin/sh\nprintf '  Name: gfx1151\\n'\n"
        )

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _write_executable(self, name: str, text: str) -> Path:
        path = self.bin_dir / name
        path.write_text(textwrap.dedent(text), encoding="utf-8")
        path.chmod(0o755)
        return path

    def _environment(self, out: Path, variant: str, server_log: str) -> dict[str, str]:
        env = {
            key: value
            for key, value in os.environ.items()
            if not re.match(
                r"^(GGML_|DFLASH_|LUCE_|HIP_|ROCR_|HSA_|LD_PRELOAD$|LD_LIBRARY_PATH$)",
                key,
            )
        }
        env.update(
            PATH=f"{self.bin_dir}{os.pathsep}{env['PATH']}",
            MODEL=str(self.model),
            LUCE_SERVER_BIN=str(self.luce),
            LLAMA_SERVER_BIN=str(self.llama),
            OUT=str(out),
            WORKLOADS="short",
            VARIANTS=variant,
            CLIENTS="2",
            SLOTS="2",
            REPEATS="1",
            GPU_DEVICE="1",
            EXPECTED_GPU_ARCH="gfx1151",
            COOLDOWN_SECONDS="0",
            HEALTH_TIMEOUT_SECONDS="2",
            CLIENT=str(self.noop),
            SUMMARIZER=str(self.noop),
            FAKE_SERVER_LOG=server_log,
        )
        return env

    def _run(self, name: str, variant: str, server_log: str, **extra: str):
        out = self.root / name
        env = self._environment(out, variant, server_log)
        env.update(extra)
        result = subprocess.run(
            [str(RUNNER)], env=env, text=True, capture_output=True, check=False
        )
        return result, out / "short" / "c2" / "r1" / variant

    def test_luce_and_llama_store_independent_runtime_evidence(self) -> None:
        luce_result, luce_case = self._run(
            "luce-ok", "luce-k8", "Device 0: Radeon 8060S Graphics, gfx1151"
        )
        self.assertEqual(luce_result.returncode, 0, luce_result.stderr)
        self.assertIn("gfx1151", (luce_case / "gpu-identity.txt").read_text())
        self.assertIn("gfx1151", (luce_case / "server-gpu-proof.txt").read_text())
        self.assertFalse((luce_case / "llama-list-devices.txt").exists())

        llama_result, llama_case = self._run(
            "llama-ok", "llama", "llama_model_load: offloaded 65/65 layers to GPU"
        )
        self.assertEqual(llama_result.returncode, 0, llama_result.stderr)
        self.assertIn("gfx1151", (llama_case / "gpu-identity.txt").read_text())
        self.assertIn("ROCm0:", (llama_case / "llama-list-devices.txt").read_text())
        self.assertIn(
            str(self.llama),
            (llama_case / "llama-list-devices-command.txt").read_text(),
        )
        llama_command = (llama_case / "server-command.txt").read_text()
        self.assertIn("-ngl all -lv 4", llama_command)
        self.assertIn("--reasoning off --reasoning-format none", llama_command)
        self.assertEqual(
            (llama_case / "server-gpu-proof.txt").read_text().strip(),
            "llama_model_load: offloaded 65/65 layers to GPU",
        )

    def test_llama_rejects_missing_rocm_device_and_partial_or_zero_offload(self) -> None:
        no_device, _ = self._run(
            "llama-no-device",
            "llama",
            "llama_model_load: offloaded 65/65 layers to GPU",
            FAKE_LIST_DEVICES="Available devices:\n  CPU: host",
        )
        self.assertEqual(no_device.returncode, 1)
        self.assertIn("did not expose a ROCm device", no_device.stderr)

        for name, line in (
            ("partial", "llama_model_load: offloaded 64/65 layers to GPU"),
            ("zero", "llama_model_load: offloaded 0/0 layers to GPU"),
        ):
            with self.subTest(name=name):
                result, case = self._run(f"llama-{name}", "llama", line)
                self.assertEqual(result.returncode, 1)
                self.assertIn("did not report a positive full GPU offload", result.stderr)
                self.assertFalse((case / "server-gpu-proof.txt").exists())

    def test_luce_rejects_preflight_runtime_identity_mismatch(self) -> None:
        result, case = self._run(
            "luce-wrong-process", "luce-k8", "Device 0: Radeon AI PRO, gfx1201"
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn("did not report expected GPU architecture gfx1151", result.stderr)
        self.assertIn("gfx1151", (case / "gpu-identity.txt").read_text())
        self.assertEqual((case / "server-gpu-proof.txt").read_text(), "")


if __name__ == "__main__":
    unittest.main()
