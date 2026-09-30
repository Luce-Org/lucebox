#!/usr/bin/env python3
"""End-to-end launch and GPU-evidence checks for the concurrency runners."""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import textwrap
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNNER = HERE / "run_qwen36_concurrency.sh"
CANONICAL_RUNNER = HERE / "run_qwen36_canonical_concurrency.sh"
AMBIENT_TUNING = re.compile(
    r"^(GGML_|LUCE_|HIP_|ROCR_|HSA_|LD_PRELOAD$|LD_LIBRARY_PATH$)"
)


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
            if not AMBIENT_TUNING.match(key)
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
            "luce-ok",
            "luce-k8",
            "Device 0: Radeon 8060S Graphics, gfx1151",
            CLIENTS="2,4",
            SLOTS="4",
            PREFILL_FIRST_BURST_STEPS="7",
            IDLE_PREFILL_TOKENS="2048",
        )
        self.assertEqual(luce_result.returncode, 0, luce_result.stderr)
        self.assertIn("gfx1151", (luce_case / "gpu-identity.txt").read_text())
        self.assertIn("gfx1151", (luce_case / "server-gpu-proof.txt").read_text())
        self.assertFalse((luce_case / "llama-list-devices.txt").exists())
        luce_command = (luce_case / "server-command.txt").read_text()
        for assignment in (
            "ROCR_VISIBLE_DEVICES=1",
            "LUCE_PREFILL_FIRST_BURST_STEPS=7",
            "LUCE_IDLE_PREFILL_TOKENS=2048",
        ):
            self.assertIn(assignment, luce_command)
        luce_metadata = json.loads((luce_case / "server-metadata.json").read_text())
        self.assertEqual(luce_metadata["rocr_visible_devices"], "1")
        self.assertEqual(luce_metadata["expected_gpu_arch"], "gfx1151")
        self.assertEqual(luce_metadata["prefill_first_burst_steps"], 7)
        self.assertEqual(luce_metadata["idle_prefill_tokens"], 2048)
        self.assertEqual(luce_metadata["client_levels"], [2, 4])
        self.assertEqual(luce_metadata["prompt_offset"], 0)
        c4_metadata = json.loads(
            (luce_case.parents[2] / "c4" / "r1" / "luce-k8" / "server-metadata.json").read_text()
        )
        self.assertEqual(c4_metadata["prompt_offset"], 2)

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
        self.assertIn("ROCR_VISIBLE_DEVICES=1", llama_command)
        self.assertNotIn("LUCE_", llama_command)
        llama_metadata = json.loads((llama_case / "server-metadata.json").read_text())
        self.assertIsNone(llama_metadata["prefill_first_burst_steps"])
        self.assertIsNone(llama_metadata["idle_prefill_tokens"])
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


class CanonicalRunnerLaunchTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.bin_dir = self.root / "bin"
        self.bin_dir.mkdir()
        self.model = self.root / "model.gguf"
        self.model.write_bytes(b"fake model")
        self.draft = self.root / "draft.gguf"
        self.draft.write_bytes(b"fake draft")
        self.noop = self.root / "noop.py"
        self.noop.write_text("raise SystemExit(0)\n", encoding="utf-8")
        self.client_calls = self.root / "client-calls"
        self.client = self.root / "client.py"
        self.client.write_text(
            "import pathlib, sys\n"
            f"with pathlib.Path({str(self.client_calls)!r}).open('a') as f:\n"
            "    f.write(' '.join(sys.argv[1:]) + '\\n')\n",
            encoding="utf-8",
        )
        self.server = self.bin_dir / "luce-server"
        self.server.write_text(FAKE_SERVER, encoding="utf-8")
        self.server.chmod(0o755)
        curl = self.bin_dir / "curl"
        curl.write_text("#!/bin/sh\nsleep 0.1\nexit 0\n", encoding="utf-8")
        curl.chmod(0o755)

    def tearDown(self) -> None:
        self.temp.cleanup()

    def _run(self, name: str, server_log: str, **extra: str):
        out = self.root / name
        env = {
            key: value
            for key, value in os.environ.items()
            if not AMBIENT_TUNING.match(key)
        }
        env.update(
            PATH=f"{self.bin_dir}{os.pathsep}{env['PATH']}",
            MODEL=str(self.model),
            DRAFT_MODEL=str(self.draft),
            SERVER_BIN=str(self.server),
            OUT=str(out),
            SUITES="he-raw",
            VARIANTS="ar,adaptive-ddtree",
            CLIENTS="1",
            GPU_DEVICE="1",
            EXPECTED_GPU_ARCH="gfx1151",
            COOLDOWN_SECONDS="0",
            HEALTH_TIMEOUT_SECONDS="2",
            CLIENT=str(self.client),
            SUMMARIZER=str(self.noop),
            FAKE_SERVER_LOG=server_log,
        )
        env.update(extra)
        result = subprocess.run(
            [str(CANONICAL_RUNNER)], env=env, text=True, capture_output=True, check=False
        )
        return result, out / "he-raw" / "c1" / "r1"

    def test_policy_reaches_every_variant_launch_and_metadata(self) -> None:
        result, cases = self._run(
            "policy",
            "Device 0: Radeon 8060S Graphics, gfx1151",
            PREFILL_FIRST_BURST_STEPS="7",
            IDLE_PREFILL_TOKENS="2048",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        for variant in ("ar", "adaptive-ddtree"):
            with self.subTest(variant=variant):
                case = cases / variant
                command = (case / "server-command.txt").read_text()
                for assignment in (
                    "ROCR_VISIBLE_DEVICES=1",
                    "LUCE_PREFILL_FIRST_BURST_STEPS=7",
                    "LUCE_IDLE_PREFILL_TOKENS=2048",
                ):
                    self.assertIn(assignment, command)
                self.assertEqual(
                    "LUCE_DDTREE_ADAPTIVE=1" in command, variant == "adaptive-ddtree"
                )
                metadata = json.loads((case / "server-metadata.json").read_text())
                self.assertEqual(metadata["rocr_visible_devices"], "1")
                self.assertEqual(metadata["expected_gpu_arch"], "gfx1151")
                self.assertEqual(metadata["prefill_first_burst_steps"], 7)
                self.assertEqual(metadata["idle_prefill_tokens"], 2048)
                self.assertIn("gfx1151", (case / "gpu-identity.txt").read_text())
        self.assertEqual(len(self.client_calls.read_text().splitlines()), 4)

    def test_policy_defaults_are_recorded(self) -> None:
        result, cases = self._run(
            "defaults",
            "Device 0: Radeon 8060S Graphics, gfx1151",
            VARIANTS="ar",
            EXPECTED_GPU_ARCH="",
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        metadata = json.loads((cases / "ar" / "server-metadata.json").read_text())
        self.assertEqual(metadata["prefill_first_burst_steps"], 0)
        self.assertEqual(metadata["idle_prefill_tokens"], 4096)
        self.assertIsNone(metadata["expected_gpu_arch"])

    def test_rejects_server_log_without_expected_arch(self) -> None:
        result, cases = self._run(
            "wrong-arch", "Device 0: Radeon AI PRO, gfx1201", VARIANTS="ar"
        )
        self.assertEqual(result.returncode, 1)
        self.assertIn(
            "server log does not identify expected GPU architecture: gfx1151",
            result.stderr,
        )
        self.assertFalse(self.client_calls.exists())


if __name__ == "__main__":
    unittest.main()
