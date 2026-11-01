"""Tests for DflashClient lifecycle and resource cleanup."""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pflash.dflash_client import DflashClient


def create_mock_daemon(ready: bool = True, exit_code: int | None = None,
                       continuous_output: bool = False) -> str:
    """Create a temporary Python executable that mocks the dflash daemon protocol."""
    script = tempfile.NamedTemporaryFile("w", suffix=".py", delete=False)
    if exit_code is not None:
        body = f"""#!/usr/bin/env python3
import sys
sys.exit({exit_code})
"""
    else:
        chatter_code = """
import threading, time
running = True
def chatter():
    i = 0
    while running:
        try:
            sys.stdout.write(f"chatter line {i}\\n")
            sys.stdout.flush()
            i += 1
            time.sleep(0.005)
        except Exception:
            break
threading.Thread(target=chatter, daemon=True).start()
""" if continuous_output else "running = False"
        stop_chatter = "running = False\n" if continuous_output else ""
        body = f"""#!/usr/bin/env python3
import os, struct, sys
{chatter_code}

stream_fd = None
for arg in sys.argv:
    if arg.startswith("--stream-fd="):
        stream_fd = int(arg.split("=", 1)[1])

if {ready}:
    sys.stdout.write("[daemon] ready\\n")
    sys.stdout.flush()

while True:
    line = sys.stdin.readline()
    if not line:
        {stop_chatter}        break
    if stream_fd is not None:
        parts = line.strip().split()
        if len(parts) == 2 and parts[1].isdigit():
            # generate request: simulate returning token 42 followed by -1 sentinel
            os.write(stream_fd, struct.pack("<i", 42))
        os.write(stream_fd, struct.pack("<i", -1))
"""
    script.write(body)
    script.close()
    os.chmod(script.name, 0o755)
    return script.name


def _track_dflash_lifecycle(captured_pipes: list[tuple[int, int]], captured_procs: list[subprocess.Popen]):
    """Context manager to intercept pipe descriptors and child processes in DflashClient.__init__."""
    real_pipe = os.pipe
    real_popen = subprocess.Popen

    def tracked_pipe():
        r, w = real_pipe()
        frame = sys._getframe(1)
        while frame:
            if frame.f_code.co_name == "__init__" and frame.f_code.co_filename.endswith("dflash_client.py"):
                self_obj = frame.f_locals.get("self")
                if isinstance(self_obj, DflashClient) and self_obj.r_pipe == -1:
                    captured_pipes.append((r, w))
                    break
            frame = frame.f_back
        return r, w

    def tracked_popen(*args, **kwargs):
        proc = real_popen(*args, **kwargs)
        cmd = args[0] if args else kwargs.get("args", [])
        if isinstance(cmd, (list, tuple)) and any(str(a).startswith("--stream-fd=") for a in cmd):
            captured_procs.append(proc)
        return proc

    pipe_patch = patch("os.pipe", side_effect=tracked_pipe)
    popen_patch = patch("subprocess.Popen", side_effect=tracked_popen)
    return pipe_patch, popen_patch


class TestDflashClient(unittest.TestCase):
    def test_normal_close_releases_read_pipe(self):
        daemon = create_mock_daemon(ready=True)
        try:
            client = DflashClient(
                bin_path=daemon,
                target_path="dummy",
                draft_path="dummy",
                boot_timeout_s=5.0,
                boot_vram_mib=0,
            )
            r_pipe = client.r_pipe
            self.assertGreaterEqual(r_pipe, 0)
            os.fstat(r_pipe)

            client.close()

            self.assertEqual(client.r_pipe, -1)
            with self.assertRaises(OSError):
                os.fstat(r_pipe)
            self.assertIsNotNone(client.proc.poll())
        finally:
            os.unlink(daemon)

    def test_close_is_idempotent(self):
        daemon = create_mock_daemon(ready=True)
        try:
            client = DflashClient(
                bin_path=daemon,
                target_path="dummy",
                draft_path="dummy",
                boot_timeout_s=5.0,
                boot_vram_mib=0,
            )
            client.close()
            client.close()
            self.assertEqual(client.r_pipe, -1)
            self.assertEqual(client.w_pipe, -1)
        finally:
            os.unlink(daemon)

    def test_init_failure_closes_descriptors(self):
        captured_pipes: list[tuple[int, int]] = []
        captured_procs: list[subprocess.Popen] = []
        pipe_patch, popen_patch = _track_dflash_lifecycle(captured_pipes, captured_procs)
        with pipe_patch, popen_patch:
            with self.assertRaises(FileNotFoundError):
                DflashClient(
                    bin_path="/nonexistent/bin/path/dflash",
                    target_path="dummy",
                    draft_path="dummy",
                    boot_timeout_s=1.0,
                    boot_vram_mib=0,
                )

        self.assertEqual(len(captured_pipes), 1)
        r, w = captured_pipes[0]
        with self.assertRaises(OSError):
            os.fstat(r)
        with self.assertRaises(OSError):
            os.fstat(w)

    def test_readiness_timeout_cleans_up_process_and_descriptors(self):
        daemon = create_mock_daemon(ready=False)
        captured_pipes: list[tuple[int, int]] = []
        captured_procs: list[subprocess.Popen] = []
        pipe_patch, popen_patch = _track_dflash_lifecycle(captured_pipes, captured_procs)
        try:
            with pipe_patch, popen_patch:
                with self.assertRaises(RuntimeError):
                    DflashClient(
                        bin_path=daemon,
                        target_path="dummy",
                        draft_path="dummy",
                        boot_timeout_s=0.3,
                        boot_vram_mib=0,
                    )

            self.assertEqual(len(captured_pipes), 1)
            r, w = captured_pipes[0]
            with self.assertRaises(OSError):
                os.fstat(r)
            with self.assertRaises(OSError):
                os.fstat(w)

            self.assertEqual(len(captured_procs), 1)
            proc = captured_procs[0]
            self.assertIsNotNone(proc.poll())
        finally:
            os.unlink(daemon)

    def test_early_daemon_crash_cleans_up_process_and_descriptors(self):
        daemon = create_mock_daemon(exit_code=42)
        captured_pipes: list[tuple[int, int]] = []
        captured_procs: list[subprocess.Popen] = []
        pipe_patch, popen_patch = _track_dflash_lifecycle(captured_pipes, captured_procs)
        try:
            with pipe_patch, popen_patch:
                with self.assertRaises(RuntimeError) as ctx:
                    DflashClient(
                        bin_path=daemon,
                        target_path="dummy",
                        draft_path="dummy",
                        boot_timeout_s=2.0,
                        boot_vram_mib=0,
                    )

            self.assertIn("dflash daemon exited before it emitted a ready banner", str(ctx.exception))
            self.assertEqual(len(captured_pipes), 1)
            r, w = captured_pipes[0]
            with self.assertRaises(OSError):
                os.fstat(r)
            with self.assertRaises(OSError):
                os.fstat(w)

            self.assertEqual(len(captured_procs), 1)
            proc = captured_procs[0]
            self.assertEqual(proc.poll(), 42)
        finally:
            os.unlink(daemon)

    def test_telemetry_exception_handling(self):
        client = object.__new__(DflashClient)
        client._initial_hip_free_mib = 0

        with patch("pflash.dflash_client._query_nvidia_vram_mib", return_value=None):
            # FileNotFoundError (e.g. rocm-smi not installed)
            with patch("subprocess.check_output", side_effect=FileNotFoundError("not found")):
                self.assertIsNone(client._read_vram_used_mib())

            # CalledProcessError (e.g. command exits non-zero)
            with patch("subprocess.check_output", side_effect=subprocess.CalledProcessError(1, ["rocm-smi"])):
                self.assertIsNone(client._read_vram_used_mib())

            # TimeoutExpired (e.g. command hangs)
            with patch("subprocess.check_output", side_effect=subprocess.TimeoutExpired(["rocm-smi"], 5)):
                self.assertIsNone(client._read_vram_used_mib())

            # Malformed text output (ValueError)
            with patch("subprocess.check_output", return_value=b"Used Memory non_numeric"):
                self.assertIsNone(client._read_vram_used_mib())

    def test_existing_client_behavior_and_context_manager(self):
        daemon = create_mock_daemon(ready=True)
        try:
            with DflashClient(
                bin_path=daemon,
                target_path="dummy",
                draft_path="dummy",
                boot_timeout_s=5.0,
                boot_vram_mib=0,
            ) as client:
                r_pipe = client.r_pipe
                os.fstat(r_pipe)

                client.unpark_draft()
                client.park_draft()
                tokens = client.generate([101, 102], n_gen=1)
                self.assertEqual(tokens, [42])

            self.assertEqual(client.r_pipe, -1)
            with self.assertRaises(OSError):
                os.fstat(r_pipe)
            self.assertIsNotNone(client.proc.poll())
        finally:
            os.unlink(daemon)

    def test_shutdown_while_reader_active(self):
        daemon = create_mock_daemon(ready=True, continuous_output=True)
        try:
            client = DflashClient(
                bin_path=daemon,
                target_path="dummy",
                draft_path="dummy",
                boot_timeout_s=5.0,
                boot_vram_mib=0,
            )
            r_pipe = client.r_pipe
            self.assertGreaterEqual(r_pipe, 0)
            os.fstat(r_pipe)

            # Confirm reader thread is running and stdout is open
            self.assertIsNotNone(client._stdout_thread)
            self.assertTrue(client._stdout_thread.is_alive())
            stdout = client.proc.stdout
            self.assertIsNotNone(stdout)
            self.assertFalse(stdout.closed)

            # Close while reader is actively reading lines
            client.close()

            # Verify thread terminated cleanly without uncaught exceptions
            self.assertFalse(client._stdout_thread.is_alive())
            self.assertEqual(client.r_pipe, -1)
            self.assertIsNotNone(client.proc.poll())
            with self.assertRaises(OSError):
                os.fstat(r_pipe)

            # Verify stdout was closed by the reader thread
            self.assertTrue(stdout.closed)

            # Idempotent second call
            client.close()
        finally:
            os.unlink(daemon)


if __name__ == "__main__":
    unittest.main()
