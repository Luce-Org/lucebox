"""Regression tests for file descriptor lifecycle and subprocess cleanup in chat.py stream_generate."""

import errno
import importlib.util
import multiprocessing as mp
import os
import signal
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch
import unittest
from pathlib import Path
from unittest.mock import patch

CHAT_PATH = Path(__file__).resolve().parents[2] / "examples" / "chat.py"
SPEC = importlib.util.spec_from_file_location("chat_example", CHAT_PATH)
chat = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(chat)


def _run_with_process_group_timeout(fn, timeout_s=5.0):
    """Run fn in a dedicated worker process inside its own process group on POSIX.
    """Run fn in a dedicated worker process inside its own process group.

    If fn blocks (e.g. on pipe buffer deadlock), the entire process group
    is terminated with SIGKILL and the worker process is reaped, preventing
    deadlocks from hanging test runs while ensuring no orphaned processes remain.
    On Windows, runs fn with timeout monitoring in a thread.
    """
    if sys.platform == "win32":
        result = {}

        def _worker():
            try:
                result["payload"] = fn()
                result["status"] = "ok"
            except BaseException as e:
                result["status"] = "err"
                result["payload"] = e

        t = threading.Thread(target=_worker, daemon=True)
        t.start()
        t.join(timeout=timeout_s)
        if t.is_alive():
            return "timed_out", None
        return result.get("status", "err"), result.get("payload")

    """
    ctx = mp.get_context("fork")
    parent_conn, child_conn = ctx.Pipe(duplex=False)

    def _worker():
        parent_conn.close()
        os.setpgrp()
        try:
            res = fn()
            child_conn.send(("ok", res))
        except BaseException as e:
            try:
                child_conn.send(("err", e))
            except Exception:
                try:
                    child_conn.send(("err", RuntimeError(repr(e))))
                except Exception:
                    pass
        finally:
            child_conn.close()

    worker = ctx.Process(target=_worker)

    def _terminate_worker():
        if worker.is_alive():
            if worker.pid is not None:
                try:
                    os.killpg(worker.pid, signal.SIGKILL)
                except OSError:
                    if worker.is_alive():
                        try:
                            worker.kill()
                        except (ProcessLookupError, OSError):
                            pass
                    try:
                        os.killpg(worker.pid, signal.SIGKILL)
                    except OSError:
                        pass
            worker.join()

    try:
        try:
            worker.start()
        finally:
            child_conn.close()

        has_data = parent_conn.poll(timeout_s)
        if not has_data:
            _terminate_worker()
            return "timed_out", None

        try:
            status, payload = parent_conn.recv()
        except EOFError:
            worker.join()
            return "crashed", None

        worker.join(timeout=1.0)
        _terminate_worker()

        return status, payload
    finally:
        parent_conn.close()
        _terminate_worker()


class MockTokenizer:
    def __init__(self, raise_on_id=None):
        self.raise_on_id = raise_on_id

    def decode(self, ids):
        if self.raise_on_id is not None and self.raise_on_id in ids:
            raise RuntimeError(f"Simulated decode failure on token {ids}")
        return "".join(f"[{tok}]" for tok in ids)


@unittest.skipIf(sys.platform == "win32", "stream_generate uses POSIX pass_fds on main")
class ChatStreamFdLifecycleTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.root = Path(self.temp_dir.name)

        # Create mock child script that parses --stream-fd and writes tokens.
        # On Windows, --stream-fd is a Win32 HANDLE that is converted to a CRT fd
        # via msvcrt.open_osfhandle, matching how the C++ test_dflash consumer
        # writes directly to the passed HANDLE.
        # Create mock child script that parses --stream-fd and writes tokens
        self.mock_script = self.root / "mock_dflash.py"
        self.mock_script.write_text(
            "import os, struct, sys, time\n"
            "fd = None\n"
            "for arg in sys.argv:\n"
            "    if arg.startswith('--stream-fd='):\n"
            "        fd = int(arg.split('=', 1)[1])\n"
            "if sys.platform == 'win32' and fd is not None:\n"
            "    import msvcrt\n"
            "    fd = msvcrt.open_osfhandle(fd, os.O_WRONLY)\n"
            "stderr_bytes = int(os.environ.get('MOCK_STDERR_BYTES', '0'))\n"
            "if stderr_bytes > 0:\n"
            "    sys.stderr.write('E' * stderr_bytes)\n"
            "    sys.stderr.flush()\n"
            "tokens = [int(x) for x in os.environ.get('MOCK_TOKENS', '1,2,3').split(',') if x]\n"
            "for tok in tokens:\n"
            "    try:\n"
            "        os.write(fd, struct.pack('<i', tok))\n"
            "    except (BrokenPipeError, OSError):\n"
            "        break\n"
            "    time.sleep(0.005)\n"
        )

        self.prompt_bin = self.root / "prompt.bin"
        self.prompt_bin.write_bytes(b"dummy prompt")
        self.draft = self.root / "draft.gguf"
        self.draft.touch()

        # Track all pipe descriptors and subprocesses allocated during tests
        self.allocated_pipes = []
        self.allocated_procs = []
        self.popen_calls = []
        self._orig_pipe = os.pipe
        self._orig_popen = subprocess.Popen

        def tracked_pipe():
            r, w = self._orig_pipe()
            f = sys._getframe(1)
            while f and (
                "unittest/mock" in f.f_code.co_filename or "unittest\\mock" in f.f_code.co_filename
            ):
                f = f.f_back
            if (
                f
                and f.f_code.co_name == "stream_generate"
                and f.f_code.co_filename.endswith("chat.py")
            ):
                self.allocated_pipes.append((r, w))
            return r, w

        def tracked_popen(*args, **kwargs):
            proc = self._orig_popen(*args, **kwargs)
            f = sys._getframe(1)
            while f and (
                "unittest/mock" in f.f_code.co_filename or "unittest\\mock" in f.f_code.co_filename
            ):
                f = f.f_back
            if (
                f
                and f.f_code.co_name == "stream_generate"
                and f.f_code.co_filename.endswith("chat.py")
            ):
                self.allocated_procs.append(proc)
                self.popen_calls.append((args, kwargs, proc))
            return proc

        self.pipe_patcher = patch.object(os, "pipe", side_effect=tracked_pipe)
        self.pipe_patcher.start()
        self.addCleanup(self.pipe_patcher.stop)
        self.popen_patcher = patch.object(subprocess, "Popen", side_effect=tracked_popen)
        self.popen_patcher.start()
        self.addCleanup(self.popen_patcher.stop)
        self.addCleanup(self._cleanup_dangling_fds)

    def _cleanup_dangling_fds(self):
        """Ensure leaked test descriptors do not pollute subsequent tests."""
        for r, w in self.allocated_pipes:
            for fd in (r, w):
                try:
                    os.close(fd)
                except OSError:
                    pass

    def _call_stream_generate(self, tok=None, tokens="1,2,3", stop_ids=None):
        if tok is None:
            tok = MockTokenizer()
        if stop_ids is None:
            stop_ids = set()

        env = os.environ.copy()
        env["MOCK_TOKENS"] = tokens
        with patch.dict(os.environ, env):
            return chat.stream_generate(
                tok=tok,
                bin_path=Path(sys.executable),
                target=self.mock_script,
                draft=self.draft,
                prompt_bin=self.prompt_bin,
                n_gen=10,
                budget=16,
                stop_ids=stop_ids,
            )

    def test_normal_completion_closes_read_descriptor(self):
        result = self._call_stream_generate(tokens="10,20,30")
        self.assertEqual(result, "[10][20][30]")
        self.assertEqual(len(self.allocated_pipes), 1)
        r, w = self.allocated_pipes[0]

        # Read descriptor MUST be closed
        with self.assertRaises(OSError) as ctx:
            os.fstat(r)
        self.assertEqual(ctx.exception.errno, errno.EBADF)

        # Child process must have been reaped by stream_generate
        self.assertEqual(len(self.allocated_procs), 1)
        proc = self.allocated_procs[0]
        self.assertIsNotNone(
            proc.returncode,
            "Child process was not reaped by stream_generate",
        )

    def test_early_termination_on_stop_token_closes_read_descriptor(self):
        result = self._call_stream_generate(
            tokens="1,2,99,4,5",
            stop_ids={99},
        )
        self.assertEqual(result, "[1][2]")
        self.assertEqual(len(self.allocated_pipes), 1)
        r, w = self.allocated_pipes[0]

        # Read descriptor MUST be closed despite early loop break
        # 999 is stop token; child also emits 30 afterwards
        result = self._call_stream_generate(tokens="10,999,30", stop_ids={999})
        self.assertEqual(result, "[10]")
        self.assertEqual(len(self.allocated_pipes), 1)
        r, _w = self.allocated_pipes[0]

        # Verify read descriptor is closed
        with self.assertRaises(OSError) as ctx:
            os.fstat(r)
        self.assertEqual(ctx.exception.errno, errno.EBADF)

        # Child process must also be reaped by stream_generate
        # Verify child process was reaped by stream_generate before any poll() call
        self.assertEqual(len(self.allocated_procs), 1)
        proc = self.allocated_procs[0]
        self.assertIsNotNone(
            proc.returncode,
            "Child process was not reaped by stream_generate",
        )

    def test_exception_during_reading_closes_read_descriptor(self):
        failing_tok = MockTokenizer(raise_on_id=2)
        with self.assertRaises(RuntimeError):
            self._call_stream_generate(
                tok=failing_tok,
                tokens="1,2,3",
            )

        self.assertEqual(len(self.allocated_pipes), 1)
        r, w = self.allocated_pipes[0]

        # Read descriptor MUST be closed despite unhandled exception
        tok = MockTokenizer(raise_on_id=20)
        with self.assertRaises(RuntimeError):
            self._call_stream_generate(tok=tok, tokens="10,20,30")

        self.assertEqual(len(self.allocated_pipes), 1)
        r, _w = self.allocated_pipes[0]

        with self.assertRaises(OSError) as ctx:
            os.fstat(r)
        self.assertEqual(ctx.exception.errno, errno.EBADF)

        # Child process must also be reaped by stream_generate on exception
        self.assertEqual(len(self.allocated_procs), 1)
        proc = self.allocated_procs[0]
        self.assertIsNotNone(
            proc.returncode,
            "Child process was not reaped by stream_generate",
        )

    def test_repeated_invocations_do_not_leak_descriptors(self):
        proc_fd_dir = Path("/proc/self/fd")
        has_proc_fd = proc_fd_dir.is_dir()

        initial_fds = set(os.listdir(proc_fd_dir)) if has_proc_fd else None

        for turn in range(5):
            self._call_stream_generate(tokens=f"{turn}0,{turn}1")
            self.assertEqual(len(self.allocated_pipes), turn + 1)
            turn_r, _w = self.allocated_pipes[turn]

            # Verify descriptor is closed immediately after each turn,
            # preventing accidental descriptor-number reuse from masking a leak.
            with self.assertRaises(
                OSError,
                msg=f"Turn {turn} descriptor {turn_r} remained open after turn return",
            ) as ctx:
                os.fstat(turn_r)
            self.assertEqual(ctx.exception.errno, errno.EBADF)

        self.assertEqual(len(self.allocated_pipes), 5)
        self.assertEqual(len(self.allocated_procs), 5)
        for proc in self.allocated_procs:
            self.assertIsNotNone(
                proc.returncode,
                "Child process was not reaped by stream_generate",
            )
            self.assertIsNone(proc.stderr, "stderr should be DEVNULL, not an open PIPE")

        if has_proc_fd:
            final_fds = set(os.listdir(proc_fd_dir))
            leaked_fds = final_fds - initial_fds
            self.assertEqual(
                leaked_fds,
                set(),
                f"File descriptors leaked into /proc/self/fd: {leaked_fds}",
            )

    def test_child_producing_substantial_stderr_does_not_deadlock(self):
        # 128 KiB exceeds standard 64 KiB OS pipe buffers; deadlocks if stderr were an unread PIPE.
        # Run with a 5.0s timeout to guarantee bounded execution and clean teardown if a deadlock occurs.
        # Run inside a process group with a 5.0s timeout to guarantee bounded execution and
        # clean teardown if a deadlock were to occur.
        def run_target():
            env = os.environ.copy()
            env["MOCK_STDERR_BYTES"] = "131072"
            env["MOCK_TOKENS"] = "1,2,3"
            orig_popen = subprocess.Popen
            captured = []

            def tracked_popen(*args, **kwargs):
                p = orig_popen(*args, **kwargs)
                captured.append(p)
                return p

            with patch.object(subprocess, "Popen", side_effect=tracked_popen):
                with patch.dict(os.environ, env):
                    result = chat.stream_generate(
                        tok=MockTokenizer(),
                        bin_path=Path(sys.executable),
                        target=self.mock_script,
                        draft=self.draft,
                        prompt_bin=self.prompt_bin,
                        n_gen=10,
                        budget=16,
                        stop_ids=set(),
                    )
            proc = captured[0] if captured else None
            return {
                "result": result,
                "stderr_is_none": proc.stderr is None if proc else False,
                "returncode": proc.returncode if proc else None,
            }

        status, payload = _run_with_process_group_timeout(run_target, timeout_s=5.0)
        self.assertNotEqual(
            status,
            "timed_out",
            "stream_generate deadlocked when child produced substantial stderr output",
        )
        self.assertEqual(status, "ok")
        self.assertEqual(payload["result"], "[1][2][3]")
        self.assertTrue(payload["stderr_is_none"], "stderr was not DEVNULL")
        self.assertIsNotNone(
            payload["returncode"],
            "Child process was not reaped by stream_generate",
        )

    def test_process_launch_failure_cleans_up_both_descriptors(self):
        nonexistent_bin = self.root / "nonexistent_binary_xyz"
        with self.assertRaises(FileNotFoundError):
            chat.stream_generate(
                tok=MockTokenizer(),
                bin_path=nonexistent_bin,
                target=self.mock_script,
                draft=self.draft,
                prompt_bin=self.prompt_bin,
                n_gen=10,
                budget=16,
                stop_ids=set(),
            )

        self.assertEqual(len(self.allocated_pipes), 1)
        r, w = self.allocated_pipes[0]
        with self.assertRaises(OSError) as ctx_r:
            os.fstat(r)
        self.assertEqual(ctx_r.exception.errno, errno.EBADF)
        with self.assertRaises(OSError) as ctx_w:
            os.fstat(w)
        self.assertEqual(ctx_w.exception.errno, errno.EBADF)

    def test_process_launch_failure_closes_second_descriptor_if_first_close_raises(self):
        """Verify process-launch failure closes w even if closing r raises OSError, and preserves original error."""
        nonexistent_bin = self.root / "nonexistent_binary_xyz"
        orig_close = os.close
        first_r_closed = []

        def faulty_close(fd):
            if self.allocated_pipes and fd == self.allocated_pipes[0][0]:
                first_r_closed.append(fd)
                raise OSError(errno.EIO, "Simulated I/O error on closing read descriptor")
            orig_close(fd)

        with patch.object(os, "close", side_effect=faulty_close):
            with self.assertRaises(FileNotFoundError):
                chat.stream_generate(
                    tok=MockTokenizer(),
                    bin_path=nonexistent_bin,
                    target=self.mock_script,
                    draft=self.draft,
                    prompt_bin=self.prompt_bin,
                    n_gen=10,
                    budget=16,
                    stop_ids=set(),
                )

        self.assertEqual(len(self.allocated_pipes), 1)
        r, w = self.allocated_pipes[0]
        self.assertEqual(first_r_closed, [r])
        with self.assertRaises(OSError) as ctx_w:
            os.fstat(w)
        self.assertEqual(ctx_w.exception.errno, errno.EBADF)

    def test_platform_subprocess_configuration_and_arguments(self):
        """Verify platform-specific Popen configuration and --stream-fd argument."""
        result = self._call_stream_generate(tokens="1,2")
        self.assertEqual(result, "[1][2]")
        self.assertEqual(len(self.allocated_pipes), 1)
        r, w = self.allocated_pipes[0]

        self.assertEqual(len(self.popen_calls), 1)
        args, kwargs, proc = self.popen_calls[0]
        cmd = args[0] if args else kwargs.get("args")

        stream_fd_arg = next((arg for arg in cmd if arg.startswith("--stream-fd=")), None)
        self.assertIsNotNone(stream_fd_arg, "Missing --stream-fd argument in cmd")
        stream_fd_val = int(stream_fd_arg.split("=", 1)[1])

        if sys.platform == "win32":
            self.assertNotIn("pass_fds", kwargs, "pass_fds must not be passed on Windows")
            self.assertEqual(kwargs.get("close_fds"), False, "close_fds must be False on Windows")
            self.assertGreater(stream_fd_val, 0, "Win32 handle value must be positive")
        else:
            self.assertIn("pass_fds", kwargs, "pass_fds must be passed on POSIX")
            self.assertEqual(
                kwargs["pass_fds"], (w,), "pass_fds must contain pipe write descriptor"
            )
            self.assertEqual(stream_fd_val, w, "--stream-fd must receive write descriptor on POSIX")

    def test_simulated_windows_invocation(self):
        """Verify Windows branch semantics under simulation regardless of host OS."""
        mock_msvcrt = MagicMock()
        mock_msvcrt.get_osfhandle.return_value = 123456

        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0
        mock_proc.wait.return_value = 0
        mock_proc.returncode = 0
        mock_proc.stderr = None

        captured_calls = []

        def fake_popen(cmd, **kwargs):
            captured_calls.append((cmd, kwargs))
            return mock_proc

        with patch("sys.platform", "win32"):
            with patch.dict(sys.modules, {"msvcrt": mock_msvcrt}):
                with patch.object(os, "set_inheritable") as mock_set_inheritable:
                    with patch.object(os, "read", side_effect=[b""]):
                        with patch.object(subprocess, "Popen", side_effect=fake_popen):
                            res = chat.stream_generate(
                                tok=MockTokenizer(),
                                bin_path=Path(sys.executable),
                                target=self.mock_script,
                                draft=self.draft,
                                prompt_bin=self.prompt_bin,
                                n_gen=10,
                                budget=16,
                                stop_ids=set(),
                            )

        self.assertEqual(res, "")
        mock_set_inheritable.assert_called_once()
        mock_msvcrt.get_osfhandle.assert_called_once()

        self.assertEqual(len(captured_calls), 1)
        cmd, kwargs = captured_calls[0]
        self.assertNotIn("pass_fds", kwargs)
        self.assertEqual(kwargs.get("close_fds"), False)
        self.assertIn("--stream-fd=123456", cmd)

    def test_simulated_posix_invocation(self):
        """Verify POSIX branch semantics under simulation regardless of host OS."""
        mock_proc = MagicMock()
        mock_proc.poll.return_value = 0
        mock_proc.wait.return_value = 0
        mock_proc.returncode = 0
        mock_proc.stderr = None

        captured_calls = []

        def fake_popen(cmd, **kwargs):
            captured_calls.append((cmd, kwargs))
            return mock_proc

        with patch("sys.platform", "linux"):
            with patch.object(os, "read", side_effect=[b""]):
                with patch.object(subprocess, "Popen", side_effect=fake_popen):
                    res = chat.stream_generate(
                        tok=MockTokenizer(),
                        bin_path=Path(sys.executable),
                        target=self.mock_script,
                        draft=self.draft,
                        prompt_bin=self.prompt_bin,
                        n_gen=10,
                        budget=16,
                        stop_ids=set(),
                    )

        self.assertEqual(res, "")
        self.assertEqual(len(captured_calls), 1)
        cmd, kwargs = captured_calls[0]
        self.assertIn("pass_fds", kwargs)
        write_fd = kwargs["pass_fds"][0]
        self.assertIn(f"--stream-fd={write_fd}", cmd)
