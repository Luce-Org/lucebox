"""Subprocess client for the patched dflash daemon (with park/unpark commands)."""
from __future__ import annotations
import ctypes
import logging
import os
import struct
import subprocess
import sys
import tempfile
import threading
import time
from typing import Optional

from . import config

log = logging.getLogger(__name__)

_READY_MARKERS = (
    "[daemon] ready",
    "[qwen3-daemon] ready",
    "[laguna-daemon] ready",
    "[gemma4-daemon] ready",
)


def _query_nvidia_vram_mib() -> Optional[int]:
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL, timeout=5)
        return int(out.decode().splitlines()[0])
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired, ValueError):
        return None


def _hip_free_mib() -> int:
    """Free GPU memory in MiB via hipMemGetInfo.

    Reliable on AMD unified-memory APUs (Strix Halo, Phoenix) where rocm-smi
    only reports the small dedicated VRAM segment (~512 MB on gfx1151) rather
    than the full shared-memory pool used by the GPU.  Returns 0 on any error
    so callers can treat 0 as "not available".
    """
    try:
        hip = ctypes.CDLL("libamdhip64.so")
        free = ctypes.c_size_t(0)
        total = ctypes.c_size_t(0)
        hip.hipMemGetInfo(ctypes.byref(free), ctypes.byref(total))
        return int(free.value // (1024 * 1024))
    except Exception:
        return 0


class DflashClient:
    def __init__(self, bin_path: str, target_path: str, draft_path: str,
                 max_ctx: int = 16384, ddtree_budget: int = 16,
                 *,
                 ddtree_temp: Optional[float] = None,
                 chain_seed: bool = True,
                 fa_window: Optional[int] = None,
                 kv_tq3: Optional[bool] = None,
                 lm_head_fix: Optional[bool] = None,
                 boot_timeout_s: float = 60.0,
                 boot_vram_mib: int = 18000):
        """Spawn the patched dflash daemon as a subprocess.

        Defaults for fa_window / kv_tq3 / lm_head_fix come from
        ``config.DFLASH_REQUIRED_ENV`` and are the only flags pflash relies on.
        Override per-call only when you know what you're doing.
        """
        self.bin_path = bin_path
        self.target_path = target_path
        self.draft_path = draft_path
        self.max_ctx = max_ctx
        self._ready_event = threading.Event()
        self._stdout_tail: list[str] = []
        env_overrides = {
            "DFLASH27B_FA_WINDOW": str(0 if fa_window is None else fa_window),
            "DFLASH27B_KV_TQ3": "1" if (kv_tq3 if kv_tq3 is not None else True) else "0",
            "DFLASH27B_LM_HEAD_FIX": "1" if lm_head_fix else "0",
        }
        env = {**os.environ, **config.DFLASH_REQUIRED_ENV, **env_overrides}
        bin_dir = os.path.dirname(os.path.abspath(bin_path))
        if sys.platform == "win32":
            # Windows uses PATH for DLL search rather than LD_LIBRARY_PATH.
            path_extra = [os.path.join(bin_dir, "bin"), bin_dir]
            env["PATH"] = os.pathsep.join(
                path_extra + ([env["PATH"]] if env.get("PATH") else []))
        else:
            ld_extra = [bin_dir, os.path.join(bin_dir, "bin")]
            env["LD_LIBRARY_PATH"] = ":".join(
                ld_extra + ([env["LD_LIBRARY_PATH"]] if env.get("LD_LIBRARY_PATH") else []))
        self.r_pipe, self.w_pipe = os.pipe()
        # subprocess.Popen pass_fds is POSIX-only. On Windows, mark the pipe
        # write-end inheritable, convert it to a Win32 HANDLE, and pass that
        # handle value as --stream-fd; the child inherits via close_fds=False.
        if sys.platform == "win32":
            import msvcrt
            os.set_inheritable(self.w_pipe, True)
            stream_fd_val = int(msvcrt.get_osfhandle(self.w_pipe))
        else:
            stream_fd_val = self.w_pipe
        cmd = [bin_path, target_path, draft_path,
               "--daemon", "--fast-rollback", "--ddtree",
               f"--ddtree-budget={ddtree_budget}",
               f"--max-ctx={max_ctx}",
               f"--stream-fd={stream_fd_val}"]
        if ddtree_temp is not None:
            cmd.append(f"--ddtree-temp={ddtree_temp}")
        if not chain_seed:
            cmd.append("--ddtree-no-chain-seed")
        log.info("spawning dflash daemon: %s", " ".join(cmd))
        # Capture free-memory baseline before spawning so _wait_until_loaded can
        # measure the delta on AMD unified-memory APUs (hipMemGetInfo approach).
        self._initial_hip_free_mib: int = _hip_free_mib()
        if sys.platform == "win32":
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE,
                                         close_fds=False, env=env)
        else:
            self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE,
                                         pass_fds=(self.w_pipe,), env=env)
        self._start_stdout_reader()
        os.close(self.w_pipe)
        self._wait_until_loaded(timeout=boot_timeout_s, vram_mib=boot_vram_mib)
        # Park draft by default; user calls unpark when needed
        self._send("park draft\n")

    def _start_stdout_reader(self):
        def _reader():
            if self.proc.stdout is None:
                return
            mirror_stdout = True
            for raw in iter(self.proc.stdout.readline, b""):
                try:
                    line = raw.decode(errors="replace")
                except AttributeError:
                    line = str(raw)
                self._stdout_tail.append(line.rstrip())
                del self._stdout_tail[:-20]
                if any(marker in line for marker in _READY_MARKERS):
                    self._ready_event.set()
                if mirror_stdout:
                    try:
                        sys.stdout.write(line)
                        sys.stdout.flush()
                    except (BrokenPipeError, OSError, ValueError):
                        mirror_stdout = False

        threading.Thread(target=_reader, name="dflash-daemon-stdout", daemon=True).start()

    def _read_vram_used_mib(self) -> Optional[int]:
        """GPU memory allocated since daemon spawn, in MiB.

        Priority order:
        1. hipMemGetInfo delta — works on AMD unified-memory APUs (Strix Halo,
           Phoenix) where rocm-smi only reports the tiny dedicated segment.
        2. rocm-smi — fallback for AMD discrete GPUs (reports total used VRAM).
        3. nvidia-smi — NVIDIA GPUs.
        Returns None when no backend is available.
        """
        # AMD UMA: hipMemGetInfo free-delta is the only reliable signal.
        if self._initial_hip_free_mib > 0:
            current = _hip_free_mib()
            if current > 0:
                delta = self._initial_hip_free_mib - current
                if delta > 0:
                    return delta
        # AMD discrete: rocm-smi total used (not delta).
        try:
            out = subprocess.check_output(
                ["rocm-smi", "--showmeminfo", "vram", "--noheader"],
                stderr=subprocess.DEVNULL, timeout=5)
            for line in out.decode().splitlines():
                if "Used Memory" in line:
                    return int(line.split()[-1]) // (1024 * 1024)
        except (FileNotFoundError, subprocess.CalledProcessError,
                subprocess.TimeoutExpired, (ValueError, StopIteration)):
            pass
        # NVIDIA.
        return _query_nvidia_vram_mib()

    def _wait_until_loaded(self, timeout: float = 60.0, vram_mib: int = 18000):
        """Block until the daemon emits its explicit ready banner.

        The memory telemetry helpers are kept for diagnostics, but readiness is
        driven by the daemon state rather than by GPU memory usage. This avoids
        false timeouts when the first GPU does not host the target model.
        """
        boot = time.time()
        while time.time() - boot < timeout:
            if self._ready_event.wait(timeout=0.1):
                return
            if self.proc.poll() is not None:
                tail = "\n".join(self._stdout_tail[-10:])
                detail = f"\nLast daemon stdout:\n{tail}" if tail else ""
                raise RuntimeError(
                    "dflash daemon exited before it emitted a ready banner. "
                    "Check the daemon's stderr." + detail)
            time.sleep(0.9)
        tail = "\n".join(self._stdout_tail[-10:])
        stdout_detail = f"\nLast daemon stdout:\n{tail}" if tail else ""
        last_used = self._read_vram_used_mib()
        memory_detail = (
            f" Last observed GPU memory signal: {last_used} MiB "
            f"(legacy threshold was {vram_mib} MiB)."
            if last_used is not None else "")
        raise RuntimeError(
            f"dflash daemon did not emit a ready banner within {timeout:.0f}s."
            f"{memory_detail} Check the daemon's stderr." + stdout_detail)

    def _send(self, cmd: str):
        self.proc.stdin.write(cmd.encode())
        self.proc.stdin.flush()
        # Read until -1 sentinel
        while True:
            b = os.read(self.r_pipe, 4)
            if not b or len(b) < 4:
                break
            if struct.unpack("<i", b)[0] == -1:
                break

    def free_drafter(self): self._send("free drafter\n")
    def park_draft(self):    self._send("park draft\n")
    def unpark_draft(self):  self._send("unpark draft\n")
    def park_target(self):   self._send("park target\n")
    def unpark_target(self): self._send("unpark target\n")

    def compress(self, prompt_ids: list[int], keep_ratio: float, drafter_gguf: str,
                 drafter_arch: str = "qwen3-0.6b") -> list[int]:
        """C++ drafter score+compress via daemon. Returns compressed token ids.

        Daemon command: compress <bin> <keep_x1000> <drafter_gguf> <drafter_arch>
        """
        fd, path = tempfile.mkstemp(suffix=".bin")
        with os.fdopen(fd, "wb") as f:
            for t in prompt_ids:
                f.write(struct.pack("<i", int(t)))
        keep_x1000 = int(round(keep_ratio * 1000))
        self.proc.stdin.write(f"compress {path} {keep_x1000} {drafter_gguf} {drafter_arch}\n".encode())
        self.proc.stdin.flush()
        toks = []
        while True:
            b = os.read(self.r_pipe, 4)
            if not b or len(b) < 4:
                break
            v = struct.unpack("<i", b)[0]
            if v == -1:
                break
            toks.append(v)
        os.unlink(path)
        return toks

    def generate(self, prompt_ids: list[int], n_gen: int) -> list[int]:
        """Send prompt + n_gen request, return generated token ids."""
        fd, path = tempfile.mkstemp(suffix=".bin")
        with os.fdopen(fd, "wb") as f:
            for t in prompt_ids:
                f.write(struct.pack("<i", int(t)))
        self.proc.stdin.write(f"{path} {n_gen}\n".encode())
        self.proc.stdin.flush()
        toks = []
        while True:
            b = os.read(self.r_pipe, 4)
            if not b or len(b) < 4:
                break
            v = struct.unpack("<i", b)[0]
            if v == -1:
                break
            toks.append(v)
        os.unlink(path)
        return toks

    def close(self, timeout: float = 5.0):
        try:
            self.proc.stdin.close()
        except Exception:
            pass
        try:
            self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self.proc.kill()
