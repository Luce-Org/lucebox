"""Tests for .github/ci/box_step.sh."""

from __future__ import annotations

import os
import subprocess
import threading
import time
from pathlib import Path

BOX_STEP = Path(__file__).resolve().with_name("box_step.sh")


def _write_fake_proc(
    proc: Path,
    pid: int,
    *,
    state: str = "S",
    ppid: int = 1,
    pgrp: int = 1,
    sid: int = 1,
    cmdline: bytes = b"luce_server\x00--port\x008098\x00",
    environ: bytes | None = None,
) -> Path:
    pdir = proc / str(pid)
    pdir.mkdir(parents=True, exist_ok=True)
    (pdir / "stat").write_text(f"{pid} (fake) {state} {ppid} {pgrp} {sid} 0 0 0\n")
    (pdir / "cmdline").write_bytes(cmdline)
    if environ is not None:
        (pdir / "environ").write_bytes(environ)
    return pdir


def _run_box_step(
    tmp_path: Path,
    script_body: str,
    *,
    kfd: Path,
    proc: Path,
    vmstat: Path,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    script = tmp_path / "step.sh"
    script.write_text(script_body)
    env = os.environ.copy()
    env.update(
        {
            "RUNNER_TEMP": str(tmp_path),
            "RUNNER_TRACKING_ID": "test-job-123",
            "BOX_KFD_PROC": str(kfd),
            "BOX_PROC": str(proc),
            "BOX_VMSTAT": str(vmstat),
            "BOX_QUIET_SECONDS": "0",
            "BOX_POLL_SECONDS": "0.05",
            "BOX_WAIT_POLL_SECONDS": "0.05",
            "BOX_WAIT_MINUTES": "1",
            "BOX_STEP_MINUTES": "1",
        }
    )
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        ["bash", str(BOX_STEP), str(script)],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_exited_kfd_pid_does_not_preempt_step(tmp_path: Path) -> None:
    """A stale KFD dir whose /proc/<pid> is already gone must not self-kill CI."""
    kfd = tmp_path / "kfd"
    proc = tmp_path / "proc"
    vmstat = tmp_path / "vmstat"
    (kfd / "346852").mkdir(parents=True)  # no matching /proc/346852
    proc.mkdir()
    vmstat.write_text("oom_kill 0\n")

    res = _run_box_step(
        tmp_path,
        "sleep 0.2\necho done\n",
        kfd=kfd,
        proc=proc,
        vmstat=vmstat,
    )
    assert res.returncode == 0, res.stderr
    assert res.stdout.strip() == "done"
    assert "Another process started using a GPU" not in res.stdout
    assert "No such file or directory" not in res.stderr


def test_zombie_kfd_pid_does_not_preempt_step(tmp_path: Path) -> None:
    """A zombie /proc/<pid> lingering in KFD must not preempt the step."""
    kfd = tmp_path / "kfd"
    proc = tmp_path / "proc"
    vmstat = tmp_path / "vmstat"
    (kfd / "400001").mkdir(parents=True)
    _write_fake_proc(proc, 400001, state="Z", ppid=1, sid=1)
    vmstat.write_text("oom_kill 0\n")

    res = _run_box_step(
        tmp_path,
        "sleep 0.2\necho ok\n",
        kfd=kfd,
        proc=proc,
        vmstat=vmstat,
    )
    assert res.returncode == 0, res.stderr
    assert res.stdout.strip() == "ok"
    assert "Another process started using a GPU" not in res.stdout


def test_monitor_and_own_job_kfd_pids_do_not_preempt(tmp_path: Path) -> None:
    """rocm-smi monitors and processes with our RUNNER_TRACKING_ID are ignored."""
    kfd = tmp_path / "kfd"
    proc = tmp_path / "proc"
    vmstat = tmp_path / "vmstat"
    (kfd / "400002").mkdir(parents=True)
    (kfd / "400003").mkdir(parents=True)
    _write_fake_proc(
        proc,
        400002,
        cmdline=b"/usr/bin/python3\x00/opt/rocm/bin/rocm-smi\x00--showevents\x00",
    )
    _write_fake_proc(
        proc,
        400003,
        environ=b"PATH=/usr/bin\x00RUNNER_TRACKING_ID=test-job-123\x00",
    )
    vmstat.write_text("oom_kill 0\n")

    res = _run_box_step(
        tmp_path,
        "sleep 0.2\necho ok\n",
        kfd=kfd,
        proc=proc,
        vmstat=vmstat,
    )
    assert res.returncode == 0, res.stderr
    assert res.stdout.strip() == "ok"
    assert "Another process started using a GPU" not in res.stdout


def test_live_external_gpu_process_yields_and_reruns(tmp_path: Path) -> None:
    """A genuine live external GPU process must still preempt and trigger a rerun."""
    kfd = tmp_path / "kfd"
    proc = tmp_path / "proc"
    vmstat = tmp_path / "vmstat"
    kfd.mkdir()
    proc.mkdir()
    vmstat.write_text("oom_kill 0\n")

    runs_file = tmp_path / "runs.log"
    started_flag = tmp_path / "started"

    # Inject a live external GPU process (with another user's unreadable
    # environ) once the step starts, then clear it so box_step.sh yields,
    # waits, and reruns the step to completion.
    def inject_external_process() -> None:
        deadline = time.monotonic() + 5.0
        while not started_flag.exists() and time.monotonic() < deadline:
            time.sleep(0.02)
        pdir = _write_fake_proc(
            proc,
            499999,
            state="R",
            ppid=1,
            sid=1,
            environ=b"USER=otherdev\x00",
        )
        (pdir / "environ").chmod(0o000)
        (kfd / "499999").mkdir()
        time.sleep(0.35)
        (kfd / "499999").rmdir()
        (pdir / "environ").chmod(0o600)

    thread = threading.Thread(target=inject_external_process, daemon=True)
    thread.start()
    try:
        res = _run_box_step(
            tmp_path,
            f'echo run >> "{runs_file}"\n'
            f'if [ ! -f "{started_flag}" ]; then touch "{started_flag}"; sleep 5; fi\n'
            "echo completed\n",
            kfd=kfd,
            proc=proc,
            vmstat=vmstat,
        )
    finally:
        thread.join(timeout=5.0)

    assert res.returncode == 0, res.stderr
    assert "Another process started using a GPU" in res.stdout
    assert "Permission denied" not in res.stderr
    assert "completed" in res.stdout
    assert len(runs_file.read_text().splitlines()) == 2
