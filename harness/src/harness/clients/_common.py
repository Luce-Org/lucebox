"""Shared helpers for harness client launchers.

Each ``harness.clients.<client>`` module exposes a ``launch()`` function with
the same shape (base_url, model, api_key, prompt, interactive, …). The
patterns below capture the bits that repeat: binary resolution, work-dir
setup, exec convention.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

DEFAULT_API_KEY = "sk-lucebox"
DEFAULT_MODEL_ID = "luce-dflash"


def find_bin(name: str, *, env_var: str, work_dir_hint: str | None = None) -> str:
    """Locate a client binary.

    Search order:
      1. ``$<env_var>`` (explicit override)
      2. ``$PATH``
      3. ``$CLIENT_WORK_DIR/<work_dir_hint>`` (test-box convention)

    Raises FileNotFoundError with a clear install hint otherwise.
    """
    explicit = os.environ.get(env_var)
    if explicit and Path(explicit).exists():
        return explicit
    on_path = shutil.which(name)
    if on_path:
        return on_path
    work_dir = os.environ.get("CLIENT_WORK_DIR")
    if work_dir and work_dir_hint:
        candidate = Path(work_dir) / work_dir_hint
        if candidate.exists():
            return str(candidate)
    raise FileNotFoundError(
        f"{name!r} binary not found. Install it or set ${env_var} to its path."
    )


def mktempdir(prefix: str) -> Path:
    """Make a working directory for client config/state. Returns Path."""
    return Path(tempfile.mkdtemp(prefix=f"lucebox-{prefix}-"))


def exec_client(
    argv: list[str],
    env: dict[str, str],
    *,
    interactive: bool,
    timeout: int | None = None,
) -> int:
    """Run a client binary with env, return its exit code.

    Interactive: inherits stdio (TUI works), no timeout.
    Non-interactive: stdin from /dev/null, optional wall-time timeout via
    ``subprocess.run(..., timeout=N)`` — no dependency on the external
    ``timeout`` binary, which isn't guaranteed across base images. On
    timeout we return 124 to match the conventional GNU ``timeout`` exit
    code, so harness scripts that branch on $? see the same value either
    way.
    """
    if interactive:
        return subprocess.run(argv, env=env).returncode
    try:
        return subprocess.run(
            argv, env=env, stdin=subprocess.DEVNULL, timeout=timeout
        ).returncode
    except subprocess.TimeoutExpired:
        return 124
