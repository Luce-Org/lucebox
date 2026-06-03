"""``lucebox profile`` — thin wrapper around ``luce-bench snapshot``.

The previous incarnation owned its own step registry / audit pipeline /
fingerprint cache. That's all moved into ``lucebench.snapshot``; the
host-side wrapper now just:

  1. probes host facts (LUCEBOX_HOST_* env vars are passed into the container),
  2. picks an output dir under ``$XDG_DATA_HOME/lucebox/profile-snapshots/``,
  3. exec's ``docker exec <container> luce-bench snapshot --level N
     --url <internal-url> --host-info <path> --out-dir <out>``,
  4. streams the subprocess output to the operator's terminal.

If no container is running we bail with a clear hint instead of trying
to bootstrap one — keeping ``profile`` predictable in CI and dry-run
scripts.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import urllib.error
import urllib.request
from dataclasses import asdict
from pathlib import Path
from typing import Any

from rich.console import Console

from lucebox.types import Config

# ── helpers reused by other modules (cli.py imports these) ─────────────────


def _server_base_urls(cfg: Config, base_url: str | None = None) -> list[str]:
    """Candidate URLs for the running lucebox server, host's-eye view.

    Tried in order: explicit override → 127.0.0.1 → host.docker.internal →
    the default docker bridge gateway. Keeps the discovery logic in one
    place so the CLI client launchers and the profile wrapper agree on
    where the server lives.
    """
    if base_url:
        return [base_url.rstrip("/")]
    return [
        f"http://127.0.0.1:{cfg.port}",
        f"http://host.docker.internal:{cfg.port}",
        f"http://172.17.0.1:{cfg.port}",
    ]


def _json_get(url: str, timeout_s: float = 5.0) -> dict[str, Any]:
    """GET ``url`` as JSON. Returns {} on any transport / decode error."""
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as resp:
            return json.loads(resp.read())
    except (OSError, urllib.error.URLError, json.JSONDecodeError):
        return {}


# ── host fact dump → JSON the bench can ingest ─────────────────────────────


def _host_info_payload(cfg: Config) -> dict[str, Any]:
    """Convert our HostFacts (env-driven) into the bench's host-info shape.

    The bench expects the canonical keys returned by
    ``lucebench.hostinfo.probe_host_info`` — cpu_model, nproc, ram_gb,
    gpu_name, gpu_count, vram_gb, gpu_sm, gpu_power_limit_w,
    driver_version, cuda_runtime_version, nvidia_smi_csv. We don't have
    a power limit or CUDA runtime version on the host side, so those
    fields stay None.
    """
    host = cfg.host
    return {
        "cpu_model": None,  # not currently probed by lucebox.sh
        "nproc": host.nproc or None,
        "ram_gb": host.ram_gb or None,
        "gpu_name": host.gpu_name or None,
        "gpu_count": host.gpu_count or None,
        "vram_gb": host.vram_gb or None,
        "gpu_sm": host.gpu_sm or None,
        "gpu_power_limit_w": None,
        "driver_version": host.driver_version or None,
        "cuda_runtime_version": None,
        "nvidia_smi_csv": None,
        # Keep the rest of HostFacts under a vendor field so we don't
        # lose the lucebox-specific bits when the snapshot is replayed.
        "lucebox_host_facts": asdict(host),
    }


def _profile_out_dir() -> Path:
    """Resolve the profile-snapshots root: $XDG_DATA_HOME/lucebox/profile-snapshots."""
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / "lucebox" / "profile-snapshots"


def _container_running(name: str) -> bool:
    """True iff ``docker inspect`` reports the named container as Running."""
    try:
        out = subprocess.check_output(
            ["docker", "inspect", "-f", "{{.State.Running}}", name],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return False
    return out.strip() == "true"


def _detect_url(cfg: Config, override: str | None) -> str:
    """Pick the first /health-answering base URL, or the first candidate."""
    if override:
        return override.rstrip("/")
    for url in _server_base_urls(cfg):
        if _json_get(url + "/health", timeout_s=1.0):
            return url
    # Caller will surface a clearer error from docker exec when the
    # server is genuinely down — we just default to localhost.
    return f"http://127.0.0.1:{cfg.port}"


def run_profile(
    cfg: Config,
    *,
    level: str,
    url: str | None = None,
    console: Console | None = None,
    out_dir: Path | None = None,
    name: str | None = None,
    label: str | None = None,
) -> int:
    """Drive ``docker exec <container> luce-bench snapshot ...`` end-to-end.

    Returns the exit code from the subprocess. ``console`` defaults to
    a fresh ``Console`` when omitted — passing one in lets the host CLI
    keep its themed output stream.

    ``out_dir``, ``name``, and ``label`` are forwarded to ``luce-bench
    snapshot`` when set — used by ``lucebox autotune --sweep`` to pin
    every cell's output under a single sweep dir (``profile-snapshots/
    sweep/cell-N/``) so the corpus stays queryable later by
    ``luce-bench report``.
    """
    console = console or Console()
    if level not in ("level0", "level1", "level2", "level3"):
        console.print(f"[red]Unknown profile level: {level!r} (expected level0..level3)[/red]")
        return 2

    if not _container_running(cfg.container_name):
        console.print(
            f"[red]No running container named {cfg.container_name!r}.[/red]\n"
            "[dim]Hint: run `lucebox start` (or `lucebox serve`) first.[/dim]"
        )
        return 2

    resolved_out = out_dir if out_dir is not None else _profile_out_dir()
    resolved_out.mkdir(parents=True, exist_ok=True)

    base_url = _detect_url(cfg, url)
    # Hand the bench a static host-info JSON so it doesn't have to
    # probe inside the container (where /proc and nvidia-smi point at
    # the container's namespace, not the host's).
    host_info_path = resolved_out / "_host-info.json"
    host_info_path.write_text(json.dumps(_host_info_payload(cfg), indent=2) + "\n")

    cmd = [
        "docker",
        "exec",
        cfg.container_name,
        "luce-bench",
        "snapshot",
        "--level",
        level,
        "--url",
        base_url,
        "--host-info",
        str(host_info_path),
        "--out-dir",
        str(resolved_out),
    ]
    if name is not None:
        cmd += ["--name", name]
    if label is not None:
        cmd += ["--label", label]
    console.print(f"[bold]running:[/bold] {' '.join(shlex.quote(a) for a in cmd)}")
    proc = subprocess.run(cmd, check=False)
    return proc.returncode


# Re-exports so legacy callers (lucebox.cli._detect_server_url, tests)
# still import ``_server_base_urls`` / ``_json_get`` from here.
__all__ = ["_json_get", "_server_base_urls", "run_profile"]
