"""Standalone host fact probe used by the snapshot primitive.

The bench may run inside a Docker container (lucebox stack) or directly
on a host (uvx luce-bench). In both cases we want a single, terse JSON
blob describing the rig — OS, kernel, WSL version, docker/CTK versions,
CPU, RAM, GPU, driver, CUDA runtime. That blob lands in ``host.json``
next to ``props.json`` in every snapshot dir AND alongside each per-area
``<area>.json`` so individual area files self-describe.

The canonical shape mirrors ``/props.host`` (props_schema 4+) so a
snapshot's host block can come from either path — server-side
verbatim, or client-side fallback — and look identical to downstream
consumers (reporter, baseline submission, comparison tables).

Resilient by design: every probe is wrapped so a missing binary
(``lscpu``, ``nvidia-smi``, ``nvidia-ctk``) sets the corresponding
field to ``None`` rather than crashing the snapshot. Callers can also
short-circuit with ``host_info_file=Path(...)`` — the file is parsed as
JSON and returned verbatim. That lets the lucebox host wrapper hand a
pre-probed view into the container instead of re-probing inside it
(where /proc and nvidia-smi may not match the host's reality).
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import subprocess
from pathlib import Path
from typing import Any


def _run(args: list[str], timeout_s: float = 5.0) -> str | None:
    """Run ``args`` and return stripped stdout, or ``None`` on any failure.

    Returns ``None`` for missing binary, non-zero exit, timeout, or any
    OSError — callers turn that into a ``None`` field in the host dict.
    """
    try:
        result = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None
    if result.returncode != 0:
        return None
    out = (result.stdout or "").strip()
    return out or None


def _probe_cpu() -> tuple[str | None, int | None]:
    """Return (cpu_model, nproc) — falls back to /proc/cpuinfo when lscpu is absent."""
    out = _run(["lscpu", "-J"])
    if out:
        try:
            data = json.loads(out)
            entries = data.get("lscpu") if isinstance(data, dict) else None
            if isinstance(entries, list):
                model: str | None = None
                nproc: int | None = None
                # lscpu -J returns a nested tree; flatten one level to find
                # the Model name + CPU(s) leaves.
                def walk(items: list[Any]) -> None:
                    nonlocal model, nproc
                    for it in items:
                        if not isinstance(it, dict):
                            continue
                        field = str(it.get("field") or "").rstrip(":")
                        data_val = it.get("data")
                        if field == "Model name" and isinstance(data_val, str):
                            model = data_val.strip()
                        elif field == "CPU(s)" and isinstance(data_val, str):
                            try:
                                nproc = int(data_val.strip())
                            except ValueError:
                                pass
                        children = it.get("children")
                        if isinstance(children, list):
                            walk(children)

                walk(entries)
                if model is not None or nproc is not None:
                    return model, nproc
        except (json.JSONDecodeError, ValueError):
            pass

    # Fallback: /proc/cpuinfo for cpu_model + count siblings.
    try:
        text = Path("/proc/cpuinfo").read_text()
    except OSError:
        return None, None
    model = None
    count = 0
    for line in text.splitlines():
        if line.startswith("model name") and model is None:
            _, _, value = line.partition(":")
            model = value.strip() or None
        if line.startswith("processor"):
            count += 1
    return model, count or None


def _probe_ram_gb() -> int | None:
    """RAM in whole GB (MemTotal // 1024**2). ``None`` if /proc/meminfo unreadable."""
    try:
        text = Path("/proc/meminfo").read_text()
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith("MemTotal:"):
            parts = line.split()
            if len(parts) >= 2:
                try:
                    kib = int(parts[1])
                except ValueError:
                    return None
                # MemTotal is reported in KiB; we want GiB.
                return kib // (1024 * 1024)
    return None


def _probe_nvidia() -> dict[str, Any]:
    """Return GPU/driver/CUDA facts via ``nvidia-smi`` (or all-None on missing).

    Surfaces two view of the same data:
      * back-compat scalars (``gpu_name`` / ``gpu_count`` / ``vram_gb`` /
        ``gpu_sm`` / ``gpu_power_limit_w`` / ``driver_version``) for the
        existing snapshot-name builder + report.
      * new ``gpus`` list (matching ``HostInfo.gpus``) with one entry per
        device carrying index/uuid/pci_bus_id/name/sm/vram_gb/power_limit_w.
    """
    # Original 5-column query (back-compat).
    csv = _run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version,compute_cap,power.limit",
            "--format=csv,noheader",
        ]
    )
    out: dict[str, Any] = {
        "gpu_name": None,
        "gpu_count": None,
        "vram_gb": None,
        "gpu_sm": None,
        "gpu_power_limit_w": None,
        "driver_version": None,
        "cuda_runtime_version": None,
        "nvidia_smi_csv": None,
        # New v2 fields. Always present (possibly empty) so consumers
        # don't have to branch on existence.
        "gpus": [],
    }
    if not csv:
        return out
    out["nvidia_smi_csv"] = csv
    rows = [line.strip() for line in csv.splitlines() if line.strip()]
    if not rows:
        return out
    out["gpu_count"] = len(rows)
    first = [cell.strip() for cell in rows[0].split(",")]
    # Layout: name, memory.total [MiB], driver_version, compute_cap, power.limit [W]
    if len(first) >= 1:
        out["gpu_name"] = first[0] or None
    if len(first) >= 2:
        mem = first[1].split()[0] if first[1] else ""
        try:
            mib = int(mem)
            out["vram_gb"] = mib // 1024
        except ValueError:
            pass
    if len(first) >= 3:
        out["driver_version"] = first[2] or None
    if len(first) >= 4:
        out["gpu_sm"] = first[3] or None
    if len(first) >= 5:
        plimit = first[4].split()[0] if first[4] else ""
        try:
            out["gpu_power_limit_w"] = float(plimit)
        except ValueError:
            pass

    # CUDA runtime version: nvidia-smi prints it at the top of the
    # default-output banner. Parse the line containing "CUDA Version".
    banner = _run(["nvidia-smi"])
    if banner:
        for line in banner.splitlines():
            if "CUDA Version" in line:
                # Format is typically "| NVIDIA-SMI ... Driver Version: 535.86.10  CUDA Version: 12.2  |"
                marker = "CUDA Version"
                idx = line.find(marker)
                tail = line[idx + len(marker) :].lstrip(": ")
                # Strip any trailing whitespace / pipe / tokens.
                token = tail.split()[0] if tail else ""
                token = token.rstrip("|").strip()
                if token:
                    out["cuda_runtime_version"] = token
                break

    # Multi-GPU enumeration for the HostInfo-shaped block. Re-query with
    # a richer column set (index/uuid/pci.bus_id/name/compute_cap/
    # memory.total/power.limit) so the per-device records match
    # /props.host.gpus[*]. Done in a separate call so the back-compat
    # 5-column branch above stays untouched (and any future change to
    # that branch can't invalidate the multi-GPU array).
    multi_csv = _run(
        [
            "nvidia-smi",
            "--query-gpu=index,uuid,pci.bus_id,name,compute_cap,memory.total,power.limit",
            "--format=csv,noheader",
        ]
    )
    if multi_csv:
        gpus: list[dict[str, Any]] = []
        for line in multi_csv.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = [c.strip() for c in line.split(",")]
            if len(parts) < 7:
                continue
            idx_raw, uuid, pci, name, sm, mem_total, power = parts[:7]
            try:
                idx = int(idx_raw)
            except ValueError:
                idx = None
            try:
                vram_gb = int(mem_total.split()[0]) // 1024
            except (ValueError, IndexError):
                vram_gb = None
            try:
                power_w = int(float(power.split()[0]) + 0.5)
            except (ValueError, IndexError):
                power_w = None
            gpus.append({
                "index": idx,
                "uuid": uuid or None,
                "pci_bus_id": pci or None,
                "name": name or None,
                "sm": sm or None,
                "vram_gb": vram_gb,
                "power_limit_w": power_w,
            })
        out["gpus"] = gpus
    return out


def _probe_os_pretty() -> str | None:
    """PRETTY_NAME from /etc/os-release (e.g. ``Ubuntu 22.04.3 LTS``)."""
    try:
        text = Path("/etc/os-release").read_text()
    except OSError:
        return None
    for line in text.splitlines():
        if line.startswith("PRETTY_NAME="):
            _, _, value = line.partition("=")
            return value.strip().strip('"').strip("'") or None
    return None


def _probe_kernel() -> str | None:
    """``uname -r`` — kernel release."""
    return _run(["uname", "-r"])


def _probe_wsl_version() -> str | None:
    """``"wsl2"`` / ``"wsl1"`` / None — match what lucebox.sh probes.

    The modern WSL2 kernel embeds ``microsoft-standard-WSL2`` in
    /proc/version; legacy WSL1 only said ``Microsoft``. Bare Linux or
    macOS returns None.
    """
    try:
        text = Path("/proc/version").read_text()
    except OSError:
        return None
    if "microsoft-standard-WSL2" in text:
        return "wsl2"
    if "Microsoft" in text:
        return "wsl1"
    return None


def _probe_docker_version() -> str | None:
    """``docker version --format '{{.Server.Version}}'``."""
    return _run(["docker", "version", "--format", "{{.Server.Version}}"])


def _probe_nvidia_ctk_version() -> str | None:
    """Trailing token of ``nvidia-ctk --version`` (e.g. ``1.16.2``).

    Output format is ``NVIDIA Container Toolkit CLI version 1.16.2``;
    take the last whitespace-separated token. Returns None if the tool
    isn't installed.
    """
    raw = _run(["nvidia-ctk", "--version"])
    if not raw:
        return None
    for line in raw.splitlines():
        line = line.strip()
        if "version" in line.lower():
            token = line.split()[-1]
            return token or None
    return None


def _probe_cuda_visible_devices() -> str | None:
    """Mirror of the env var; None means "all GPUs visible"."""
    raw = os.environ.get("CUDA_VISIBLE_DEVICES")
    if raw is None or raw == "":
        return None
    return raw


def probe_host_info(host_info_file: Path | None = None) -> dict[str, Any]:
    """Probe the rig and return a host-info dict.

    Two layers in one dict:

      * back-compat scalars (``cpu_model``, ``nproc``, ``ram_gb``,
        ``gpu_name``, ``gpu_count``, ``vram_gb``, ``gpu_sm``,
        ``gpu_power_limit_w``, ``driver_version``,
        ``cuda_runtime_version``, ``nvidia_smi_csv``) for the snapshot's
        name-derivation logic and the existing report.
      * canonical HostInfo fields (``os_pretty``, ``kernel``,
        ``wsl_version``, ``docker_version``, ``nvidia_driver``,
        ``nvidia_ctk_version``, ``cpu_model``, ``nproc``, ``ram_gb``,
        ``gpus`` array, ``cuda_visible_devices``, ``source``,
        ``collector``, ``collected_at``) mirroring ``/props.host``.

    Each field is ``None`` when its probe failed or is N/A (e.g. no
    GPU). When ``host_info_file`` is supplied, the file is read as JSON
    and returned verbatim — lets the lucebox host wrapper avoid
    double-probing inside a container that can't see the real /proc.
    """
    if host_info_file is not None:
        try:
            data = json.loads(Path(host_info_file).read_text())
        except (OSError, json.JSONDecodeError) as e:
            raise ValueError(f"could not read host info from {host_info_file}: {e}") from e
        if not isinstance(data, dict):
            raise ValueError(f"host info at {host_info_file} is not a JSON object")
        return data

    cpu_model, nproc = _probe_cpu()
    ram_gb = _probe_ram_gb()
    gpu = _probe_nvidia()
    out: dict[str, Any] = {
        "cpu_model": cpu_model,
        "nproc": nproc,
        "ram_gb": ram_gb,
        **gpu,
    }
    # Canonical HostInfo fields. The /props.host shape calls the NVIDIA
    # driver string ``nvidia_driver``; we keep ``driver_version`` as the
    # back-compat alias.
    out["os_pretty"] = _probe_os_pretty()
    out["kernel"] = _probe_kernel()
    out["wsl_version"] = _probe_wsl_version()
    out["docker_version"] = _probe_docker_version()
    out["nvidia_driver"] = out.get("driver_version")
    out["nvidia_ctk_version"] = _probe_nvidia_ctk_version()
    out["cuda_visible_devices"] = _probe_cuda_visible_devices()
    out["collected_at"] = _dt.datetime.now(tz=_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    out["collector"] = "luce-bench"
    # source defaults to "client-fallback" — the caller (snapshot) may
    # overwrite when /props.host is available.
    out.setdefault("source", "client-fallback")
    return out


def host_info_to_canonical(d: dict[str, Any]) -> dict[str, Any]:
    """Project a probe_host_info() dict (or /props.host) onto the
    canonical HostInfo field set.

    Drops the back-compat scalars (gpu_name, vram_gb, …) so the result
    is a clean JSON serialisation of HostInfo. The dropped scalars are
    redundant with the ``gpus[0]`` entry on lucebox-shaped probes.
    """
    canonical_keys = (
        "os_pretty",
        "kernel",
        "wsl_version",
        "docker_version",
        "nvidia_driver",
        "nvidia_ctk_version",
        "cpu_model",
        "nproc",
        "ram_gb",
        "gpus",
        "cuda_visible_devices",
        "source",
        "collector",
        "collected_at",
    )
    out: dict[str, Any] = {k: d.get(k) for k in canonical_keys}
    if out["gpus"] is None:
        out["gpus"] = []
    return out
