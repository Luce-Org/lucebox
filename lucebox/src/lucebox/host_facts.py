"""Read HostFacts from the LUCEBOX_HOST_* env vars that lucebox.sh exports.

We deliberately don't try to detect anything ourselves on the Python side —
inside the container, /proc/meminfo reports the container's view, not the
host's, and nvidia-smi may or may not be available depending on how the
caller invoked us. The host wrapper is the only thing that can see the
truth, and it's already paid for the probe.
"""

from __future__ import annotations

import os
from typing import cast

from lucebox.types import CtkStatus, GpuVendor, HostFacts


def _env_int(key: str, default: int = 0) -> int:
    raw = os.environ.get(key, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_bool(key: str) -> bool:
    return os.environ.get(key, "").strip() in {"1", "true", "yes", "on"}


def from_env() -> HostFacts:
    vendor: GpuVendor = "none"
    raw_vendor = os.environ.get("LUCEBOX_HOST_GPU_VENDOR", "none")
    if raw_vendor in {"nvidia", "amd", "none"}:
        vendor = cast(GpuVendor, raw_vendor)

    ctk: CtkStatus = "none"
    raw_ctk = os.environ.get("LUCEBOX_HOST_HAS_CTK", "none")
    if raw_ctk in {"runtime", "cdi", "installed-unwired", "none"}:
        ctk = cast(CtkStatus, raw_ctk)

    return HostFacts(
        nproc=_env_int("LUCEBOX_HOST_NPROC"),
        ram_gb=_env_int("LUCEBOX_HOST_RAM_GB"),
        gpu_vendor=vendor,
        gpu_name=os.environ.get("LUCEBOX_HOST_GPU_NAME", ""),
        gpu_count=_env_int("LUCEBOX_HOST_GPU_COUNT"),
        vram_gb=_env_int("LUCEBOX_HOST_VRAM_GB"),
        gpu_sm=os.environ.get("LUCEBOX_HOST_GPU_SM", ""),
        driver_version=os.environ.get("LUCEBOX_HOST_DRIVER_VERSION", ""),
        driver_major=_env_int("LUCEBOX_HOST_DRIVER_MAJOR"),
        has_systemd=_env_bool("LUCEBOX_HOST_HAS_SYSTEMD"),
        is_wsl=_env_bool("LUCEBOX_HOST_IS_WSL"),
        has_docker=_env_bool("LUCEBOX_HOST_HAS_DOCKER"),
        docker_version=os.environ.get("LUCEBOX_HOST_DOCKER_VERSION", ""),
        ctk=ctk,
    )
