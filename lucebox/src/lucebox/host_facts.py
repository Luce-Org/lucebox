"""Read HostFacts from the LUCEBOX_HOST_* env vars that lucebox.sh exports.

We deliberately don't try to detect anything ourselves on the Python side —
inside the container, /proc/meminfo reports the container's view, not the
host's, and nvidia-smi/amd-smi may or may not be available depending on how the
caller invoked us. The host wrapper is the only thing that can see the
truth, and it's already paid for the probe.
"""

from __future__ import annotations

import os
from dataclasses import replace
from typing import cast

from lucebox.types import CtkStatus, GpuVendor, HostFacts


def nvidia_variant(host: HostFacts) -> str:
    """Select the published CUDA image matching the detected architecture."""
    architecture = host.nvidia_gpu_arch or (
        host.gpu_sm if host.gpu_vendor == "nvidia" else ""
    )
    name = host.nvidia_gpu_name or (
        host.gpu_name if host.gpu_vendor == "nvidia" else ""
    )
    if architecture == "121" and "GB10" in name.upper():
        return "cuda13"
    if architecture == "120":
        return "cuda128"
    return "cuda12"


def compatible_variant(host: HostFacts, variant: str) -> str:
    """Migrate the former broad ``cuda12`` tag on newer architectures."""
    if variant.casefold() == "cuda12":
        return nvidia_variant(host)
    return variant


def for_variant(host: HostFacts, variant: str) -> HostFacts:
    """Project a full host inventory onto the backend selected by an image.

    The wrapper normally performs this projection before entering a container.
    Keeping the same operation in Python is necessary while switching models:
    a CUDA container on an RTX + Strix machine must be able to evaluate whether
    the already-detected AMD device can run a model before persisting ``rocm``.
    """
    normalized = variant.casefold()
    if "rocm" in normalized or "hip" in normalized:
        if not (host.has_amd_gpu or host.gpu_vendor == "amd"):
            return host
        generic_is_amd = host.gpu_vendor == "amd"
        return replace(
            host,
            gpu_vendor="amd",
            gpu_name=host.amd_gpu_name or (host.gpu_name if generic_is_amd else ""),
            gpu_count=host.amd_gpu_count or (host.gpu_count if generic_is_amd else 0),
            vram_gb=host.amd_vram_gb or (host.vram_gb if generic_is_amd else 0),
            gpu_sm=host.amd_gpu_arch or (host.gpu_sm if generic_is_amd else ""),
        )
    if "cuda" in normalized:
        if not (host.has_nvidia_gpu or host.gpu_vendor == "nvidia"):
            return host
        generic_is_nvidia = host.gpu_vendor == "nvidia"
        return replace(
            host,
            gpu_vendor="nvidia",
            gpu_name=host.nvidia_gpu_name or (host.gpu_name if generic_is_nvidia else ""),
            gpu_count=(
                host.nvidia_gpu_count or (host.gpu_count if generic_is_nvidia else 0)
            ),
            vram_gb=host.nvidia_vram_gb or (host.vram_gb if generic_is_nvidia else 0),
            gpu_sm=host.nvidia_gpu_arch or (host.gpu_sm if generic_is_nvidia else ""),
        )
    return host


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
        has_nvidia_gpu=_env_bool("LUCEBOX_HOST_HAS_NVIDIA_GPU"),
        has_amd_gpu=_env_bool("LUCEBOX_HOST_HAS_AMD_GPU"),
        gpu_name=os.environ.get("LUCEBOX_HOST_GPU_NAME", ""),
        gpu_count=_env_int("LUCEBOX_HOST_GPU_COUNT"),
        vram_gb=_env_int("LUCEBOX_HOST_VRAM_GB"),
        gpu_sm=os.environ.get("LUCEBOX_HOST_GPU_SM", ""),
        driver_version=os.environ.get("LUCEBOX_HOST_DRIVER_VERSION", ""),
        driver_major=_env_int("LUCEBOX_HOST_DRIVER_MAJOR"),
        rocm_version=os.environ.get("LUCEBOX_HOST_ROCM_VERSION", ""),
        has_kfd=_env_bool("LUCEBOX_HOST_HAS_KFD"),
        has_dri=_env_bool("LUCEBOX_HOST_HAS_DRI"),
        has_systemd=_env_bool("LUCEBOX_HOST_HAS_SYSTEMD"),
        is_wsl=_env_bool("LUCEBOX_HOST_IS_WSL"),
        has_docker=_env_bool("LUCEBOX_HOST_HAS_DOCKER"),
        docker_version=os.environ.get("LUCEBOX_HOST_DOCKER_VERSION", ""),
        ctk=ctk,
        nvidia_gpu_name=os.environ.get("LUCEBOX_HOST_NVIDIA_GPU_NAME", ""),
        nvidia_gpu_count=_env_int("LUCEBOX_HOST_NVIDIA_GPU_COUNT"),
        nvidia_vram_gb=_env_int("LUCEBOX_HOST_NVIDIA_VRAM_GB"),
        nvidia_gpu_arch=os.environ.get("LUCEBOX_HOST_NVIDIA_GPU_ARCH", ""),
        nvidia_gpu_list_csv=os.environ.get("LUCEBOX_HOST_NVIDIA_GPU_LIST_CSV", ""),
        nvidia_unified_memory=_env_bool("LUCEBOX_HOST_NVIDIA_UNIFIED_MEMORY"),
        amd_gpu_name=os.environ.get("LUCEBOX_HOST_AMD_GPU_NAME", ""),
        amd_gpu_count=_env_int("LUCEBOX_HOST_AMD_GPU_COUNT"),
        amd_vram_gb=_env_int("LUCEBOX_HOST_AMD_VRAM_GB"),
        amd_gpu_arch=os.environ.get("LUCEBOX_HOST_AMD_GPU_ARCH", ""),
        amd_gpu_list_csv=os.environ.get("LUCEBOX_HOST_AMD_GPU_LIST_CSV", ""),
        hybrid_runtime=_env_bool("LUCEBOX_HOST_HAS_HYBRID_RUNTIME"),
    )
