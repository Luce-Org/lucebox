# Vendored from antoinezambelli/forge-guardrails v0.7.1.
# See dflash/scripts/fixtures/forge_eval/_forge/LICENSE for the upstream MIT.
# Local modifications: import paths rewritten from `forge.X` to relative imports.
"""Hardware detection for GPU capabilities.

detect_hardware() reads total VRAM/unified-memory from a probe ladder:
nvidia-smi (NVIDIA discrete) → AMD sysfs (`/sys/class/drm/card*`).
Used by ServerManager for VRAM tier lookup.

ROCm tooling (rocm-smi) is intentionally not probed — forge's AMD backend
is Vulkan/RADV. See issue #61.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from ..errors import HardwareDetectionError

logger = logging.getLogger(__name__)

# Bits-per-weight for common GGUF quantisation levels.
_QUANT_BPW: dict[str, float] = {
    "Q4_0": 4.0,
    "Q4_K_M": 4.83,
    "Q4_K_S": 4.58,
    "Q5_0": 5.0,
    "Q5_K_M": 5.68,
    "Q5_K_S": 5.52,
    "Q6_K": 6.56,
    "Q8_0": 8.0,
    "F16": 16.0,
}

# AMD PCI vendor ID exposed at /sys/class/drm/card*/device/vendor.
_PCI_VENDOR_AMD = "0x1002"


@dataclass
class HardwareProfile:
    """Detected GPU capabilities (total memory only — a stable value).

    ``memory_kind`` distinguishes discrete VRAM (NVIDIA, discrete AMD) from
    unified system RAM carved out for the GPU (Strix Halo / Ryzen AI 300).
    """

    gpu_name: str
    vram_total_mb: int
    gpu_vendor: str = "nvidia"
    memory_kind: Literal["discrete", "unified"] = "discrete"

    @property
    def vram_total_gb(self) -> float:
        return self.vram_total_mb / 1024


def detect_hardware() -> HardwareProfile | None:
    """Auto-detect GPU. Returns None if no probe succeeds.

    Probe order:
      1. nvidia-smi (NVIDIA discrete)
      2. AMD sysfs (`/sys/class/drm/card*/device/{vendor,mem_info_vram_total}`)

    On total failure, logs a single WARN listing what was tried.
    """
    attempted: list[str] = []

    nvidia = _detect_nvidia(attempted)
    if nvidia is not None:
        return nvidia

    amd = _detect_amd_sysfs(attempted)
    if amd is not None:
        return amd

    logger.warning(
        "GPU detection failed; all probes returned no result. Attempted: %s. "
        "Downstream Ollama tier budget will fall back to 4096 tokens.",
        "; ".join(attempted),
    )
    return None


def _detect_nvidia(attempted: list[str]) -> HardwareProfile | None:
    """nvidia-smi probe. Appends a status string to *attempted*."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
    except FileNotFoundError:
        attempted.append("nvidia-smi: not installed")
        return None
    except subprocess.TimeoutExpired:
        attempted.append("nvidia-smi: timeout")
        return None

    if result.returncode != 0:
        attempted.append(f"nvidia-smi: exit {result.returncode}")
        return None

    try:
        line = result.stdout.strip().split("\n")[0]
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 2:
            raise ValueError(f"Expected 2 CSV fields, got {len(parts)}: {line!r}")

        return HardwareProfile(
            gpu_name=parts[0],
            vram_total_mb=int(parts[1]),
            gpu_vendor="nvidia",
            memory_kind="discrete",
        )
    except (ValueError, IndexError) as exc:
        raise HardwareDetectionError(exc) from exc


def _detect_amd_sysfs(attempted: list[str]) -> HardwareProfile | None:
    """AMD sysfs probe — reads /sys/class/drm/card*/device/.

    Iterates `/sys/class/drm/card*` (skipping render nodes), checks
    ``device/vendor`` for AMD, reads ``device/mem_info_vram_total`` for size
    in bytes, and pulls a name from ``device/uevent`` (PCI_ID / DRIVER) when
    available.
    """
    drm_root = Path("/sys/class/drm")
    if not drm_root.exists():
        attempted.append("amd-sysfs: /sys/class/drm missing")
        return None

    cards = sorted(p for p in drm_root.glob("card*") if p.name.split("card", 1)[-1].isdigit())
    if not cards:
        attempted.append("amd-sysfs: no card* entries")
        return None

    for card in cards:
        vendor_file = card / "device" / "vendor"
        vram_file = card / "device" / "mem_info_vram_total"
        if not vendor_file.exists() or not vram_file.exists():
            continue

        vendor = vendor_file.read_text().strip()
        if vendor != _PCI_VENDOR_AMD:
            continue

        try:
            vram_bytes = int(vram_file.read_text().strip())
        except ValueError as exc:
            raise HardwareDetectionError(exc) from exc

        gpu_name = _amd_gpu_name(card) or f"AMD GPU ({card.name})"
        # Sysfs-reported memory on AMD APU/Strix-Halo class is the BIOS-carved
        # chunk of unified system RAM (not separate VRAM).
        return HardwareProfile(
            gpu_name=gpu_name,
            vram_total_mb=vram_bytes // (1024 * 1024),
            gpu_vendor="amd",
            memory_kind="unified",
        )

    attempted.append("amd-sysfs: no AMD card with mem_info_vram_total")
    return None


def _amd_gpu_name(card: Path) -> str | None:
    """Best-effort human name for an AMD card from sysfs uevent."""
    uevent = card / "device" / "uevent"
    if not uevent.exists():
        return None
    try:
        for line in uevent.read_text().splitlines():
            if line.startswith("PCI_ID="):
                return f"AMD GPU [{line.split('=', 1)[1]}]"
    except OSError:
        return None
    return None
