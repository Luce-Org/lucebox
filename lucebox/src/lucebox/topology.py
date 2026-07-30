"""Normalized accelerator inventory for placement and tuning.

The host wrapper is the only component that can reliably see every physical
GPU. It exports vendor-specific CSV snapshots; this module turns those opaque
probe strings into a small typed topology. No device is selected by summing
VRAM: the planner must name an explicit placement for every device it uses.
"""

from __future__ import annotations

import csv
import io
import re
from dataclasses import dataclass

from lucebox.types import Config, GpuVendor, HostFacts

_INTEGER_RE = re.compile(r"([0-9]+)")


@dataclass(frozen=True, slots=True)
class GpuDevice:
    vendor: GpuVendor
    index: int
    name: str
    architecture: str
    physical_vram_gb: int
    effective_memory_gb: int
    unified_memory: bool = False

    @property
    def backend(self) -> str:
        return "cuda" if self.vendor == "nvidia" else "hip"

    @property
    def placement_name(self) -> str:
        return f"{self.backend}:{self.index}"

    @property
    def label(self) -> str:
        arch = f", {self.architecture}" if self.architecture else ""
        memory_kind = " shared" if self.unified_memory else " VRAM"
        return f"{self.name or self.vendor} ({self.effective_memory_gb} GB{memory_kind}{arch})"


@dataclass(frozen=True, slots=True)
class HardwareTopology:
    devices: tuple[GpuDevice, ...]
    primary: GpuDevice | None

    @property
    def companions(self) -> tuple[GpuDevice, ...]:
        if self.primary is None:
            return self.devices
        return tuple(device for device in self.devices if device != self.primary)

    @property
    def heterogeneous(self) -> bool:
        return len({device.vendor for device in self.devices}) > 1 or len(self.devices) > 1


def _int_from_cell(value: str) -> int:
    match = _INTEGER_RE.search(value.replace(",", ""))
    return int(match.group(1)) if match else 0


def _rows(raw: str) -> list[list[str]]:
    return [
        [cell.strip() for cell in row]
        for row in csv.reader(io.StringIO(raw))
        if row and any(cell.strip() for cell in row)
    ]


def _nvidia_devices(host: HostFacts) -> list[GpuDevice]:
    devices: list[GpuDevice] = []
    for row in _rows(host.nvidia_gpu_list_csv):
        if len(row) < 6:
            continue
        try:
            index = int(row[0])
        except ValueError:
            continue
        architecture = row[4].replace(".", "")
        memory_gb = _int_from_cell(row[5]) // 1024
        devices.append(
            GpuDevice(
                vendor="nvidia",
                index=index,
                name=row[3],
                architecture=architecture,
                physical_vram_gb=memory_gb,
                effective_memory_gb=memory_gb,
            )
        )
    if not devices and (host.has_nvidia_gpu or host.gpu_vendor == "nvidia"):
        name = host.nvidia_gpu_name
        architecture = host.nvidia_gpu_arch
        memory_gb = host.nvidia_vram_gb
        if host.gpu_vendor == "nvidia":
            name = name or host.gpu_name
            architecture = architecture or host.gpu_sm
            memory_gb = memory_gb or host.vram_gb
        devices.append(
            GpuDevice(
                vendor="nvidia",
                index=0,
                name=name or "NVIDIA GPU",
                architecture=architecture,
                physical_vram_gb=memory_gb,
                effective_memory_gb=memory_gb,
            )
        )
    return devices


def _amd_devices(host: HostFacts) -> list[GpuDevice]:
    devices: list[GpuDevice] = []
    for row in _rows(host.amd_gpu_list_csv):
        # probe_host emits: index, blank, blank, name, gfx arch, MiB, blank
        if len(row) < 6:
            continue
        try:
            index = int(row[0])
        except ValueError:
            continue
        physical_gb = _int_from_cell(row[5]) // 1024
        architecture = row[4]
        # gfx1151 is the Strix Halo integrated GPU. Firmware/driver versions
        # disagree on whether SMI reports only the carve-out or a larger UMA
        # aperture, so the architecture — not the reported VRAM size — is the
        # stable signal that model memory comes from the shared system pool.
        unified = architecture == "gfx1151"
        # Keep 16 GB for the OS, file cache, and CPU-side engine buffers. The
        # HIP allocator performs its own live free-memory check at load time;
        # this number is only the planner's conservative capacity ceiling.
        effective_gb = max(physical_gb, max(0, host.ram_gb - 16)) if unified else physical_gb
        devices.append(
            GpuDevice(
                vendor="amd",
                index=index,
                name=row[3],
                architecture=architecture,
                physical_vram_gb=physical_gb,
                effective_memory_gb=effective_gb,
                unified_memory=unified,
            )
        )
    if not devices and (host.has_amd_gpu or host.gpu_vendor == "amd"):
        name = host.amd_gpu_name
        architecture = host.amd_gpu_arch
        memory_gb = host.amd_vram_gb
        if host.gpu_vendor == "amd":
            name = name or host.gpu_name
            architecture = architecture or host.gpu_sm
            memory_gb = memory_gb or host.vram_gb
        unified = architecture == "gfx1151"
        effective_gb = max(memory_gb, max(0, host.ram_gb - 16)) if unified else memory_gb
        devices.append(
            GpuDevice(
                vendor="amd",
                index=0,
                name=name or "AMD GPU",
                architecture=architecture,
                physical_vram_gb=0 if unified else memory_gb,
                effective_memory_gb=effective_gb,
                unified_memory=unified,
            )
        )
    return devices


def _selected_vendor(cfg: Config) -> GpuVendor:
    variant = cfg.variant.lower()
    if "rocm" in variant or "hip" in variant:
        return "amd"
    if "cuda" in variant:
        return "nvidia"
    return cfg.host.gpu_vendor


def _pick_primary(cfg: Config, devices: list[GpuDevice]) -> GpuDevice | None:
    vendor = _selected_vendor(cfg)
    candidates = [device for device in devices if device.vendor == vendor]
    if not candidates:
        return None

    # The wrapper's selected facts encode policy as well as capacity. In
    # particular, R9700 remains primary over a larger-capacity Strix UMA GPU
    # because the discrete card is substantially faster for a fitting model.
    selected_name = cfg.host.gpu_name
    selected_arch = cfg.host.gpu_sm
    for device in candidates:
        if selected_name and device.name == selected_name:
            return device
    for device in candidates:
        if selected_arch and device.architecture == selected_arch:
            return device
    return max(
        candidates,
        key=lambda device: (
            not device.unified_memory,
            device.physical_vram_gb,
            device.effective_memory_gb,
            -device.index,
        ),
    )


def from_config(cfg: Config) -> HardwareTopology:
    devices = _nvidia_devices(cfg.host) + _amd_devices(cfg.host)
    # Older config snapshots and direct Python callers may only provide the
    # historical generic GPU fields. The selected image variant still gives us
    # enough information to form a single-device topology without pretending a
    # second accelerator exists.
    if not devices and cfg.host.vram_gb > 0:
        vendor = _selected_vendor(cfg)
        if vendor != "none":
            devices.append(
                GpuDevice(
                    vendor=vendor,
                    index=0,
                    name=cfg.host.gpu_name or f"{vendor.upper()} GPU",
                    architecture=cfg.host.gpu_sm,
                    physical_vram_gb=cfg.host.vram_gb,
                    effective_memory_gb=cfg.host.vram_gb,
                )
            )
    devices.sort(key=lambda device: (device.vendor, device.index))
    return HardwareTopology(tuple(devices), _pick_primary(cfg, devices))
