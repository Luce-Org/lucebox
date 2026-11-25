"""Tests for ``lucebench.hostinfo`` — host fact probe."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from lucebench import hostinfo


def test_probe_host_info_from_file(tmp_path: Path) -> None:
    """``host_info_file`` short-circuits the probes and returns verbatim JSON."""
    payload = {
        "cpu_model": "AMD Ryzen 9 7950X",
        "nproc": 32,
        "ram_gb": 128,
        "gpu_name": "NVIDIA GeForce RTX 5090",
        "gpu_count": 1,
        "vram_gb": 32,
        "gpu_sm": "12.0",
        "gpu_power_limit_w": 600.0,
        "driver_version": "595.71.05",
        "cuda_runtime_version": "12.8",
        "nvidia_smi_csv": "NVIDIA GeForce RTX 5090, 32760 MiB, ...",
    }
    p = tmp_path / "host.json"
    p.write_text(json.dumps(payload))
    assert hostinfo.probe_host_info(p) == payload


def test_probe_host_info_rejects_non_object(tmp_path: Path) -> None:
    p = tmp_path / "host.json"
    p.write_text(json.dumps([1, 2, 3]))
    with pytest.raises(ValueError, match="not a JSON object"):
        hostinfo.probe_host_info(p)


def test_probe_host_info_rejects_bad_json(tmp_path: Path) -> None:
    p = tmp_path / "host.json"
    p.write_text("{not-json")
    with pytest.raises(ValueError, match="could not read host info"):
        hostinfo.probe_host_info(p)


def test_probe_returns_canonical_field_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """All canonical fields are present.

    Back-compat scalars + the v2 HostInfo-shaped fields ride in the same
    dict. The ``gpus`` array is empty when nvidia-smi is missing; the
    ``collected_at`` / ``collector`` / ``source`` metadata is filled
    on every probe (probe-time provenance).
    """
    monkeypatch.setattr(hostinfo, "_run", lambda *_a, **_k: None)
    monkeypatch.setattr(hostinfo, "_probe_ram_gb", lambda: None)
    monkeypatch.setattr(hostinfo, "_probe_cpu", lambda: (None, None))
    monkeypatch.setattr(hostinfo, "_probe_os_pretty", lambda: None)
    monkeypatch.setattr(hostinfo, "_probe_kernel", lambda: None)
    monkeypatch.setattr(hostinfo, "_probe_wsl_version", lambda: None)
    monkeypatch.setattr(hostinfo, "_probe_docker_version", lambda: None)
    monkeypatch.setattr(hostinfo, "_probe_nvidia_ctk_version", lambda: None)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    out = hostinfo.probe_host_info()
    # Back-compat scalars.
    legacy = {
        "cpu_model",
        "nproc",
        "ram_gb",
        "gpu_name",
        "gpu_count",
        "vram_gb",
        "gpu_sm",
        "gpu_power_limit_w",
        "driver_version",
        "cuda_runtime_version",
        "nvidia_smi_csv",
    }
    # v2 HostInfo-shaped fields.
    v2 = {
        "os_pretty",
        "kernel",
        "wsl_version",
        "docker_version",
        "nvidia_driver",
        "nvidia_ctk_version",
        "cuda_visible_devices",
        "gpus",
        "source",
        "collector",
        "collected_at",
    }
    assert legacy.issubset(out.keys()), f"missing legacy keys: {legacy - out.keys()}"
    assert v2.issubset(out.keys()), f"missing v2 keys: {v2 - out.keys()}"
    # Probe-failed values are None except for the always-present meta:
    # source/collector/collected_at + gpus=[].
    always_present = {"source", "collector", "collected_at"}
    for k, v in out.items():
        if k == "gpus":
            assert v == [], f"expected empty gpus list, got {v!r}"
            continue
        if k in always_present:
            assert v, f"expected {k} populated, got {v!r}"
            continue
        assert v is None, f"expected {k} to be None on missing probes, got {v!r}"


def test_probe_wsl_detection(monkeypatch: pytest.MonkeyPatch) -> None:
    """microsoft-standard-WSL2 → "wsl2"; bare "Microsoft" → "wsl1"; else None."""
    real_read_text = Path.read_text

    def wrap(text: str):
        def fake(self: Path, *a, **k):
            if str(self) == "/proc/version":
                return text
            return real_read_text(self, *a, **k)
        return fake

    monkeypatch.setattr(Path, "read_text", wrap("Linux ... 6.6.87.2-microsoft-standard-WSL2 ..."))
    assert hostinfo._probe_wsl_version() == "wsl2"

    monkeypatch.setattr(Path, "read_text", wrap("Linux 4.4.0-Microsoft #1 ..."))
    assert hostinfo._probe_wsl_version() == "wsl1"

    monkeypatch.setattr(Path, "read_text", wrap("Linux 5.15.0-generic ..."))
    assert hostinfo._probe_wsl_version() is None


def test_probe_nvidia_ctk_version_graceful_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``nvidia-ctk --version`` returns None on missing binary."""
    monkeypatch.setattr(hostinfo, "_run", lambda *_a, **_k: None)
    assert hostinfo._probe_nvidia_ctk_version() is None


def test_probe_nvidia_ctk_version_parses_banner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_run(args, timeout_s: float = 5.0):
        if args[:2] == ["nvidia-ctk", "--version"]:
            return "NVIDIA Container Toolkit CLI version 1.16.2"
        return None

    monkeypatch.setattr(hostinfo, "_run", fake_run)
    assert hostinfo._probe_nvidia_ctk_version() == "1.16.2"


def test_probe_multi_gpu_array_populated(monkeypatch: pytest.MonkeyPatch) -> None:
    """Multi-row CSV → one ``gpus`` entry per device with full per-device fields."""
    legacy_csv = (
        "NVIDIA H100, 81920 MiB, 595.71.05, 9.0, 700.00 W\n"
        "NVIDIA H100, 81920 MiB, 595.71.05, 9.0, 700.00 W\n"
    )
    multi_csv = (
        "0, GPU-uuid-1, 00000000:01:00.0, NVIDIA H100, 9.0, 81920 MiB, 700.00 W\n"
        "1, GPU-uuid-2, 00000000:02:00.0, NVIDIA H100, 9.0, 81920 MiB, 700.00 W\n"
    )

    def fake_run(args, timeout_s: float = 5.0):
        if "query-gpu=name,memory.total" in " ".join(args):
            return legacy_csv
        if "query-gpu=index,uuid,pci.bus_id" in " ".join(args):
            return multi_csv
        return None

    monkeypatch.setattr(hostinfo, "_run", fake_run)
    out = hostinfo._probe_nvidia()
    assert out["gpu_count"] == 2
    assert len(out["gpus"]) == 2
    assert out["gpus"][0]["index"] == 0
    assert out["gpus"][0]["uuid"] == "GPU-uuid-1"
    assert out["gpus"][0]["pci_bus_id"] == "00000000:01:00.0"
    assert out["gpus"][0]["name"] == "NVIDIA H100"
    assert out["gpus"][0]["sm"] == "9.0"
    assert out["gpus"][0]["vram_gb"] == 80
    assert out["gpus"][0]["power_limit_w"] == 700
    assert out["gpus"][1]["index"] == 1


def test_host_info_to_canonical_drops_legacy_scalars() -> None:
    """``host_info_to_canonical`` keeps only the HostInfo-shaped keys."""
    full = {
        "cpu_model": "Test",
        "nproc": 8,
        "ram_gb": 32,
        "gpu_name": "Test GPU",
        "gpu_count": 1,
        "vram_gb": 24,
        "gpu_sm": "12.0",
        "gpu_power_limit_w": 175.0,
        "driver_version": "595.71.05",
        "cuda_runtime_version": "12.8",
        "nvidia_smi_csv": "raw csv",
        "os_pretty": "Ubuntu 22.04",
        "kernel": "6.6.87.2-microsoft-standard-WSL2",
        "wsl_version": "wsl2",
        "docker_version": "29.1.3",
        "nvidia_driver": "595.71.05",
        "nvidia_ctk_version": "1.16.2",
        "gpus": [{"index": 0, "name": "Test GPU"}],
        "cuda_visible_devices": None,
        "source": "client-fallback",
        "collector": "luce-bench",
        "collected_at": "2026-05-28T20:31:42Z",
    }
    canonical = hostinfo.host_info_to_canonical(full)
    # No legacy scalars survive.
    for legacy_key in (
        "gpu_name",
        "gpu_count",
        "vram_gb",
        "gpu_sm",
        "gpu_power_limit_w",
        "driver_version",
        "cuda_runtime_version",
        "nvidia_smi_csv",
        "cpu_model",  # cpu_model is in BOTH layers — present in canonical too
    ):
        if legacy_key == "cpu_model":
            assert canonical["cpu_model"] == "Test"
            continue
        assert legacy_key not in canonical
    assert canonical["os_pretty"] == "Ubuntu 22.04"
    assert canonical["wsl_version"] == "wsl2"
    assert canonical["gpus"][0]["name"] == "Test GPU"


def test_probe_cpu_falls_back_to_proc_cpuinfo(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When ``lscpu`` is missing, /proc/cpuinfo is parsed instead."""
    # _run returns None for both lscpu calls.
    monkeypatch.setattr(hostinfo, "_run", lambda *_a, **_k: None)

    cpuinfo = (
        "processor\t: 0\n"
        "model name\t: Test CPU @ 3.5GHz\n"
        "processor\t: 1\n"
        "model name\t: Test CPU @ 3.5GHz\n"
    )
    fake_proc = tmp_path / "cpuinfo"
    fake_proc.write_text(cpuinfo)
    real_read_text = Path.read_text

    def fake_read_text(self: Path, *a: Any, **k: Any) -> str:
        if str(self) == "/proc/cpuinfo":
            return cpuinfo
        return real_read_text(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", fake_read_text)
    model, nproc = hostinfo._probe_cpu()
    assert model == "Test CPU @ 3.5GHz"
    assert nproc == 2


def test_probe_ram_gb_parses_meminfo(monkeypatch: pytest.MonkeyPatch) -> None:
    real_read_text = Path.read_text

    def fake_read_text(self: Path, *a: Any, **k: Any) -> str:
        if str(self) == "/proc/meminfo":
            return "MemTotal:       132999088 kB\nMemFree:         1234 kB\n"
        return real_read_text(self, *a, **k)

    monkeypatch.setattr(Path, "read_text", fake_read_text)
    # 132999088 KiB // (1024*1024) == 126 GiB
    assert hostinfo._probe_ram_gb() == 126


def test_probe_nvidia_parses_csv(monkeypatch: pytest.MonkeyPatch) -> None:
    """When nvidia-smi shows a single GPU, we parse name + vram + driver + SM."""
    csv_value = "NVIDIA GeForce RTX 5090, 32760 MiB, 595.71.05, 12.0, 600.00 W"
    banner = (
        "+-----------------------------------------------------------------------------+\n"
        "| NVIDIA-SMI 595.71.05     Driver Version: 595.71.05    CUDA Version: 12.8   |\n"
        "+-----------------------------------------------------------------------------+\n"
    )

    def fake_run(args: list[str], timeout_s: float = 5.0) -> str | None:
        if args[:2] == ["nvidia-smi", "--query-gpu=name,memory.total,driver_version,compute_cap,power.limit"]:
            return csv_value
        if args == ["nvidia-smi"]:
            return banner
        return None

    monkeypatch.setattr(hostinfo, "_run", fake_run)
    out = hostinfo._probe_nvidia()
    assert out["gpu_name"] == "NVIDIA GeForce RTX 5090"
    assert out["gpu_count"] == 1
    assert out["vram_gb"] == 31  # 32760 MiB // 1024
    assert out["driver_version"] == "595.71.05"
    assert out["gpu_sm"] == "12.0"
    assert out["gpu_power_limit_w"] == pytest.approx(600.0)
    assert out["cuda_runtime_version"] == "12.8"
    assert out["nvidia_smi_csv"] == csv_value


def test_probe_nvidia_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hostinfo, "_run", lambda *_a, **_k: None)
    out = hostinfo._probe_nvidia()
    assert out["gpu_name"] is None
    assert out["gpu_count"] is None
    assert out["vram_gb"] is None
    assert out["driver_version"] is None
    assert out["nvidia_smi_csv"] is None


def test_probe_nvidia_counts_multiple_gpus(monkeypatch: pytest.MonkeyPatch) -> None:
    """Multi-row CSV → ``gpu_count`` matches the row count."""
    csv_value = (
        "NVIDIA H100, 81920 MiB, 595.71.05, 9.0, 700.00 W\n"
        "NVIDIA H100, 81920 MiB, 595.71.05, 9.0, 700.00 W\n"
    )

    def fake_run(args: list[str], timeout_s: float = 5.0) -> str | None:
        if args[0] == "nvidia-smi" and any("query-gpu" in a for a in args):
            return csv_value
        return None

    monkeypatch.setattr(hostinfo, "_run", fake_run)
    out = hostinfo._probe_nvidia()
    assert out["gpu_count"] == 2
    assert out["gpu_name"] == "NVIDIA H100"
