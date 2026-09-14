from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from harness.qualification.qwen36.profiles import QwenArProfile

_ACCELERATOR_IDENTITIES = {
    "Radeon AI PRO R9700": (
        frozenset({"Radeon AI PRO R9700", "AMD Radeon AI PRO R9700"}),
        "0x7551",
        64,
    ),
    "Strix Halo Radeon 8060S": (
        frozenset({"Strix Halo Radeon 8060S", "AMD Radeon Graphics"}),
        "0x1586",
        40,
    ),
}


def _capture(command: list[str], timeout_seconds: float = 30.0) -> str:
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
        )
    except subprocess.TimeoutExpired as error:
        raise ValueError(f"environment command timed out: {command[0]}") from error
    value = result.stdout.strip()
    if not value:
        raise ValueError(f"environment command produced no output: {command[0]}")
    return value


def _kernel_identity() -> str:
    version = " ".join(
        line.strip() for line in Path("/proc/version").read_text().splitlines() if line.strip()
    )
    if not version:
        raise ValueError("kernel driver identity is unavailable")
    return version


def _first_gpu(payload: dict[str, Any]) -> dict[str, Any]:
    devices = payload.get("gpu_data")
    if not isinstance(devices, list) or len(devices) != 1 or not isinstance(devices[0], dict):
        raise ValueError("amd-smi did not return exactly one selected accelerator")
    return devices[0]


def _selected_live_device(payload: dict[str, Any], visible_device: str) -> dict[str, Any]:
    key = f"card{visible_device}"
    if set(payload) != {key} or not isinstance(payload.get(key), dict):
        raise ValueError(f"ROCm telemetry did not return only the selected accelerator {key}")
    return payload[key]


def _static_identity(
    profile: QwenArProfile, gpu: dict[str, Any]
) -> tuple[dict[str, Any], dict[str, Any], str]:
    asic = gpu.get("asic", {})
    driver = gpu.get("driver", {})
    if not isinstance(asic, dict) or not isinstance(driver, dict):
        raise ValueError("AMD static telemetry identity fields are invalid")
    if asic.get("target_graphics_version") != profile.accelerator.architecture:
        raise ValueError("observed accelerator architecture differs from the profile")
    observed_name = asic.get("market_name")
    expected_identity = _ACCELERATOR_IDENTITIES.get(profile.accelerator.name)
    expected_names = (
        expected_identity[0]
        if expected_identity is not None
        else frozenset({profile.accelerator.name})
    )
    if observed_name not in expected_names:
        raise ValueError("observed accelerator name differs from the profile")
    if expected_identity is not None and (
        asic.get("device_id") != expected_identity[1]
        or asic.get("num_compute_units") != expected_identity[2]
    ):
        raise ValueError("observed accelerator hardware identity differs from the profile")
    return asic, driver, observed_name


def observe_amd_environment(
    profile: QwenArProfile, configure_arguments: list[str]
) -> tuple[dict[str, Any], dict[str, Any]]:
    if profile.telemetry != "amd":
        raise ValueError(f"unsupported required telemetry: {profile.telemetry}")
    static_text = _capture(
        [
            "/opt/rocm/bin/amd-smi",
            "static",
            "-g",
            profile.visible_device,
            "-a",
            "-d",
            "-C",
            "ALL",
            "-l",
            "-P",
            "--json",
        ]
    )
    live_text = _capture(
        [
            "/opt/rocm/bin/rocm-smi",
            "-d",
            profile.visible_device,
            "--showmeminfo",
            "vram",
            "--showpower",
            "--showtemp",
            "--showclocks",
            "--showperflevel",
            "--json",
        ]
    )
    static_payload = json.loads(static_text)
    live_payload = json.loads(live_text)
    if not isinstance(static_payload, dict) or not isinstance(live_payload, dict):
        raise ValueError("AMD telemetry returned a non-object JSON payload")
    gpu = _first_gpu(static_payload)
    asic, driver, observed_name = _static_identity(profile, gpu)
    driver_name = driver.get("name")
    driver_version = driver.get("version")
    if (
        not isinstance(driver_name, str)
        or not driver_name.strip()
        or not isinstance(driver_version, str)
        or not driver_version.strip()
    ):
        raise ValueError("AMD driver identity is unavailable")
    live_device = _selected_live_device(live_payload, profile.visible_device)
    performance_level = live_device.get("Performance Level")
    if not isinstance(performance_level, str) or not performance_level:
        raise ValueError("accelerator performance policy is unavailable")
    hip_version = _capture(["/opt/rocm/bin/hipcc", "--version"])
    hip_lines = hip_version.splitlines()
    if len(hip_lines) < 2:
        raise ValueError("HIP runtime and compiler identity are unavailable")
    cmake_version = _capture(["cmake", "--version"]).splitlines()[0]
    kernel_identity = _kernel_identity()
    driver_identity = f"{driver_name} {driver_version}"
    normalized = {
        "accelerator": {
            "role": profile.accelerator.role,
            "declared_name": profile.accelerator.name,
            "observed_name": observed_name,
            "architecture": asic.get("target_graphics_version"),
            "device_id": asic.get("device_id"),
            "num_compute_units": asic.get("num_compute_units"),
            "visible_device": profile.visible_device,
        },
        "driver": driver_identity,
        "kernel": kernel_identity,
        "runtime": hip_lines[0],
        "compiler": hip_lines[1],
        "cmake": cmake_version,
        "power_profile": profile.power_profile,
        "performance_level": performance_level,
        "build_type": "Release",
        "configure_arguments": configure_arguments,
    }
    raw = {
        "amd_smi_static": static_payload,
        "kernel_driver_identity": kernel_identity,
        "rocm_smi_live": live_payload,
        "hipcc_version": hip_version,
        "cmake_version": cmake_version,
    }
    return normalized, raw


def write_environment_evidence(
    output_dir: Path, normalized: dict[str, Any], raw: dict[str, Any]
) -> None:
    output_dir.joinpath("environment.normalized.json").write_text(
        json.dumps(normalized, indent=2, allow_nan=False) + "\n"
    )
    output_dir.joinpath("environment.raw.json").write_text(
        json.dumps(raw, indent=2, allow_nan=False) + "\n"
    )
