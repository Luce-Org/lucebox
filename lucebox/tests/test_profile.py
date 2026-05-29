"""Tests for the collapsed ``lucebox profile`` wrapper.

The profile module is now a thin shim over ``luce-bench snapshot``: it
probes host facts, picks an output dir, and exec's ``docker exec``. The
tests below pin the wrapper contract — no behavior tests of the bench
itself (those live in luce-bench's own test suite).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
from lucebox.types import Config, HostFacts

from lucebox import profile


def _cfg(tmp_path: Path) -> Config:
    """Build a Config with a deterministic models_dir + sentinel host facts."""
    return Config(
        models_dir=tmp_path / "models",
        host=HostFacts(
            gpu_vendor="nvidia",
            gpu_name="Test GPU 5090",
            vram_gb=32,
            nproc=16,
            ram_gb=64,
            driver_version="595.71.05",
            gpu_sm="12.0",
            gpu_count=1,
        ),
    )


def test_server_base_urls_includes_docker_host_route(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    urls = profile._server_base_urls(cfg)
    assert urls[0] == f"http://127.0.0.1:{cfg.port}"
    assert any("host.docker.internal" in u for u in urls)
    assert any("172.17.0.1" in u for u in urls)


def test_server_base_urls_honors_override(tmp_path: Path) -> None:
    cfg = _cfg(tmp_path)
    assert profile._server_base_urls(cfg, "http://example/") == ["http://example"]


def test_host_info_payload_carries_canonical_keys(tmp_path: Path) -> None:
    """The payload handed to luce-bench matches the bench's expected schema."""
    cfg = _cfg(tmp_path)
    payload = profile._host_info_payload(cfg)
    expected = {
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
        "lucebox_host_facts",
    }
    assert expected.issubset(payload.keys())
    assert payload["gpu_name"] == "Test GPU 5090"
    assert payload["vram_gb"] == 32


def test_run_profile_rejects_unknown_level(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _cfg(tmp_path)
    rc = profile.run_profile(cfg, level="level42")
    assert rc == 2


def test_run_profile_errors_when_container_not_running(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Clear error when there's no container — the wrapper must NOT try to boot one."""
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(profile, "_container_running", lambda name: False)
    rc = profile.run_profile(cfg, level="level1")
    assert rc == 2


def test_run_profile_exec_docker_exec_with_expected_args(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Happy path: builds the docker exec argv and writes a host-info file."""
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(profile, "_container_running", lambda name: True)
    monkeypatch.setattr(profile, "_json_get", lambda url, timeout_s=5.0: {})
    # Pin the output dir under tmp_path so the test is hermetic.
    monkeypatch.setattr(profile, "_profile_out_dir", lambda: tmp_path / "snaps")

    invocations: list[list[str]] = []

    def fake_run(cmd: list[str], check: bool = False) -> subprocess.CompletedProcess[Any]:
        invocations.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    rc = profile.run_profile(cfg, level="level1", url="http://127.0.0.1:8080")
    assert rc == 0
    assert len(invocations) == 1
    cmd = invocations[0]
    assert cmd[:3] == ["docker", "exec", cfg.container_name]
    assert "luce-bench" in cmd
    assert "snapshot" in cmd
    assert "--level" in cmd
    i = cmd.index("--level")
    assert cmd[i + 1] == "level1"
    # host-info file was written.
    host_info_path = tmp_path / "snaps" / "_host-info.json"
    assert host_info_path.exists()
    payload = json.loads(host_info_path.read_text())
    assert payload["gpu_name"] == "Test GPU 5090"


def test_run_profile_passes_url_override_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(profile, "_container_running", lambda name: True)
    monkeypatch.setattr(profile, "_profile_out_dir", lambda: tmp_path / "snaps")
    captured: list[list[str]] = []

    def fake_run(cmd: list[str], check: bool = False) -> subprocess.CompletedProcess[Any]:
        captured.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    rc = profile.run_profile(cfg, level="level2", url="http://example:9000")
    assert rc == 0
    cmd = captured[0]
    i = cmd.index("--url")
    assert cmd[i + 1] == "http://example:9000"


def test_run_profile_returns_subprocess_rc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg = _cfg(tmp_path)
    monkeypatch.setattr(profile, "_container_running", lambda name: True)
    monkeypatch.setattr(profile, "_profile_out_dir", lambda: tmp_path / "snaps")

    def fake_run(cmd: list[str], check: bool = False) -> subprocess.CompletedProcess[Any]:
        return subprocess.CompletedProcess(cmd, 7, "", "")

    monkeypatch.setattr(subprocess, "run", fake_run)
    rc = profile.run_profile(cfg, level="level0", url="http://x")
    assert rc == 7
