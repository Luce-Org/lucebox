"""Tests for ``lucebox check`` — readiness report.

The check command has two surfaces that must stay independent:

  * pass/fail checks → drive the exit code, so the command is usable
    as a CI exit-code gate;
  * Host facts section → informational, prints the LUCEBOX_HOST_*
    convoy that gets baked into /opt/lucebox-hub/HOST_INFO inside
    the container.
"""

from __future__ import annotations

import pytest
from lucebox.cli import app
from lucebox.types import HostFacts
from rich.console import Console
from typer.testing import CliRunner

from lucebox import host_check


def test_check_prints_host_facts_section(monkeypatch: pytest.MonkeyPatch) -> None:
    """`lucebox check` includes a Host facts block sourced from LUCEBOX_HOST_*."""
    monkeypatch.setenv("LUCEBOX_HOST_OS_PRETTY", "Ubuntu 22.04.3 LTS")
    monkeypatch.setenv("LUCEBOX_HOST_KERNEL", "6.6.87.2-microsoft-standard-WSL2")
    monkeypatch.setenv("LUCEBOX_HOST_WSL_VERSION", "wsl2")
    monkeypatch.setenv("LUCEBOX_HOST_DOCKER_VERSION", "29.1.3")
    monkeypatch.setenv("LUCEBOX_HOST_DRIVER_VERSION", "596.36")
    monkeypatch.setenv("LUCEBOX_HOST_NVIDIA_CTK_VERSION", "1.16.2")
    monkeypatch.setenv("LUCEBOX_HOST_CPU_MODEL", "Intel Test CPU")
    monkeypatch.setenv(
        "LUCEBOX_HOST_GPU_LIST_CSV",
        "0, GPU-abc, 00000000:01:00.0, NVIDIA RTX 5090, 12.0, 24576 MiB, 175.00 W",
    )
    # Stub HostFacts so the pass/fail checks succeed at least minimally.
    # `cli.check` imports `from_env` into its module namespace, so patch
    # both names.
    def stub() -> HostFacts:
        return HostFacts(
            nproc=24,
            ram_gb=64,
            gpu_vendor="nvidia",
            gpu_name="NVIDIA RTX 5090",
            gpu_count=1,
            vram_gb=24,
            gpu_sm="120",
            driver_version="596.36",
            driver_major=596,
            has_systemd=True,
            is_wsl=True,
            has_docker=True,
            docker_version="29.1.3",
            ctk="runtime",
        )
    monkeypatch.setattr("lucebox.host_facts.from_env", stub)
    monkeypatch.setattr("lucebox.cli.from_env", stub)
    result = CliRunner().invoke(app, ["check"])
    # The pass/fail half of `check` should still exit 0 on this stubbed host.
    assert result.exit_code == 0, result.stdout
    assert "Host facts" in result.stdout
    assert "Ubuntu 22.04.3 LTS" in result.stdout
    assert "wsl2" in result.stdout
    assert "1.16.2" in result.stdout
    assert "Intel Test CPU" in result.stdout
    # Multi-GPU table line.
    assert "NVIDIA RTX 5090" in result.stdout


def test_render_host_facts_unset_env_shows_placeholders(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """All LUCEBOX_HOST_* unset → section still renders with explicit (unset) markers."""
    for k in list(__import__("os").environ):
        if k.startswith("LUCEBOX_HOST_"):
            monkeypatch.delenv(k, raising=False)
    console = Console(force_terminal=False, no_color=True, record=True)
    host_check.render_host_facts(console)
    text = console.export_text()
    assert "Host facts" in text
    # Multi-line section renders even when no env was passed in.
    assert "(unset)" in text
    assert "gpus" in text


def test_check_exit_code_independent_of_host_facts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Host facts section must not change the exit-code semantics of check.

    Drives the pass/fail logic through a known-fail HostFacts (no docker)
    and asserts the exit code is still 1, regardless of what the Host
    facts block prints.
    """
    monkeypatch.setenv("LUCEBOX_HOST_OS_PRETTY", "Bare Linux")
    def stub() -> HostFacts:
        return HostFacts(
            nproc=8,
            ram_gb=16,
            gpu_vendor="nvidia",
            gpu_name="X",
            gpu_count=1,
            vram_gb=24,
            gpu_sm="86",
            driver_version="555.00",
            driver_major=555,
            has_systemd=False,
            is_wsl=False,
            has_docker=False,  # → fail
            docker_version="",
            ctk="none",  # also fail
        )
    monkeypatch.setattr("lucebox.host_facts.from_env", stub)
    monkeypatch.setattr("lucebox.cli.from_env", stub)
    result = CliRunner().invoke(app, ["check"])
    assert result.exit_code == 1
    # Host facts block still printed despite the failure.
    assert "Host facts" in result.stdout
