"""Tests for the top-level Typer surface."""

from __future__ import annotations

import os

import pytest
from lucebox.cli import app
from typer.testing import CliRunner


def test_config_subcommand_is_registered() -> None:
    result = CliRunner().invoke(app, ["config", "--help"])
    assert result.exit_code == 0
    assert "get" in result.output
    assert "set" in result.output
    assert "unset" in result.output


def test_models_subcommand_is_registered() -> None:
    result = CliRunner().invoke(app, ["models", "--help"])
    assert result.exit_code == 0
    assert "list" in result.output
    assert "download" in result.output


@pytest.mark.parametrize(
    "verb",
    [
        "autotune",
        "sweep",
        "profile",
        "smoke",
        "claude",
        "codex",
        "opencode",
        "hermes",
        "pi",
        "openclaw",
    ],
)
def test_deferred_verbs_are_not_registered(verb: str) -> None:
    """autotune/sweep, profile/smoke and the client launchers are deferred to
    follow-up PRs — this core CLI (launch / serve / install / download) must
    not expose them."""
    result = CliRunner().invoke(app, [verb, "--help"])
    assert result.exit_code != 0


def test_core_verbs_present_in_app() -> None:
    """The core launch/serve surface stays wired into the Typer command table."""
    registered = {
        c.name or (c.callback.__name__ if c.callback else "")
        for c in app.registered_commands
    }
    for verb in ("check", "pull", "print-run", "print-serve-argv", "version"):
        assert verb in registered


def test_legacy_subcommands_are_removed() -> None:
    """`configure` and `download-models` were folded into config/models."""
    cfg = CliRunner().invoke(app, ["configure", "--help"])
    assert cfg.exit_code != 0
    dl = CliRunner().invoke(app, ["download-models", "--help"])
    assert dl.exit_code != 0


def test_server_run_spec_forwards_lucebox_host_env(monkeypatch) -> None:
    """server_run_spec carries LUCEBOX_HOST_* from the orchestrator into the server.

    lucebox.sh exports the LUCEBOX_HOST_* convoy before `docker run` on the
    orchestrator; the orchestrator inherits them and we forward each one
    as ``-e KEY=VALUE`` to the server container so entrypoint.sh's
    write_host_info() can populate /opt/lucebox-hub/HOST_INFO.
    """
    import lucebox.docker_run as docker_run
    from lucebox.config import live_config

    # Scrub any pre-existing LUCEBOX_HOST_* env so the test sees only what we set.
    for k in list(os.environ):
        if k.startswith("LUCEBOX_HOST_"):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("LUCEBOX_HOST_OS_PRETTY", "Ubuntu 22.04.3 LTS")
    monkeypatch.setenv("LUCEBOX_HOST_KERNEL", "6.6.87.2-microsoft-standard-WSL2")
    monkeypatch.setenv("LUCEBOX_HOST_WSL_VERSION", "wsl2")
    monkeypatch.setenv(
        "LUCEBOX_HOST_GPU_LIST_CSV",
        "0, GPU-x, 00000000:01:00.0, NVIDIA RTX 5090, 12.0, 24576 MiB, 175.00 W",
    )

    cfg = live_config()
    spec = docker_run.server_run_spec(cfg)
    env_keys = {k for k, _ in spec.env}
    assert "LUCEBOX_HOST_OS_PRETTY" in env_keys
    assert "LUCEBOX_HOST_KERNEL" in env_keys
    assert "LUCEBOX_HOST_WSL_VERSION" in env_keys
    assert "LUCEBOX_HOST_GPU_LIST_CSV" in env_keys
    # DFLASH_* still present.
    assert "DFLASH_BUDGET" in env_keys
    # Values surface verbatim.
    env_map = dict(spec.env)
    assert env_map["LUCEBOX_HOST_OS_PRETTY"] == "Ubuntu 22.04.3 LTS"
