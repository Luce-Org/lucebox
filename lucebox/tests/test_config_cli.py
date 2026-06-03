"""Tests for the ``lucebox config`` sub-app CLI."""

from __future__ import annotations

from pathlib import Path

import pytest
from lucebox.cli import app
from typer.testing import CliRunner


def _set_config_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("LUCEBOX_HOME", str(tmp_path))
    return tmp_path / "config.toml"


def test_config_set_then_get_round_trip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg_path = _set_config_path(tmp_path, monkeypatch)
    set_result = CliRunner().invoke(app, ["config", "set", "dflash.budget=12"])
    assert set_result.exit_code == 0
    assert cfg_path.exists()
    get_result = CliRunner().invoke(app, ["config", "get", "dflash.budget"])
    assert get_result.exit_code == 0
    assert "12" in get_result.stdout
    assert "from file" in get_result.stdout


def test_config_get_with_no_key_lists_every_registered_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_config_path(tmp_path, monkeypatch)
    result = CliRunner().invoke(app, ["config", "get"])
    assert result.exit_code == 0
    # Every registered dotted key shows up at least once.
    for key in ("model.preset", "dflash.budget", "port"):
        assert key in result.stdout


def test_config_unset_drops_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg_path = _set_config_path(tmp_path, monkeypatch)
    CliRunner().invoke(app, ["config", "set", "dflash.budget=9"])
    assert "budget = 9" in cfg_path.read_text()
    unset_result = CliRunner().invoke(app, ["config", "unset", "dflash.budget"])
    assert unset_result.exit_code == 0
    body = cfg_path.read_text()
    assert "budget" not in body


def test_config_set_unknown_key_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_config_path(tmp_path, monkeypatch)
    result = CliRunner().invoke(app, ["config", "set", "totally.unknown=1"])
    assert result.exit_code == 2


def test_config_set_rejects_missing_equals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_config_path(tmp_path, monkeypatch)
    result = CliRunner().invoke(app, ["config", "set", "dflash.budget"])
    assert result.exit_code == 2


def test_config_set_creates_file_when_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg_path = _set_config_path(tmp_path, monkeypatch)
    assert not cfg_path.exists()
    CliRunner().invoke(app, ["config", "set", "port=9090"])
    assert cfg_path.exists()
    assert "port = 9090" in cfg_path.read_text()


def test_load_or_build_env_overrides_persisted_config(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """LUCEBOX_* env vars must win over config.toml.

    Regression test for the precedence bug fixed in this commit: prior
    to the fix, `_load_or_build()` returned `config_mod.load()`'s result
    verbatim when config.toml existed, so the systemd unit's
    `Environment=LUCEBOX_IMAGE=...` was silently ignored. Sindri's
    config.toml had `[image]` without `registry`, which made the
    dataclass default `ghcr.io/luce-org/lucebox-hub` win over the
    intended easel image.
    """
    from lucebox.cli import _load_or_build

    cfg_path = _set_config_path(tmp_path, monkeypatch)
    # Write a config.toml WITHOUT an image.registry line — the
    # bug-trigger shape on sindri.
    cfg_path.write_text(
        '[image]\nvariant = "cuda12"\n[runtime]\nport = 9090\n'
        '[dflash]\nbudget = 22\n'
    )
    # Env should override what config.toml says (and what dataclass
    # defaults fill in for missing keys).
    monkeypatch.setenv("LUCEBOX_IMAGE", "ghcr.io/myfork/lucebox-hub")
    monkeypatch.setenv("LUCEBOX_PORT", "7777")
    monkeypatch.setenv("LUCEBOX_CONTAINER", "lucebox-test")
    cfg = _load_or_build()
    assert cfg.image == "ghcr.io/myfork/lucebox-hub"  # env beats dataclass default
    assert cfg.port == 7777                            # env beats config.toml
    assert cfg.container_name == "lucebox-test"        # env applied
    # variant is in config.toml — config.toml value (no env override).
    assert cfg.variant == "cuda12"
    # dflash IS persisted in config.toml — env doesn't touch it (no DFLASH_*
    # env hooks at this layer).
    assert cfg.dflash.budget == 22


def test_load_or_build_no_toml_env_overrides_defaults(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When config.toml is absent, env must still override defaults."""
    from lucebox.cli import _load_or_build

    _set_config_path(tmp_path, monkeypatch)
    # Don't write a config.toml — exercise the live_config() fallback.
    monkeypatch.setenv("LUCEBOX_IMAGE", "ghcr.io/myfork/lucebox-hub")
    cfg = _load_or_build()
    assert cfg.image == "ghcr.io/myfork/lucebox-hub"
