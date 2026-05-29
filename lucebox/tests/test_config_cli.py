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
