"""Tests for the ``lucebox models`` sub-app."""

from __future__ import annotations

from pathlib import Path

import pytest
from lucebox.cli import app
from lucebox.download import PRESETS
from lucebox.types import HostFacts
from typer.testing import CliRunner

from lucebox import config as config_mod
from lucebox import download as download_mod


def _set_config_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("LUCEBOX_HOME", str(tmp_path))
    monkeypatch.setenv("LUCEBOX_MODELS", str(tmp_path / "models"))
    return tmp_path / "config.toml"


def _stub_host(monkeypatch: pytest.MonkeyPatch, vram_gb: int) -> None:
    monkeypatch.setattr("lucebox.host_facts.from_env", lambda: HostFacts(vram_gb=vram_gb))
    monkeypatch.setattr("lucebox.cli.from_env", lambda: HostFacts(vram_gb=vram_gb))


def test_models_list_shows_every_registered_preset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_config_path(tmp_path, monkeypatch)
    _stub_host(monkeypatch, vram_gb=24)
    result = CliRunner().invoke(app, ["models", "list"])
    assert result.exit_code == 0
    for name in PRESETS:
        assert name in result.stdout


def test_models_default_view_lists_only_installed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_config_path(tmp_path, monkeypatch)
    _stub_host(monkeypatch, vram_gb=24)
    # No models on disk → default view says "no presets installed".
    result = CliRunner().invoke(app, ["models"])
    assert result.exit_code == 0
    assert "No presets installed" in result.stdout or "Models dir" in result.stdout


def test_models_download_recommends_when_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No preset configured + nothing on argv → auto-recommend + auto-activate."""
    cfg_path = _set_config_path(tmp_path, monkeypatch)
    _stub_host(monkeypatch, vram_gb=24)

    # Stub the network calls so the test doesn't try to talk to HF.
    monkeypatch.setattr(download_mod, "download_preset", lambda cfg, pres: 0)
    monkeypatch.setattr(
        download_mod,
        "status",
        lambda cfg, pres: {"target_present": True, "draft_present": True},
    )

    result = CliRunner().invoke(app, ["models", "download"])
    assert result.exit_code == 0
    assert "Recommended preset" in result.stdout
    assert cfg_path.exists()
    # The active preset should now be model.preset = qwen3.6-27b.
    entries = config_mod.config_get(path=cfg_path)
    assert entries["model.preset"] == ("qwen3.6-27b", "file")


def test_models_download_refuses_silent_switch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When a preset is already active, `download` with no arg refuses."""
    cfg_path = _set_config_path(tmp_path, monkeypatch)
    _stub_host(monkeypatch, vram_gb=24)
    config_mod.config_set("model.preset", "qwen3.6-27b", path=cfg_path)

    result = CliRunner().invoke(app, ["models", "download"])
    assert result.exit_code == 2
    assert "already active" in result.stdout.lower()


def test_models_download_explicit_preset_no_activate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Passing a preset without --activate downloads but doesn't flip model.preset."""
    cfg_path = _set_config_path(tmp_path, monkeypatch)
    _stub_host(monkeypatch, vram_gb=24)
    monkeypatch.setattr(download_mod, "download_preset", lambda cfg, pres: 0)
    monkeypatch.setattr(
        download_mod,
        "status",
        lambda cfg, pres: {"target_present": False, "draft_present": False},
    )

    result = CliRunner().invoke(app, ["models", "download", "gemma-4-26b"])
    assert result.exit_code == 0
    if cfg_path.exists():
        entries = config_mod.config_get(path=cfg_path)
        assert entries["model.preset"] == ("", "default")


def test_models_download_explicit_preset_with_activate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg_path = _set_config_path(tmp_path, monkeypatch)
    _stub_host(monkeypatch, vram_gb=24)
    monkeypatch.setattr(download_mod, "download_preset", lambda cfg, pres: 0)
    monkeypatch.setattr(
        download_mod,
        "status",
        lambda cfg, pres: {"target_present": False, "draft_present": False},
    )

    result = CliRunner().invoke(app, ["models", "download", "gemma-4-26b", "--activate"])
    assert result.exit_code == 0
    entries = config_mod.config_get(path=cfg_path)
    assert entries["model.preset"] == ("gemma-4-26b", "file")


def test_installed_helpers_track_presence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``installed_status`` / ``installed_size_gb`` reflect on-disk byte counts."""
    _set_config_path(tmp_path, monkeypatch)
    _stub_host(monkeypatch, vram_gb=24)
    from lucebox.config import live_config

    cfg = live_config()
    cfg.models_dir.mkdir(parents=True, exist_ok=True)
    laguna = PRESETS["laguna-xs.2"]
    assert download_mod.installed_status(cfg, laguna) == "absent"

    target = cfg.models_dir / laguna.target_file
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"x" * (5 * 10**9))
    assert download_mod.installed_status(cfg, laguna) == "installed"
    assert download_mod.installed_size_gb(cfg, laguna) == pytest.approx(5.0, rel=0.01)
