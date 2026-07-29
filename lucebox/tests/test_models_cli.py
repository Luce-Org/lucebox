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
    host = HostFacts(vram_gb=vram_gb, ram_gb=64)
    monkeypatch.setattr("lucebox.host_facts.from_env", lambda: host)
    monkeypatch.setattr("lucebox.cli.from_env", lambda: host)


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
    assert "No presets installed" in result.stdout


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


def test_models_select_activates_preloaded_model_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A factory-preloaded buyer can switch models without a network call."""
    cfg_path = _set_config_path(tmp_path, monkeypatch)
    _stub_host(monkeypatch, vram_gb=24)
    preset = PRESETS["qwen3.6-27b"]
    models = tmp_path / "models"
    (models / preset.target_file).parent.mkdir(parents=True, exist_ok=True)
    (models / preset.target_file).write_bytes(b"preloaded-target")
    assert preset.draft_file is not None
    (models / "draft").mkdir()
    (models / "draft" / preset.draft_file).write_bytes(b"preloaded-draft")

    def fail_network(*args: object, **kwargs: object) -> object:
        raise AssertionError("preloaded selection must not contact Hugging Face")

    monkeypatch.setattr(download_mod, "status", fail_network)
    monkeypatch.setattr(download_mod, "download_preset", fail_network)

    result = CliRunner().invoke(app, ["models", "select", preset.name, "--yes"])
    assert result.exit_code == 0
    assert "Activated" in result.output
    entries = config_mod.config_get(path=cfg_path)
    assert entries["model.preset"] == (preset.name, "file")
    assert entries["model.target_file"] == (preset.target_file, "file")
    assert entries["model.draft_file"] == (preset.draft_file, "file")


def test_models_select_numbered_picker_downloads_and_activates(
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
    # qwen3.6-27b is the fourth entry in the stable alphabetical menu.
    result = CliRunner().invoke(app, ["models", "select"], input="4\ny\n")
    assert result.exit_code == 0
    assert "Choose a model" in result.output
    entries = config_mod.config_get(path=cfg_path)
    assert entries["model.preset"] == ("qwen3.6-27b", "file")


def test_optimize_resets_to_hardware_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg_path = _set_config_path(tmp_path, monkeypatch)
    _stub_host(monkeypatch, vram_gb=24)
    config_mod.config_set("dflash.max_ctx", 4096, path=cfg_path)

    result = CliRunner().invoke(app, ["optimize", "--yes"])
    assert result.exit_code == 0
    assert "Automatic optimization applied" in result.output
    entries = config_mod.config_get(path=cfg_path)
    assert entries["dflash.max_ctx"] == (98304, "file")
    assert entries["dflash.cache_type_k"] == ("", "file")


def test_optimize_applies_model_aware_qwen_stack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg_path = _set_config_path(tmp_path, monkeypatch)
    _stub_host(monkeypatch, vram_gb=24)
    config_mod.config_set("model.preset", "qwen3.6-27b", path=cfg_path)
    cfg = config_mod.load(cfg_path)
    assert cfg is not None
    cfg = config_mod.overlay_env(cfg)
    scorer = download_mod.optimizer_drafter_path(cfg)
    scorer.parent.mkdir(parents=True, exist_ok=True)
    scorer.write_bytes(b"preloaded")
    preset = PRESETS["qwen3.6-27b"]
    assert preset.draft_file is not None
    draft = cfg.models_dir / "draft" / preset.draft_file
    draft.parent.mkdir(parents=True, exist_ok=True)
    draft.write_bytes(b"preloaded-draft")

    result = CliRunner().invoke(app, ["optimize", "--yes"])

    assert result.exit_code == 0
    assert "DFlash" in result.output
    assert "PFlash" in result.output
    loaded = config_mod.load(cfg_path)
    assert loaded is not None
    assert loaded.dflash.speculative_decode is True
    assert loaded.dflash.prefill_mode == "auto"
    assert loaded.dflash.kvflash == "off"
    assert config_mod.optimization_mode(path=cfg_path) == "automatic"


def test_optimize_installs_shared_scorer_for_constrained_moe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg_path = _set_config_path(tmp_path, monkeypatch)
    _stub_host(monkeypatch, vram_gb=24)
    config_mod.config_set("model.preset", "qwen3.6-moe", path=cfg_path)

    def fake_download(cfg: object) -> int:
        assert isinstance(cfg, config_mod.Config)
        path = download_mod.optimizer_drafter_path(cfg)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"downloaded")
        return 0

    monkeypatch.setattr(download_mod, "download_optimizer_drafter", fake_download)

    result = CliRunner().invoke(app, ["optimize", "--yes"])

    assert result.exit_code == 0
    assert "Shared optimizer installed" in result.output
    loaded = config_mod.load(cfg_path)
    assert loaded is not None
    assert loaded.dflash.kvflash == "auto"
    assert loaded.dflash.kvflash_policy == "drafter"
    assert loaded.dflash.spark is True


def test_optimize_advanced_writes_a_custom_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg_path = _set_config_path(tmp_path, monkeypatch)
    _stub_host(monkeypatch, vram_gb=24)
    config_mod.config_set("model.preset", "qwen3.6-27b", path=cfg_path)
    cfg = config_mod.load(cfg_path)
    assert cfg is not None
    cfg = config_mod.overlay_env(cfg)
    scorer = download_mod.optimizer_drafter_path(cfg)
    scorer.parent.mkdir(parents=True, exist_ok=True)
    scorer.write_bytes(b"preloaded")
    preset = PRESETS["qwen3.6-27b"]
    assert preset.draft_file is not None
    draft = cfg.models_dir / "draft" / preset.draft_file
    draft.parent.mkdir(parents=True, exist_ok=True)
    draft.write_bytes(b"preloaded-draft")

    # DFlash yes, PFlash no, KVFlash yes, qk policy, apply yes.
    result = CliRunner().invoke(
        app,
        ["optimize", "--advanced"],
        input="y\nn\ny\nqk\ny\n",
    )

    assert result.exit_code == 0
    loaded = config_mod.load(cfg_path)
    assert loaded is not None
    assert loaded.dflash.speculative_decode is True
    assert loaded.dflash.prefill_mode == "off"
    assert loaded.dflash.kvflash == "auto"
    assert loaded.dflash.kvflash_policy == "qk"
    assert config_mod.optimization_mode(path=cfg_path) == "custom"


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
    target.touch()
    assert download_mod.installed_status(cfg, laguna) == "absent"
    # Apparent size is what the CLI reports; a sparse file exercises the same
    # stat path without allocating and writing a 5 GB Python byte string.
    with target.open("wb") as sparse:
        sparse.truncate(5 * 10**9)
    assert download_mod.installed_status(cfg, laguna) == "installed"
    assert download_mod.installed_size_gb(cfg, laguna) == pytest.approx(5.0, rel=0.01)
