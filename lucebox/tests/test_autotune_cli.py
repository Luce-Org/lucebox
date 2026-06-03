"""Tests for the ``lucebox autotune`` CLI surface."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from lucebox.cli import app
from lucebox.types import HostFacts
from typer.testing import CliRunner

from lucebox import autotune as autotune_mod
from lucebox import config as config_mod


def _set_config_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Pin the on-disk config.toml path under tmp."""
    cfg_path = tmp_path / "config.toml"
    monkeypatch.setenv("LUCEBOX_HOME", str(tmp_path))
    return cfg_path


def _stub_host(monkeypatch: pytest.MonkeyPatch, vram_gb: int) -> None:
    """Force the env-driven HostFacts to a known VRAM tier."""
    monkeypatch.setattr(
        "lucebox.host_facts.from_env",
        lambda: HostFacts(vram_gb=vram_gb, gpu_name="Test", gpu_count=1),
    )
    # cli.py imports from_env directly into its module namespace.
    monkeypatch.setattr("lucebox.cli.from_env", lambda: HostFacts(vram_gb=vram_gb))


def test_autotune_json_dumps_dflashruntime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _set_config_path(tmp_path, monkeypatch)
    _stub_host(monkeypatch, vram_gb=24)
    result = CliRunner().invoke(app, ["autotune", "--json"])
    assert result.exit_code == 0
    # Trim ANSI escapes by parsing the bare body.
    payload = json.loads(result.stdout)
    # 24 GB native → max_ctx=98304, budget=22 (see autotune tiers).
    assert payload["budget"] == 22
    assert payload["max_ctx"] == 98304


def test_autotune_apply_writes_eleven_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg_path = _set_config_path(tmp_path, monkeypatch)
    _stub_host(monkeypatch, vram_gb=24)
    result = CliRunner().invoke(app, ["autotune", "--apply"])
    assert result.exit_code == 0
    assert cfg_path.exists()
    # Read back via the same sparse-get to confirm each of the 11
    # allowlisted keys is now "from file" rather than "from default".
    from lucebox.cli import DFLASH_ALLOWLIST

    entries = config_mod.config_get(path=cfg_path)
    for name in DFLASH_ALLOWLIST:
        value, origin = entries[f"dflash.{name}"]
        assert origin == "file", f"dflash.{name} did not land in config.toml"
    # think_max is NOT in the allowlist — autotune --apply must not touch it.
    _value, origin = entries["dflash.think_max"]
    assert origin == "default"


def test_autotune_default_view_does_not_write_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg_path = _set_config_path(tmp_path, monkeypatch)
    _stub_host(monkeypatch, vram_gb=24)
    result = CliRunner().invoke(app, ["autotune"])
    assert result.exit_code == 0
    # No --apply → no file created.
    assert not cfg_path.exists()


def test_autotune_apply_refuses_when_persisted_value_differs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A previously-persisted dflash.* key with a different value blocks --apply."""
    cfg_path = _set_config_path(tmp_path, monkeypatch)
    _stub_host(monkeypatch, vram_gb=24)
    # First apply: lands fresh recommendations.
    rc1 = CliRunner().invoke(app, ["autotune", "--apply"])
    assert rc1.exit_code == 0
    # Now overwrite one key with a hand-tuned value (simulates a bench
    # winner: budget=16 instead of the recommendation's 22).
    config_mod.config_set("dflash.budget", 16, path=cfg_path)
    config_mod.config_set("dflash.max_ctx", 65536, path=cfg_path)
    # Re-invoke --apply: drift guard fires.
    result = CliRunner().invoke(app, ["autotune", "--apply"])
    assert result.exit_code == 1
    assert "already differ" in result.stdout
    assert "dflash.budget" in result.stdout
    assert "current=16" in result.stdout
    assert "recommended=22" in result.stdout
    # Persisted bench winner stayed in place.
    entries = config_mod.config_get(path=cfg_path)
    assert entries["dflash.budget"][0] == 16
    assert entries["dflash.budget"][1] == "file"


def test_autotune_apply_force_bypasses_drift_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cfg_path = _set_config_path(tmp_path, monkeypatch)
    _stub_host(monkeypatch, vram_gb=24)
    config_mod.config_set("dflash.budget", 16, path=cfg_path)
    config_mod.config_set("dflash.max_ctx", 65536, path=cfg_path)
    result = CliRunner().invoke(app, ["autotune", "--apply", "--force"])
    assert result.exit_code == 0, result.stdout
    entries = config_mod.config_get(path=cfg_path)
    # The recommendation overwrites the bench winner.
    assert entries["dflash.budget"][0] == 22


def test_autotune_apply_first_time_no_guard(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When no keys are persisted yet, --apply lands clean (no drift to compare to)."""
    cfg_path = _set_config_path(tmp_path, monkeypatch)
    _stub_host(monkeypatch, vram_gb=24)
    result = CliRunner().invoke(app, ["autotune", "--apply"])
    assert result.exit_code == 0
    assert cfg_path.exists()


def test_autotune_sweep_flag_is_registered() -> None:
    """``--sweep`` and ``--yes`` appear on the autotune CLI surface."""
    result = CliRunner().invoke(app, ["autotune", "--help"])
    assert result.exit_code == 0
    assert "--sweep" in result.output
    assert "--yes" in result.output


def test_autotune_sweep_and_apply_are_mutually_exclusive(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`--sweep --apply` is refused — sweep applies its own winner."""
    _set_config_path(tmp_path, monkeypatch)
    _stub_host(monkeypatch, vram_gb=24)
    result = CliRunner().invoke(app, ["autotune", "--sweep", "--apply", "--yes"])
    assert result.exit_code == 2  # typer.Exit(code=2)  # noqa: PLR2004


def test_autotune_sweep_dispatches_to_run_sweep(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`autotune --sweep` calls into lucebox.sweep.run_sweep (mocked)."""
    _set_config_path(tmp_path, monkeypatch)
    _stub_host(monkeypatch, vram_gb=24)

    called = {"n": 0}

    def fake_run_sweep(**kw):  # noqa: ARG001
        called["n"] += 1
        return 0

    monkeypatch.setattr("lucebox.sweep.run_sweep", fake_run_sweep)
    result = CliRunner().invoke(app, ["autotune", "--sweep", "--yes"])
    assert result.exit_code == 0
    assert called["n"] == 1


def test_recommend_preset_tiers() -> None:
    """The preset recommender follows the VRAM thresholds in the spec."""
    assert autotune_mod.recommend_preset(HostFacts(vram_gb=24)) == "qwen3.6-27b"
    assert autotune_mod.recommend_preset(HostFacts(vram_gb=22)) == "qwen3.6-27b"
    assert autotune_mod.recommend_preset(HostFacts(vram_gb=20)) == "laguna-xs.2"
    assert autotune_mod.recommend_preset(HostFacts(vram_gb=16)) == "laguna-xs.2"
    assert autotune_mod.recommend_preset(HostFacts(vram_gb=12)) is None
    assert autotune_mod.recommend_preset(HostFacts(vram_gb=0)) is None
