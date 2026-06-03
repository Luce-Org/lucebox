"""Tests for the sparse TOML config persistence layer."""

from __future__ import annotations

from pathlib import Path

import pytest
from lucebox.config import config_get, config_set, config_unset

from lucebox import config


def test_legacy_env_migration_skips_invalid_values(tmp_path: Path) -> None:
    legacy = tmp_path / "config.env"
    legacy.write_text("DFLASH_BUDGET=not-an-int\nDFLASH_MAX_CTX=65536\nDFLASH_LAZY=true\n")

    cfg, _doc = config._load_legacy_env(legacy)

    assert cfg.dflash.budget == 22
    assert cfg.dflash.max_ctx == 65536
    assert cfg.dflash.lazy is True


def test_image_variant_round_trips_from_toml(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        "[image]\n"
        'registry = "ghcr.io/luce-org/lucebox-hub"\n'
        'variant = "integration-props-uv-squared-clean-cuda12"\n'
    )

    cfg = config._load_toml(path)

    assert cfg.image == "ghcr.io/luce-org/lucebox-hub"
    assert cfg.variant == "integration-props-uv-squared-clean-cuda12"


def test_model_preset_round_trips_through_set_and_load(tmp_path: Path) -> None:
    """Setting model.preset writes a sparse TOML doc that loads back correctly."""
    path = tmp_path / "config.toml"
    config_set("model.preset", "gemma-4-26b", path=path)
    config_set("model.target_file", "google_gemma-4-26B-A4B-it-Q4_K_M.gguf", path=path)

    cfg = config._load_toml(path)
    assert cfg.model.preset == "gemma-4-26b"
    assert cfg.model.target_file == "google_gemma-4-26B-A4B-it-Q4_K_M.gguf"


def test_legacy_config_without_model_section_stays_unpinned(tmp_path: Path) -> None:
    """Legacy configs (no [model] section) must NOT silently pin to qwen."""
    path = tmp_path / "config.toml"
    path.write_text('[image]\nvariant = "cuda12"\n')

    cfg = config._load_toml(path)

    assert cfg.model.preset == ""
    assert cfg.model.target_file == ""
    assert cfg.model.draft_file == ""


def test_model_section_picks_target_file_from_registry(tmp_path: Path) -> None:
    """A bare [model] preset="..." entry pulls target_file from the registry."""
    path = tmp_path / "config.toml"
    path.write_text('[model]\npreset = "gemma-4-31b"\n')

    cfg = config._load_toml(path)

    assert cfg.model.preset == "gemma-4-31b"
    assert cfg.model.target_file == "google_gemma-4-31B-it-Q4_K_M.gguf"


def test_model_section_picks_draft_file_from_registry(tmp_path: Path) -> None:
    """When preset has a published draft GGUF, [model] preset="..." picks draft_file too."""
    path = tmp_path / "config.toml"
    path.write_text('[model]\npreset = "qwen3.6-27b"\n')

    cfg = config._load_toml(path)
    assert cfg.model.preset == "qwen3.6-27b"
    assert cfg.model.draft_file == "dflash-draft-3.6-q4_k_m.gguf"


def test_config_set_writes_only_named_key(tmp_path: Path) -> None:
    """Sparse persistence: setting one key does NOT serialize every default."""
    path = tmp_path / "config.toml"
    config_set("dflash.budget", 16, path=path)
    body = path.read_text()
    # The only [dflash] field that should appear is budget — none of the others.
    assert "[dflash]" in body
    assert "budget = 16" in body
    assert "max_ctx" not in body  # not user-set, must not appear
    assert "lazy" not in body
    assert "[host]" not in body  # whole section absent
    assert "[image]" not in body  # not touched either


def test_config_set_preserves_existing_keys(tmp_path: Path) -> None:
    """Setting a new key leaves previously-set keys intact."""
    path = tmp_path / "config.toml"
    config_set("dflash.budget", 16, path=path)
    config_set("model.preset", "qwen3.6-27b", path=path)
    body = path.read_text()
    assert "budget = 16" in body
    assert 'preset = "qwen3.6-27b"' in body


def test_config_unset_removes_one_key(tmp_path: Path) -> None:
    """Unset removes the named key and leaves siblings alone."""
    path = tmp_path / "config.toml"
    config_set("dflash.budget", 16, path=path)
    config_set("dflash.max_ctx", 65536, path=path)
    changed = config_unset("dflash.budget", path=path)
    assert changed is True
    body = path.read_text()
    assert "budget" not in body
    assert "max_ctx = 65536" in body


def test_config_unset_drops_empty_section(tmp_path: Path) -> None:
    """Unsetting the last key in a section drops the empty section."""
    path = tmp_path / "config.toml"
    config_set("dflash.budget", 16, path=path)
    config_unset("dflash.budget", path=path)
    body = path.read_text()
    # The section may still exist as an empty table but `[dflash]` shouldn't.
    assert "[dflash]" not in body


def test_config_get_reports_origin(tmp_path: Path) -> None:
    """Each key carries an origin label — `file` when overridden, `default` otherwise."""
    path = tmp_path / "config.toml"
    config_set("dflash.budget", 9, path=path)
    entries = config_get(path=path)
    assert entries["dflash.budget"] == (9, "file")
    # max_ctx wasn't set so should report the live default.
    value, origin = entries["dflash.max_ctx"]
    assert origin == "default"
    assert value == 16384  # DflashRuntime.max_ctx default


def test_config_get_rejects_unknown_key(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    with pytest.raises(KeyError):
        config_get("not.a.key", path=path)


def test_config_set_rejects_unknown_key(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    with pytest.raises(KeyError):
        config_set("not.a.key", 1, path=path)


def test_config_set_auto_creates_file(tmp_path: Path) -> None:
    """`config set` creates a missing config.toml on first write."""
    path = tmp_path / "config.toml"
    assert not path.exists()
    config_set("port", 9090, path=path)
    assert path.exists()
    assert "port = 9090" in path.read_text()


def test_save_writes_sparse_doc(tmp_path: Path) -> None:
    """`save` writes whatever doc is handed in — no defaults serialized."""
    path = tmp_path / "config.toml"
    cfg = config._from_dict({})
    config.save(cfg, path, doc={"dflash": {"budget": 9}})
    body = path.read_text()
    assert "budget = 9" in body
    assert "max_ctx" not in body


def test_live_config_uses_recommend_preset_indirectly(tmp_path: Path) -> None:
    """``live_config()`` returns a Config — no implicit preset when none given."""
    # The function probes the env-provided HostFacts; with no preset arg
    # we must NOT silently pin one (that would surprise legacy installs).
    cfg = config.live_config()
    assert cfg.model.preset == ""
