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


def test_toml_load_uses_strict_bool_and_prefill_casters(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        '[dflash]\nlazy = "false"\ndebug_thinking_logits = "false"\nprefill_mode = "auto"\n'
    )

    cfg = config._load_toml(path)

    assert cfg.dflash.lazy is False
    assert cfg.dflash.debug_thinking_logits is False
    assert cfg.dflash.prefill_mode == "auto"


def test_toml_load_rejects_invalid_prefill_mode(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('[dflash]\nprefill_mode = "sometimes"\n')

    with pytest.raises(ValueError, match="prefill_mode"):
        config._load_toml(path)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("dflash.prefill_keep_ratio", "0"),
        ("dflash.prefill_keep_ratio", "1.01"),
        ("dflash.think_soft_close_min_ratio", "-0.01"),
        ("dflash.think_soft_close_min_ratio", "1.01"),
        ("dflash.think_soft_close_min_ratio", "nan"),
    ],
)
def test_config_set_rejects_out_of_range_ratios(tmp_path: Path, key: str, value: str) -> None:
    path = tmp_path / "config.toml"

    with pytest.raises(ValueError, match="interval"):
        config_set(key, value, path=path)

    assert not path.exists()


def test_toml_load_rejects_out_of_range_ratios(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text("[dflash]\nprefill_keep_ratio = 0.05\nthink_soft_close_min_ratio = 2.0\n")

    with pytest.raises(ValueError, match="think_soft_close_min_ratio"):
        config._load_toml(path)


@pytest.mark.parametrize(
    "host_body",
    [
        'gpu_vendor = "intel"\n',
        'ctk = "maybe"\n',
    ],
)
def test_toml_load_rejects_invalid_host_literals(tmp_path: Path, host_body: str) -> None:
    path = tmp_path / "config.toml"
    path.write_text(f"[host]\n{host_body}")

    with pytest.raises(ValueError):
        config._load_toml(path)


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


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("port", "0", "port"),
        ("port", "65536", "port"),
        ("models_dir", "relative/models", "absolute"),
        ("model.target_file", "../secret.gguf", "below models_dir"),
        ("model.draft_file", "/tmp/draft.gguf", "below models_dir"),
    ],
)
def test_config_set_rejects_unsafe_runtime_values(
    tmp_path: Path, key: str, value: str, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        config_set(key, value, path=tmp_path / "config.toml")


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


def test_live_config_selects_rocm_for_amd(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LUCEBOX_HOST_GPU_VENDOR", "amd")
    monkeypatch.delenv("LUCEBOX_VARIANT", raising=False)

    cfg = config.live_config()

    assert cfg.variant == "rocm"


def test_live_config_variant_override_wins_on_amd(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LUCEBOX_HOST_GPU_VENDOR", "amd")
    monkeypatch.setenv("LUCEBOX_VARIANT", "test-cuda12")

    cfg = config.live_config()

    assert cfg.variant == "test-cuda12"


def test_seed_dflash_writes_heuristic_when_absent(tmp_path: Path) -> None:
    """First-time activate seeds the VRAM-tier heuristic into config.toml.

    A config.toml with only [model] (what `models download --activate`
    writes) would otherwise load with DflashRuntime class defaults
    (max_ctx=16384), ignoring the host tier. Seeding writes the heuristic.
    """
    from lucebox.types import HostFacts

    path = tmp_path / "config.toml"
    path.write_text('[model]\npreset = "qwen3.6-27b"\n')
    wrote = config.seed_dflash_from_host(HostFacts(vram_gb=24, is_wsl=True), path=path)
    assert wrote is True
    loaded = config.load(path)
    assert loaded is not None
    # A 24 GB WSL host keeps extra virtualization headroom while still
    # replacing the 16K class default.
    assert loaded.dflash.max_ctx == 65536
    # Provenance recorded; [model] preserved.
    doc = config.load_doc(path)
    assert doc["autotune"]["source"] == "heuristic"
    assert doc["model"]["preset"] == "qwen3.6-27b"


def test_seed_dflash_is_noop_when_dflash_present(tmp_path: Path) -> None:
    """Never clobber a [dflash] the user or a prior tune already wrote."""
    from lucebox.types import HostFacts

    path = tmp_path / "config.toml"
    path.write_text("[dflash]\nmax_ctx = 4096\n")
    wrote = config.seed_dflash_from_host(HostFacts(vram_gb=80), path=path)
    assert wrote is False
    assert config.load(path).dflash.max_ctx == 4096


def test_seed_dflash_migrates_legacy_config_before_writing(tmp_path: Path) -> None:
    from lucebox.types import HostFacts

    path = tmp_path / "config.toml"
    path.with_suffix(".env").write_text("DFLASH_PORT=9090\n")

    wrote = config.seed_dflash_from_host(HostFacts(vram_gb=24), path=path)

    assert wrote is True
    loaded = config.load(path)
    assert loaded is not None
    assert loaded.port == 9090
    assert loaded.dflash.max_ctx == 98304


def test_optimization_fields_round_trip_and_validate(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        "[dflash]\n"
        "speculative_decode = false\n"
        'kvflash = "auto"\n'
        'kvflash_policy = "qk"\n'
        "kvflash_tau = 96\n"
        "spark = true\n"
        "spark_vram_gb = 14.5\n"
        'ds4_prefill = "sparse"\n'
    )

    loaded = config.load(path)
    assert loaded is not None
    assert loaded.dflash.speculative_decode is False
    assert loaded.dflash.kvflash == "auto"
    assert loaded.dflash.kvflash_policy == "qk"
    assert loaded.dflash.kvflash_tau == 96
    assert loaded.dflash.spark is True
    assert loaded.dflash.spark_vram_gb == 14.5
    assert loaded.dflash.ds4_prefill == "sparse"


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("dflash.kvflash", "banana"),
        ("dflash.kvflash", "0"),
        ("dflash.kvflash_policy", "random"),
        ("dflash.kvflash_tau", "0"),
        ("dflash.spark_vram_gb", "-1"),
        ("dflash.spark_vram_gb", "nan"),
        ("dflash.spark_vram_gb", "inf"),
        ("dflash.ds4_prefill", "fast"),
    ],
)
def test_config_rejects_invalid_optimization_values(tmp_path: Path, key: str, value: str) -> None:
    with pytest.raises(ValueError):
        config_set(key, value, path=tmp_path / "config.toml")


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("prefix_cache_slots", "prefix_cache_slots"),
        ("prefill_cache_slots", "prefill_cache_slots"),
    ],
)
def test_cache_slot_validation_names_the_invalid_field(field: str, message: str) -> None:
    from lucebox.types import DflashRuntime

    with pytest.raises(ValueError, match=message):
        DflashRuntime(**{field: -1})


def test_direct_optimization_edit_marks_profile_custom(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"

    config_set("dflash.spark", "true", path=path)

    assert config.optimization_mode(path=path) == "custom"
    assert config.load_doc(path)["autotune"]["source"] == "manual"


def test_manual_edit_rejects_incompatible_kvflash_and_fa_window(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config.toml"
    config_set("dflash.fa_window", 512, path=path)

    with pytest.raises(ValueError, match="mutually exclusive"):
        config_set("dflash.kvflash", "auto", path=path)

    loaded = config.load(path)
    assert loaded is not None
    assert loaded.dflash.fa_window == 512
    assert loaded.dflash.kvflash == "off"


def test_automatic_profile_replans_on_model_switch(tmp_path: Path) -> None:
    from lucebox.types import Config, HostFacts, ModelMeta

    path = tmp_path / "config.toml"
    config.seed_dflash_from_host(HostFacts(vram_gb=24), path=path)
    cfg = Config(
        models_dir=tmp_path / "models",
        host=HostFacts(gpu_vendor="nvidia", vram_gb=24, ram_gb=64, gpu_sm="86"),
        model=ModelMeta(preset="qwen3.6-moe"),
    )

    assert config.seed_optimization_from_config(cfg, path=path) is True
    loaded = config.load(path)
    assert loaded is not None
    assert loaded.dflash.speculative_decode is False
    assert loaded.dflash.spark is True
    assert config.optimization_mode(path=path) == "automatic"


def test_model_switch_preserves_custom_profile(tmp_path: Path) -> None:
    from lucebox.types import Config, HostFacts, ModelMeta

    path = tmp_path / "config.toml"
    config_set("dflash.max_ctx", 8192, path=path)
    cfg = Config(
        models_dir=tmp_path / "models",
        host=HostFacts(vram_gb=24),
        model=ModelMeta(preset="qwen3.6-moe"),
    )

    assert config.seed_optimization_from_config(cfg, path=path) is False
    loaded = config.load(path)
    assert loaded is not None
    assert loaded.dflash.max_ctx == 8192


def test_optimization_and_placement_are_persisted_atomically(tmp_path: Path) -> None:
    from lucebox.types import DflashRuntime, PlacementRuntime

    path = tmp_path / "config.toml"
    placement = PlacementRuntime(
        mode="layer-split",
        target_devices=("hip:0", "hip:1"),
        target_layer_split=(0.7, 0.3),
        peer_access=True,
    )

    config.write_optimization_runtime(
        DflashRuntime(max_ctx=32768),
        placement=placement,
        path=path,
    )
    loaded = config.load(path)

    assert loaded is not None
    assert loaded.dflash.max_ctx == 32768
    assert loaded.placement == placement
    assert config.optimization_mode(path=path) == "automatic"


def test_runtime_reset_clears_stale_placement(tmp_path: Path) -> None:
    from lucebox.types import DflashRuntime, PlacementRuntime

    path = tmp_path / "config.toml"
    config.write_optimization_runtime(
        DflashRuntime(),
        placement=PlacementRuntime(
            mode="layer-split",
            target_devices=("hip:0", "hip:1"),
            target_layer_split=(0.7, 0.3),
        ),
        path=path,
    )

    config.write_optimization_runtime(DflashRuntime(max_ctx=4096), path=path)

    assert "placement" not in config.load_doc(path)


def test_model_and_execution_profile_switch_atomically(tmp_path: Path) -> None:
    from lucebox.types import DflashRuntime, ModelMeta, PlacementRuntime

    path = tmp_path / "config.toml"
    path.write_text(
        '[model]\npreset = "old"\ntarget_file = "old.gguf"\ndraft_file = "old-draft.gguf"\n'
    )
    placement = PlacementRuntime(mode="single", target_device="cuda:0")

    config.write_model_profile(
        ModelMeta(preset="new", target_file="new.gguf"),
        DflashRuntime(max_ctx=32768),
        placement,
        path=path,
    )

    loaded = config.load(path)
    assert loaded is not None
    assert loaded.model == ModelMeta(preset="new", target_file="new.gguf")
    assert loaded.dflash.max_ctx == 32768
    assert loaded.placement == placement
    assert "draft_file" not in config.load_doc(path)["model"]


def test_automatic_seed_refuses_unrunnable_placement(tmp_path: Path) -> None:
    from lucebox.types import Config, HostFacts, ModelMeta

    path = tmp_path / "config.toml"
    cfg = Config(
        models_dir=tmp_path / "models",
        host=HostFacts(),
        model=ModelMeta(preset="qwen3.6-27b"),
    )

    with pytest.raises(ValueError, match="no runnable placement"):
        config.seed_optimization_from_config(cfg, path=path)

    assert not path.exists()


def test_config_rejects_unpaired_mixed_backend_placement(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text(
        "[placement]\n"
        'mode = "heterogeneous"\n'
        'target_device = "cuda:0"\n'
        'draft_device = "hip:0"\n'
        "remote_draft = false\n"
    )

    with pytest.raises(ValueError, match="requires remote_draft"):
        config.load(path)
