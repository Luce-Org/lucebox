from pathlib import Path

from lucebox.autotune import automatic_plan, runtime_from_host
from lucebox.types import Config, HostFacts, ModelMeta

from lucebox import download


def test_wsl_24gb_defaults_leave_cuda_headroom() -> None:
    runtime = runtime_from_host(HostFacts(vram_gb=24, is_wsl=True))

    assert runtime.budget == 16
    # Bumped 65536 → 98304 on 2026-05-30 after the gemma4-26b coding-
    # agent-loop sweep proved 98K serves 90K-token agentic prompts
    # with ~3 GB VRAM headroom and no CUDA VMM failures on the 3090 Ti
    # WSL configuration (see
    # docs/experiments/gemma4-26b-coding-agent-loop-sweep-2026-05-30.md).
    assert runtime.max_ctx == 98304
    # lazy is False because the heuristic path does NOT set prefill_drafter,
    # and the C++ server silently ignores --lazy-draft without it. Flipping
    # to False makes the host config match runtime behaviour. See the
    # `entrypoint.sh` warning emitted when the two are out-of-sync.
    assert runtime.lazy is False
    assert runtime.prefix_cache_slots == 0


def test_native_24gb_caps_context_below_vmm_failure_boundary() -> None:
    runtime = runtime_from_host(HostFacts(vram_gb=24, is_wsl=False))

    assert runtime.budget == 22
    assert runtime.max_ctx == 98304
    assert runtime.lazy is False  # see WSL test above
    assert runtime.prefix_cache_slots == 0


def test_no_heuristic_tier_sets_lazy_without_prefill_drafter() -> None:
    """Regression for the `--lazy-draft ignored` silent no-op.

    The C++ dflash_server drops `--lazy-draft` unless `--prefill-drafter`
    is also passed. The heuristic doesn't set `prefill_drafter`, so any
    tier that sets `lazy=True` would produce a host config that doesn't
    match what actually ran — exactly the mismatch the sindri decode
    sweep tripped over (every docker.stderr contained the warning).
    """
    for vram in (0, 8, 16, 24, 40, 80):
        for is_wsl in (False, True):
            rt = runtime_from_host(HostFacts(vram_gb=vram, is_wsl=is_wsl))
            if rt.lazy:
                assert rt.prefill_drafter, (
                    f"vram={vram} is_wsl={is_wsl}: lazy=True without "
                    f"prefill_drafter → silent no-op on the C++ server"
                )


def _cfg(tmp_path: Path, preset: str, host: HostFacts) -> Config:
    return Config(models_dir=tmp_path / "models", host=host, model=ModelMeta(preset=preset))


def _install_optimizer_drafter(cfg: Config) -> None:
    path = download.optimizer_drafter_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"test-scorer")


def _install_decode_draft(cfg: Config) -> None:
    preset = download.PRESETS[cfg.model.preset]
    assert preset.draft_file is not None
    path = cfg.models_dir / "draft" / preset.draft_file
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"test-draft")


def test_unconfigured_plan_does_not_claim_an_active_optimization(tmp_path: Path) -> None:
    cfg = Config(
        models_dir=tmp_path / "models",
        host=HostFacts(gpu_vendor="amd", vram_gb=31, gpu_sm="gfx1201"),
    )

    plan = automatic_plan(cfg)

    assert plan.active_names == ()
    assert plan.dflash.available is False
    assert "choose a model" in plan.dflash.reason


def test_qwen_24gb_enables_dflash_and_pflash_but_prefers_full_kv(
    tmp_path: Path,
) -> None:
    cfg = _cfg(
        tmp_path,
        "qwen3.6-27b",
        HostFacts(gpu_vendor="nvidia", vram_gb=24, gpu_sm="86"),
    )
    _install_optimizer_drafter(cfg)
    _install_decode_draft(cfg)

    plan = automatic_plan(cfg)

    assert plan.dflash.enabled is True
    assert plan.pflash.enabled is True
    assert plan.kvflash.enabled is False
    assert plan.spark.enabled is False
    assert plan.runtime.prefill_mode == "auto"
    assert plan.runtime.prefill_keep_ratio == 0.10
    assert plan.runtime.max_ctx == 98304
    assert plan.active_names == ("DFlash", "PFlash")


def test_published_draft_is_not_claimed_until_it_is_installed(tmp_path: Path) -> None:
    cfg = _cfg(
        tmp_path,
        "qwen3.6-27b",
        HostFacts(gpu_vendor="nvidia", vram_gb=24, gpu_sm="86"),
    )

    plan = automatic_plan(cfg, optimizer_drafter_available=False)

    assert plan.dflash.enabled is False
    assert plan.dflash.available is False
    assert "not installed" in plan.dflash.reason


def test_tight_qwen_uses_drafter_free_qk_kvflash_until_scorer_is_installed(
    tmp_path: Path,
) -> None:
    cfg = _cfg(
        tmp_path,
        "qwen3.6-27b",
        HostFacts(gpu_vendor="nvidia", vram_gb=20, gpu_sm="86"),
    )

    plan = automatic_plan(cfg)

    assert plan.pflash.enabled is False
    assert plan.kvflash.enabled is True
    assert plan.runtime.kvflash == "auto"
    assert plan.runtime.kvflash_policy == "qk"
    assert plan.needs_optimizer_drafter is True


def test_constrained_qwen_moe_enables_spark_and_scored_kvflash(
    tmp_path: Path,
) -> None:
    cfg = _cfg(
        tmp_path,
        "qwen3.6-moe",
        HostFacts(gpu_vendor="nvidia", vram_gb=24, ram_gb=64, gpu_sm="86"),
    )
    _install_optimizer_drafter(cfg)

    plan = automatic_plan(cfg)

    assert plan.dflash.enabled is False
    assert plan.pflash.enabled is False
    assert plan.kvflash.enabled is True
    assert plan.runtime.kvflash_policy == "drafter"
    assert plan.spark.enabled is True


def test_r9700_qwen_moe_keeps_exact_all_gpu_path(tmp_path: Path) -> None:
    cfg = _cfg(
        tmp_path,
        "qwen3.6-moe",
        HostFacts(gpu_vendor="amd", vram_gb=31, gpu_sm="gfx1201"),
    )

    plan = automatic_plan(cfg, optimizer_drafter_available=False)

    assert plan.kvflash.enabled is False
    assert plan.spark.enabled is False
    assert "fits" in plan.spark.reason


def test_laguna_uses_native_context_and_spark_under_24gb_pressure(
    tmp_path: Path,
) -> None:
    cfg = _cfg(
        tmp_path,
        "laguna-xs.2",
        HostFacts(gpu_vendor="nvidia", vram_gb=24, ram_gb=64, gpu_sm="86"),
    )

    plan = automatic_plan(cfg, optimizer_drafter_available=False)

    assert plan.runtime.max_ctx == 4096
    assert plan.dflash.enabled is False
    assert plan.pflash.available is False
    assert plan.kvflash.enabled is False
    assert plan.spark.enabled is True


def test_spark_stays_off_when_host_ram_cannot_hold_cold_experts(
    tmp_path: Path,
) -> None:
    cfg = _cfg(
        tmp_path,
        "laguna-xs.2",
        HostFacts(gpu_vendor="nvidia", vram_gb=24, ram_gb=16, gpu_sm="86"),
    )

    plan = automatic_plan(cfg, optimizer_drafter_available=False)

    assert plan.spark.available is True
    assert plan.spark.enabled is False
    assert plan.runtime.spark is False
    assert "16 GB host RAM" in plan.spark.reason


def test_gemma_does_not_offer_unsupported_pflash(tmp_path: Path) -> None:
    cfg = _cfg(
        tmp_path,
        "gemma-4-31b",
        HostFacts(gpu_vendor="amd", vram_gb=31, gpu_sm="gfx1201"),
    )
    _install_optimizer_drafter(cfg)

    plan = automatic_plan(cfg)

    assert plan.pflash.available is False
    assert plan.pflash.enabled is False
    assert plan.runtime.prefill_mode == "off"
    assert "no PFlash" in plan.pflash.reason


def test_gemma_context_is_capped_when_exact_cache_has_little_headroom(
    tmp_path: Path,
) -> None:
    cfg = _cfg(
        tmp_path,
        "gemma-4-31b",
        HostFacts(gpu_vendor="nvidia", vram_gb=24, gpu_sm="86"),
    )

    plan = automatic_plan(cfg, optimizer_drafter_available=False)

    assert plan.runtime.max_ctx == 8192
    assert plan.runtime.cache_type_k == ""
    assert plan.runtime.cache_type_v == ""


def test_automatic_hardware_tiers_never_force_quality_risky_kv_types() -> None:
    for vram_gb in (0, 8, 16, 24, 32, 80):
        runtime = runtime_from_host(HostFacts(vram_gb=vram_gb))
        assert runtime.cache_type_k == ""
        assert runtime.cache_type_v == ""


def test_known_gpu_specific_ddtree_budgets() -> None:
    assert runtime_from_host(HostFacts(vram_gb=24, gpu_sm="gfx1100")).budget == 8
    assert runtime_from_host(HostFacts(vram_gb=32, gpu_sm="120")).budget == 40
    assert runtime_from_host(HostFacts(vram_gb=31, gpu_sm="gfx1201")).budget == 22
