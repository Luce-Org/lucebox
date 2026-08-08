"""Integrity tests for the product optimization contract."""

import pytest
from lucebox.capabilities import (
    ARCHITECTURE_CAPABILITIES,
    MODEL_OPTIMIZATION_PROFILES,
    Qualification,
    model_profile,
)
from lucebox.download import PRESETS


def test_every_catalog_model_has_exactly_one_optimization_profile() -> None:
    assert set(MODEL_OPTIMIZATION_PROFILES) == set(PRESETS)


@pytest.mark.parametrize("preset_name", PRESETS)
def test_profile_matches_catalog_architecture_and_covers_every_phase(
    preset_name: str,
) -> None:
    preset = PRESETS[preset_name]
    profile = model_profile(preset_name)

    assert profile.preset == preset_name
    assert profile.architecture == preset.architecture
    assert profile.architecture in ARCHITECTURE_CAPABILITIES
    assert all(label.strip() for label in profile.phase_baselines)


def test_deepseek_uses_native_mla_instead_of_claiming_generic_flash_paths() -> None:
    profile = model_profile("deepseek-v4-flash")

    assert profile.pflash is None
    assert profile.kvflash is None
    assert profile.prefix_cache_slots == 4
    assert "MLA" in profile.prefill_baseline
    assert "MLA" in profile.kv_baseline
    assert profile.deepseek_prefill is not None
    assert (
        profile.deepseek_prefill.feature.qualification_on("hip")
        is Qualification.PREVIEW
    )
    assert profile.deepseek_prefill.feature.automatic_on("hip") is False
    assert profile.deepseek_prefill.mode == "sparse"


def test_laguna_kvflash_is_explicitly_preview_not_an_automatic_claim() -> None:
    profile = model_profile("laguna-xs.2")

    assert profile.kvflash is not None
    assert profile.kvflash.policies == ("drafter", "lru")
    assert profile.kvflash.feature.available_on("cuda") is True
    assert profile.kvflash.feature.available_on("hip") is True
    assert profile.kvflash.feature.automatic_on("cuda") is False
    assert profile.kvflash.feature.automatic_on("hip") is False


def test_laguna_catalog_uses_the_published_native_context() -> None:
    assert PRESETS["laguna-xs.2"].native_context == 262_144
