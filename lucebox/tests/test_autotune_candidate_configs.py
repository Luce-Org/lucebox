"""Tests for ``autotune.candidate_configs`` — the sweep bracket.

Pure-function tests: the bracket size matters per VRAM tier so a 5090-
Laptop test rig doesn't waste 90 min on configs that obviously won't
fit, and so a 24 GB box gets a useful empirical comparison against
the heuristic prior.
"""

from __future__ import annotations

from lucebox import autotune as autotune_mod
from lucebox.types import HostFacts


def _key(rt) -> tuple:
    """Stable identity tuple for de-dup checks across a config list."""
    return (rt.budget, rt.max_ctx, rt.cache_type_k, rt.cache_type_v)


def test_no_vram_signal_returns_single_config() -> None:
    """``vram_gb=0`` → base only. No sweep — we have no signal to bracket."""
    configs = autotune_mod.candidate_configs(HostFacts(vram_gb=0))
    assert len(configs) == 1


def test_8gb_host_no_sweep() -> None:
    """8 GB → 1 config. Model barely fits; sweeping risks OOM."""
    configs = autotune_mod.candidate_configs(HostFacts(vram_gb=8))
    assert len(configs) == 1


def test_24gb_host_six_to_twelve_configs() -> None:
    """24 GB (the 5090/3090 tier) → ~6-12 cells covering budget × KV."""
    configs = autotune_mod.candidate_configs(HostFacts(vram_gb=24))
    assert 6 <= len(configs) <= 12  # noqa: PLR2004


def test_24gb_host_includes_heuristic_config() -> None:
    """The heuristic prior must be in the bracket so we can prove or disprove it."""
    host = HostFacts(vram_gb=24)
    base = autotune_mod.runtime_from_host(host)
    configs = autotune_mod.candidate_configs(host)
    assert any(_key(c) == _key(base) for c in configs), (
        f"heuristic {_key(base)} missing from bracket {[_key(c) for c in configs]}"
    )


def test_80gb_host_at_most_eight_configs() -> None:
    """80 GB → ≤ 8 configs. Larger brackets waste GPU time on
    differences the H100/A100 won't show.
    """
    configs = autotune_mod.candidate_configs(HostFacts(vram_gb=80))
    assert len(configs) <= 8  # noqa: PLR2004


def test_no_duplicates_in_returned_list() -> None:
    """Per-tier brackets must not generate redundant cells."""
    for vram in (8, 16, 24, 40, 80):
        configs = autotune_mod.candidate_configs(HostFacts(vram_gb=vram))
        keys = [_key(c) for c in configs]
        assert len(set(keys)) == len(keys), (
            f"vram={vram} GB produced duplicate cells: {keys}"
        )


def test_16gb_host_three_configs() -> None:
    """12-21 GB tier: budget bracket only, 3 cells (including base)."""
    configs = autotune_mod.candidate_configs(HostFacts(vram_gb=16))
    # Three budget values × 1 KV combo = 3 cells.
    assert len(configs) == 3  # noqa: PLR2004


def test_40gb_host_includes_f16_kv() -> None:
    """32-47 GB tier opens up f16 KV — the bracket should test it."""
    configs = autotune_mod.candidate_configs(HostFacts(vram_gb=40))
    kv_types = {c.cache_type_k for c in configs}
    assert "f16" in kv_types
    assert "tq3_0" in kv_types
    assert "q8_0" in kv_types


def test_wsl_24gb_includes_wsl_heuristic() -> None:
    """WSL drops the heuristic max_ctx to 65536; the bracket follows."""
    host = HostFacts(vram_gb=24, is_wsl=True)
    base = autotune_mod.runtime_from_host(host)
    assert base.max_ctx == 65536  # WSL heuristic, see runtime_from_host  # noqa: PLR2004
    configs = autotune_mod.candidate_configs(host)
    assert any(_key(c) == _key(base) for c in configs)
