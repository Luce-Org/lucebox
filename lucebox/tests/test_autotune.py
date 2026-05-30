import pytest
from lucebox.autotune import PROFILES, get_profile, runtime_from_host
from lucebox.types import HostFacts


def test_wsl_24gb_defaults_leave_cuda_headroom() -> None:
    runtime = runtime_from_host(HostFacts(vram_gb=24, is_wsl=True))

    assert runtime.budget == 16
    assert runtime.max_ctx == 65536
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


# ── Profile abstraction + coding-agent-loop bracket ───────────────────────


def test_profiles_registered():
    """Two profiles ship: legacy heuristic + coding-agent-loop."""
    assert "heuristic" in PROFILES
    assert "coding-agent-loop" in PROFILES
    assert PROFILES["heuristic"].scorer == "decode_tps_snapshot"
    assert PROFILES["coding-agent-loop"].scorer == "agent_replay_pass_rate"


def test_get_profile_unknown_raises():
    with pytest.raises(KeyError) as excinfo:
        get_profile("does-not-exist")
    assert "known" in str(excinfo.value).lower()


def test_coding_agent_loop_gemma_bracket_excludes_kv_axis():
    """Gemma4's KV is hardcoded F16 — the gemma bracket must NOT vary
    cache_type_k/v (a no-op axis there). Regression guard."""
    host = HostFacts(vram_gb=24, is_wsl=True)
    cells = PROFILES["coding-agent-loop"].candidate_configs(host, "gemma-4-26b")
    assert cells, "gemma bracket must produce at least one cell at 24 GB"
    kv_values = {(c.cache_type_k, c.cache_type_v) for c in cells}
    assert kv_values == {("", "")}, (
        f"gemma cells should not vary KV quant — got {kv_values}"
    )


def test_coding_agent_loop_gemma_bracket_varies_target_axes():
    """The gemma bracket must vary max_ctx × fa_window × budget."""
    host = HostFacts(vram_gb=24, is_wsl=True)
    cells = PROFILES["coding-agent-loop"].candidate_configs(host, "gemma-4-26b")
    assert len({c.max_ctx for c in cells}) >= 2, "max_ctx should be a swept axis"
    assert len({c.fa_window for c in cells}) >= 2, "fa_window should be a swept axis"
    assert len({c.budget for c in cells}) >= 2, "budget should be a swept axis"


def test_coding_agent_loop_qwen_bracket_includes_kv_axis():
    """Qwen3.6 / laguna respect cache_type_k/v — their bracket must
    sweep KV quant."""
    host = HostFacts(vram_gb=24, is_wsl=True)
    cells = PROFILES["coding-agent-loop"].candidate_configs(host, "qwen3.6-27b")
    kv_values = {c.cache_type_k for c in cells}
    assert "tq3_0" in kv_values and "q8_0" in kv_values, (
        f"qwen bracket should include both tq3_0 and q8_0; got {kv_values}"
    )


def test_coding_agent_loop_low_vram_falls_back_to_base():
    """Below 22 GB the model barely fits — sweeping risks OOM. Both
    arch builders should return just the heuristic base."""
    host = HostFacts(vram_gb=12)
    cells_g = PROFILES["coding-agent-loop"].candidate_configs(host, "gemma-4-26b")
    cells_q = PROFILES["coding-agent-loop"].candidate_configs(host, "qwen3.6-27b")
    assert len(cells_g) == 1
    assert len(cells_q) == 1
