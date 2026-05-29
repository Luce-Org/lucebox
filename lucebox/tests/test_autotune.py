from lucebox.autotune import runtime_from_host
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
