"""Heuristic autotune: VRAM tier → DflashRuntime defaults + preset picker.

The recommended runtime is computed from HostFacts (VRAM, is_wsl) and the
recommended preset from VRAM tier alone. Both are stateless — they take
HostFacts in and return a fresh value — so the CLI can apply them with
``lucebox autotune --apply`` without holding any global state.
"""

from __future__ import annotations

from dataclasses import replace

from lucebox.types import DflashRuntime, HostFacts


def runtime_from_host(host: HostFacts) -> DflashRuntime:
    """Pick a conservative DflashRuntime that 'should work' on this VRAM tier.

    Tiers (NVIDIA, target = Qwen3.6-27B Q4_K_M ~16 GB + Q4_K_M DFlash
    draft ~1 GB):
        <12 GB  — too small for 27B; pick min ctx as a floor so a fallback
                  start at least gets an error from the daemon rather than
                  a silent OOM.
        12-21   — fits but tight; cap ctx.
        22-31   — 24 GB-class consumer flagships (3090/4090/5090/5090-Laptop).
                  Cap at 96 K by default; 112 K can start, but long prompts
                  have reproduced CUDA VMM allocation failures on 24 GB cards.
                  WSL needs more CUDA/VMM headroom, so it starts lower and lets
                  the stress optimizer prove higher settings before persisting.
        32-47   — RTX 6000 Ada / A100 40 GB. Full 128 K.
        ≥48     — A100 80 GB / H100 / RTX 6000 Pro. Full 128 K.

    Prefix cache remains an explicit sweep tunable, but the automatic
    baseline keeps it off because tool prompts currently exercise a daemon
    snapshot path that is not reliable with prefix slots enabled.

    On `lazy`: the C++ server requires `--prefill-drafter` (and `--draft`)
    to be set for `--lazy-draft` to take effect, and silently ignores it
    otherwise (`--lazy-draft ignored: requires both --prefill-drafter and
    --draft`). Since the heuristic path does NOT set `prefill_drafter`,
    we default `lazy=False` here — "what we say" matches "what runs".
    Users who explicitly opt in via config.toml will be warned at server
    startup that the flag is being dropped (see entrypoint.sh).
    """
    if host.vram_gb <= 0:
        return DflashRuntime()  # no VRAM signal — stick with class defaults

    if host.vram_gb < 12:
        return DflashRuntime(max_ctx=4096)
    if host.vram_gb < 22:
        return DflashRuntime(max_ctx=32768)
    if host.vram_gb < 32:
        if host.is_wsl:
            return DflashRuntime(budget=16, max_ctx=65536)
        return DflashRuntime(max_ctx=98304)
    if host.vram_gb < 48:
        return DflashRuntime(max_ctx=131072)
    return DflashRuntime(max_ctx=131072)


def candidate_configs(host: HostFacts) -> list[DflashRuntime]:
    """Empirical bracket worth testing on this host.

    Returns ~6-12 DflashRuntime configs around runtime_from_host(host)
    — small enough to sweep in under 30 min on a 24 GB rig, large
    enough that the empirical winner usually beats the heuristic prior
    on the host's real workload.

    Per-tier brackets:
      <12 GB  → base only (no sweep — model barely fits)
      12-21   → 3 configs: budget × {smaller, equal, larger}
      22-31   → ~8 configs: budget × {16, 22, 32}  ×  kv × {tq3_0, q8_0}
      32-47   → ~6 configs: budget × {22, 32}  ×  kv × {tq3_0, q8_0, f16}
      ≥48     → ~6 configs: budget × {32, 48}  ×  kv × {tq3_0, q8_0, f16}

    The base config (from runtime_from_host) is always the seed; every
    returned candidate is a `replace()` of that base so the 11
    DflashRuntime fields outside the swept axes stay aligned with the
    heuristic. Duplicates are dropped via a (budget, max_ctx,
    cache_type_k, cache_type_v) tuple-set so a swept axis that happens
    to land on the heuristic value doesn't generate a redundant cell.
    """
    base = runtime_from_host(host)

    # <12 GB → base only. Model barely fits; sweeping risks OOM more
    # than it improves throughput. Caller is expected to treat a
    # 1-config "sweep" as a smoke test, not a tuning pass.
    if host.vram_gb <= 0 or host.vram_gb < 12:
        return [base]

    candidates: list[DflashRuntime] = []
    seen: set[tuple[int, int, str, str]] = set()

    def add(runtime: DflashRuntime) -> None:
        key = (
            runtime.budget,
            runtime.max_ctx,
            runtime.cache_type_k,
            runtime.cache_type_v,
        )
        if key in seen:
            return
        seen.add(key)
        candidates.append(runtime)

    # The base config is always included so the sweep validates the
    # heuristic prior alongside the bracket.
    add(base)

    if host.vram_gb < 22:
        # 12-21 GB: tight fit. Budget bracket only, keep ctx and KV at base.
        # Three cells in total (including base).
        for budget in (8, 16, 22):
            add(replace(base, budget=budget))
        return candidates

    if host.vram_gb < 32:
        # 22-31 GB: the 24 GB consumer flagships (3090/4090/5090). KV
        # quantization is the highest-leverage knob here (tq3_0 frees
        # several GB of VRAM at the cost of a tiny capability hit; q8_0
        # is the safer middle ground). Budget sweep covers the small
        # decode-throughput sensitivity around 22.
        for budget in (16, 22, 32):
            for kv in ("tq3_0", "q8_0"):
                add(replace(base, budget=budget, cache_type_k=kv, cache_type_v=kv))
        return candidates

    if host.vram_gb < 48:
        # 32-47 GB: RTX 6000 Ada / A100 40 GB. f16 KV is in budget here,
        # so we add it to the matrix. Drop budget=16 since these GPUs
        # have enough decode bandwidth that the smaller budget isn't
        # the win it is on the 24 GB tier.
        for budget in (22, 32):
            for kv in ("tq3_0", "q8_0", "f16"):
                add(replace(base, budget=budget, cache_type_k=kv, cache_type_v=kv))
        return candidates

    # ≥48 GB: A100 80 GB / H100 / RTX 6000 Pro. Drop budget=22 and pick
    # up budget=48 — larger trees pay off on the higher-end cards.
    for budget in (32, 48):
        for kv in ("tq3_0", "q8_0", "f16"):
            add(replace(base, budget=budget, cache_type_k=kv, cache_type_v=kv))
    return candidates


def recommend_preset(host: HostFacts) -> str | None:
    """Pick a default preset for first-run install. None = ask the user.

    Tiers follow the model size catalog: 22 GB+ → Qwen3.6-27B (the
    Lucebox default), 16-21 GB → Laguna-XS.2 (small target-only). Below
    16 GB we punt and let the user pick explicitly — the registered
    presets all need at least 16 GB to run usefully.
    """
    if host.vram_gb >= 22:
        return "qwen3.6-27b"
    if host.vram_gb >= 16:
        return "laguna-xs.2"
    return None
