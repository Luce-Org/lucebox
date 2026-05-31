"""Heuristic autotune: VRAM tier → DflashRuntime defaults + preset picker.

The recommended runtime is computed from HostFacts (VRAM, is_wsl) and the
recommended preset from VRAM tier alone. Both are stateless — they take
HostFacts in and return a fresh value — so the CLI can apply them with
``lucebox autotune --apply`` without holding any global state.

Profiles
--------

``lucebox autotune --sweep --profile <name>`` selects a workload-specific
bracket + winner-pick strategy. The default profile is ``heuristic`` —
preset-agnostic, ranks cells by mean ``decode_tokens_per_sec``. The
``coding-agent-loop`` profile brackets per architecture (gemma4 vs
qwen3.6/laguna): for gemma4 the KV-quant axis is dead (KV is hardcoded
F16 in the backend; see ``server/src/gemma4/gemma4_loader.cpp``), so
the bracket sweeps ``max_ctx × fa_window × budget`` instead (pflash is
kept off in the sweep: pflash requires both a drafter file AND
prefix_cache_slots > 0; with the default prefix_cache_slots=0 all
KV chunks are forced → zero compression). For qwen3.6/laguna the KV
quant axis is live and the bracket includes it. Winner is picked by
composite ``pass-rate then speed`` against the agent_recorded multi-turn
fixture — the sweep driver in ``sweep.py`` calls the profile's
``scorer`` per cell.

Profiles are intentionally a lightweight dataclass + module-level
registry: add a profile by appending to ``PROFILES``; no plugin
machinery, no entry points. A second flavor (e.g. ``research-loop``)
should follow when there's a second workload worth profiling.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable

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
        # tq3_0 is required at 98K on 23 GB cards: model (~18-19 GB) +
        # q8_0 KV cache at 98K (~5-6 GB) = 24-25 GB → OOM.
        # tq3_0 KV (~2 GB) leaves ~3 GB headroom. Confirmed on bragi
        # (RTX 5090 Laptop, 23 GB VRAM) 2026-05-30 — q8_0 timed out on
        # every 98K cell; all tq3_0 cells passed. See
        # docs/experiments/qwen3.6-27b-coding-agent-loop-sweep-bragi-2026-05-30.md
        if host.is_wsl:
            # Bumped from max_ctx=65536 → 98304 on 2026-05-30 after the
            # coding-agent-loop sweep on sindri proved 98K serves real
            # 90K-token agentic prompts with ~3 GB VRAM headroom and no
            # CUDA VMM failures. See
            # docs/experiments/gemma4-26b-coding-agent-loop-sweep-2026-05-30.md.
            # The original 65K cap cited unverified VMM failures —
            # bisect history showed no commit reproducing them.
            return DflashRuntime(
                budget=16, max_ctx=98304,
                cache_type_k="tq3_0", cache_type_v="tq3_0",
            )
        return DflashRuntime(
            max_ctx=98304,
            cache_type_k="tq3_0", cache_type_v="tq3_0",
        )
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


# ── Profile abstraction ────────────────────────────────────────────────────
#
# Each Profile owns:
#   * a candidate_configs(host, preset) → list[DflashRuntime]
#   * a scorer key string consumed by sweep.py to pick the right
#     measurement + winner-selection path. We keep the scorer as a
#     string discriminator (rather than a Callable) because the scorer
#     needs runtime imports (subprocess, HTTP client, snapshot parser)
#     that don't belong in autotune.py — sweep.py owns those.
#
# The candidate function is preset-aware so the same profile can pick
# a different bracket per architecture. Gemma4 backends ignore the
# cache_type knob; qwen3.6 and laguna respect it. Per-preset bracket
# code lives below the Profile class.


@dataclass(frozen=True, slots=True)
class Profile:
    name: str
    # Human-readable description, surfaced by `lucebox autotune --list-profiles`.
    description: str
    # Builder taking (host, preset_name) and returning a list of
    # candidate DflashRuntime configs to sweep. Preset is the active
    # ``model.preset`` string (empty when unset).
    candidate_configs: Callable[[HostFacts, str], list[DflashRuntime]]
    # Discriminator for which scorer the sweep driver invokes.
    # Values: "decode_tps_snapshot" (legacy heuristic path) or
    # "agent_replay_pass_rate" (coding-agent-loop). Adding a new
    # scorer means landing a new branch in sweep.py::run_sweep.
    scorer: str = "decode_tps_snapshot"


# ── coding-agent-loop bracket builders ─────────────────────────────────────


_GEMMA_PRESETS: frozenset[str] = frozenset({"gemma-4-26b"})
_QWEN_PRESETS: frozenset[str] = frozenset({"qwen3.6-27b", "laguna-xs.2"})


def _is_gemma_preset(preset: str) -> bool:
    return preset in _GEMMA_PRESETS or preset.startswith("gemma")


def _coding_agent_loop_gemma_bracket(
    host: HostFacts, base: DflashRuntime
) -> list[DflashRuntime]:
    """Gemma4 bracket: ``max_ctx × fa_window × budget × pflash_mode``.

    KV-quant axis is intentionally absent — gemma4_loader.cpp forces
    F16 regardless of ``cache_type_k/v``. The 24 GB tier targets up to
    131K max_ctx; higher tiers also peak at 131K because that is the
    model's practical ceiling (196K KV doesn't fit alongside the Q4_K_M
    weights even on 48 GB cards — model+KV would be ~33 GB).

    131K is confirmed viable on 23-24 GB VRAM: model (~14-15 GB) +
    F16 KV at 131K (~7-8 GB) ≈ 22-23 GB total. Validated on bragi
    (RTX 5090 Laptop, 23 GB) 2026-05-30 — all 6 131K cells passed.
    Earlier sindri run (RTX 3090 Ti, 24 GB) appeared to fail at 131K,
    but that was a fixture-picker issue (selected the 100K case which
    expanded to ~130K real tokens > 126976 server ceiling), not VRAM.
    See docs/experiments/gemma4-26b-coding-agent-loop-sweep-bragi-2026-05-30.md.

    Per-tier:
      <22 GB → base only (model barely fits; sweeping risks OOM)
      22-31  → 12 cells: max_ctx × {98K, 131K}, fa_window × {0, 2048},
               budget × {16, 22, 32}, pflash off (pflash needs both a
               drafter file AND prefix_cache_slots > 0 to compress
               anything; with prefix_cache_slots=0 all chunks are
               forced → zero compression. Enable manually for real
               multi-turn sessions; see pflash A/B test 2026-05-31)
      ≥32     → same shape, defaulting max_ctx=131K (with VRAM headroom
               to take fa_window=0 paths)
    """
    candidates: list[DflashRuntime] = []
    seen: set[tuple[int, int, int, str]] = set()

    def add(rt: DflashRuntime) -> None:
        key = (rt.max_ctx, rt.fa_window, rt.budget, rt.prefill_mode)
        if key in seen:
            return
        seen.add(key)
        candidates.append(rt)

    if host.vram_gb < 22:
        add(base)
        return candidates

    # 22+ GB tier: gemma's 131K ceiling is achievable. Bracket the
    # interesting axes; keep cardinality modest so the full sweep
    # finishes in ~20 min.
    for max_ctx in (98_304, 131_072):
        for fa_window in (0, 2048):
            for budget in (16, 22, 32):
                add(
                    replace(
                        base,
                        max_ctx=max_ctx,
                        fa_window=fa_window,
                        budget=budget,
                        cache_type_k="",
                        cache_type_v="",
                        prefill_mode="off",
                    )
                )
    return candidates


def _coding_agent_loop_qwen_bracket(
    host: HostFacts, base: DflashRuntime
) -> list[DflashRuntime]:
    """Qwen3.6 / laguna bracket: ``max_ctx × cache_type × budget × fa_window``.

    KV-quant axis is the high-leverage knob on this family — tq3_0
    frees several GB of VRAM relative to q8_0, which unlocks larger
    max_ctx on 24 GB cards. The PFlash axis stays off here too
    (operator must configure a drafter file before flipping).
    """
    candidates: list[DflashRuntime] = []
    seen: set[tuple[int, int, str, int]] = set()

    def add(rt: DflashRuntime) -> None:
        key = (rt.max_ctx, rt.budget, rt.cache_type_k, rt.fa_window)
        if key in seen:
            return
        seen.add(key)
        candidates.append(rt)

    if host.vram_gb < 22:
        add(base)
        return candidates

    if host.vram_gb < 32:
        # 24 GB tier: tq3_0 vs q8_0 at 65K; tq3_0-only at 98K.
        # q8_0 at 98K OOMs on 23 GB cards: model (~18-19 GB) +
        # q8_0 KV at 96K (~5-6 GB) = 24-25 GB. Verified on bragi
        # (RTX 5090 Laptop 23 GB) 2026-05-30 — all q8_0/98K cells
        # timed out. See
        # docs/experiments/qwen3.6-27b-coding-agent-loop-sweep-bragi-2026-05-30.md
        for max_ctx in (65_536, 98_304):
            kvs: tuple[str, ...] = ("tq3_0", "q8_0") if max_ctx <= 65_536 else ("tq3_0",)
            for kv in kvs:
                for budget in (16, 22, 32):
                    add(
                        replace(
                            base,
                            max_ctx=max_ctx,
                            budget=budget,
                            cache_type_k=kv,
                            cache_type_v=kv,
                            fa_window=0,
                            prefill_mode="off",
                        )
                    )
        return candidates

    # 32+ GB: full f16 in budget, push max_ctx to 131K.
    for kv in ("tq3_0", "q8_0", "f16"):
        for budget in (22, 32):
            add(
                replace(
                    base,
                    max_ctx=131_072,
                    budget=budget,
                    cache_type_k=kv,
                    cache_type_v=kv,
                    fa_window=0,
                    prefill_mode="off",
                )
            )
    return candidates


def _coding_agent_loop_candidates(host: HostFacts, preset: str) -> list[DflashRuntime]:
    """Dispatch coding-agent-loop bracket per architecture.

    Both gemma4 and qwen3.6/laguna paths use the heuristic base from
    :func:`runtime_from_host` as their seed; the per-arch builder
    decides which axes to vary around that base.
    """
    base = runtime_from_host(host)
    if _is_gemma_preset(preset):
        return _coding_agent_loop_gemma_bracket(host, base)
    # Default to the qwen-shape bracket — the only other supported
    # presets today are qwen3.6 and laguna, and the bracket
    # gracefully degrades to base-only on tiny VRAM tiers.
    return _coding_agent_loop_qwen_bracket(host, base)


def _heuristic_candidates(host: HostFacts, preset: str) -> list[DflashRuntime]:
    """Legacy preset-agnostic bracket (the original ``candidate_configs``).

    ``preset`` is accepted but ignored — the heuristic profile sweeps
    KV-quant axes for every preset, which is wrong for gemma4 (no-op)
    but preserves the existing behavior for anyone still calling the
    bare ``candidate_configs`` entry point.
    """
    return candidate_configs(host)


PROFILES: dict[str, Profile] = {
    "heuristic": Profile(
        name="heuristic",
        description=(
            "Preset-agnostic bracket; ranks cells by mean decode_tokens_per_sec "
            "across luce-bench level1 areas."
        ),
        candidate_configs=_heuristic_candidates,
        scorer="decode_tps_snapshot",
    ),
    "coding-agent-loop": Profile(
        name="coding-agent-loop",
        description=(
            "Architecture-aware bracket for agentic coding workloads; "
            "ranks cells by pass-rate on the agent_recorded multi-turn fixture, "
            "then by completion_tokens / wall_seconds."
        ),
        candidate_configs=_coding_agent_loop_candidates,
        scorer="agent_replay_pass_rate",
    ),
}


def get_profile(name: str) -> Profile:
    """Return the registered profile or raise ``KeyError``.

    Public surface for sweep.py + cli.py — they call this with the
    user's ``--profile`` argument; an unknown name produces a clear
    error rather than silently falling back to the heuristic.
    """
    if name not in PROFILES:
        known = ", ".join(sorted(PROFILES))
        raise KeyError(f"unknown profile {name!r}; known: {known}")
    return PROFILES[name]


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
