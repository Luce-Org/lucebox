"""Safe, explainable optimization planning for a Lucebox machine.

The planner combines host facts, the selected model preset, and locally
installed optimization assets. It intentionally does not claim to benchmark
the workload: every automatic decision is a conservative rule backed by a
known engine capability, and the CLI prints the reason for every on/off choice.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from lucebox.placement import PlacementPlan, automatic_placement
from lucebox.types import Config, DflashRuntime, HostFacts


@dataclass(frozen=True, slots=True)
class OptimizationDecision:
    """One user-visible optimization decision and its explanation."""

    name: str
    enabled: bool
    available: bool
    reason: str


@dataclass(frozen=True, slots=True)
class OptimizationPlan:
    """Resolved optimization and accelerator-placement decisions."""

    runtime: DflashRuntime
    model_name: str
    placement: PlacementPlan
    dflash: OptimizationDecision
    pflash: OptimizationDecision
    kvflash: OptimizationDecision
    spark: OptimizationDecision
    needs_optimizer_drafter: bool = False

    @property
    def decisions(self) -> tuple[OptimizationDecision, ...]:
        return (self.dflash, self.pflash, self.kvflash, self.spark)

    @property
    def active_names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.decisions if item.enabled)


def _budget_for_host(host: HostFacts) -> int:
    """Use the documented per-architecture DDTree sweet spot when known."""
    if host.is_wsl and 22 <= host.vram_gb < 32:
        return 16
    if host.gpu_sm == "gfx1100":
        return 8
    if host.gpu_sm == "120":
        return 40
    return 22


def runtime_from_host(host: HostFacts) -> DflashRuntime:
    """Pick conservative context/cache defaults from the selected GPU.

    These tiers target the Qwen3.6-27B Q4_K_M stack. ``automatic_plan`` adds
    model capability and installed-asset decisions on top of this baseline.
    Only the selected/primary GPU is counted; multiple cards are never summed
    until the engine has an explicit layer-split plan.
    """
    budget = _budget_for_host(host)
    if host.vram_gb <= 0:
        return DflashRuntime(budget=budget)
    if host.vram_gb < 12:
        return DflashRuntime(budget=budget, max_ctx=4096)
    if host.vram_gb < 22:
        return DflashRuntime(budget=budget, max_ctx=32768)
    if host.vram_gb < 32:
        # Cache types stay model-owned. In particular, tq3_0 is unsafe for
        # Laguna and was removed from the engine's automatic defaults in
        # July 2026. The CLI must not silently reintroduce it here.
        # The earlier 98K WSL result depended on a blanket tq3_0 cache.
        # Automatic now preserves model-family cache types, so keep extra
        # headroom for WSL's virtualization overhead instead.
        max_ctx = 65536 if host.is_wsl else 98304
        return DflashRuntime(budget=budget, max_ctx=max_ctx)
    return DflashRuntime(budget=budget, max_ctx=131072)


def draft_available(cfg: Config, preset: object) -> bool:
    """Whether the selected preset's speculative decoder is installed."""
    from lucebox.download import local_artifact_present

    draft_file = cfg.model.draft_file or getattr(preset, "draft_file", None)
    if draft_file:
        return local_artifact_present(cfg.models_dir / "draft" / str(draft_file))
    speculator_dir = getattr(preset, "speculator_dir", None)
    if not speculator_dir:
        return False
    root = cfg.models_dir / "draft" / str(speculator_dir)
    if not root.is_dir():
        return False
    try:
        return any(
            local_artifact_present(candidate)
            for pattern in ("*.gguf", "*.safetensors")
            for candidate in root.rglob(pattern)
        )
    except OSError:
        return False


def placement_for_runtime(cfg: Config, runtime: DflashRuntime) -> PlacementPlan:
    """Resolve placement after an Advanced-mode optimization edit."""
    from lucebox import download as download_mod

    preset = download_mod.PRESETS.get(cfg.model.preset)
    if preset is None:
        return automatic_placement(
            cfg,
            runtime,
            object(),
            has_draft=False,
            optimizer_drafter_available=False,
        )
    return automatic_placement(
        cfg,
        runtime,
        preset,
        has_draft=runtime.speculative_decode and draft_available(cfg, preset),
        optimizer_drafter_available=download_mod.optimizer_drafter_installed(cfg),
    )


def _cap_exact_context(
    runtime: DflashRuntime,
    *,
    headroom_gb: float,
    arch: str,
) -> DflashRuntime:
    """Keep exact-cache families inside conservative primary-GPU headroom.

    Qwen families have an automatic bounded-residency path below; other
    families currently keep their exact cache in Automatic mode. Those models
    get a smaller context when their weights/draft leave little working room.
    """
    if arch in {"qwen35", "qwen35moe"}:
        return runtime
    if headroom_gb < 4:
        return replace(runtime, max_ctx=min(runtime.max_ctx, 8192))
    if headroom_gb < 7:
        return replace(runtime, max_ctx=min(runtime.max_ctx, 32768))
    return runtime


CapacityAdjustment = Literal["pflash", "kvflash-qk", "kvflash-off", "dflash"]


def _uses_optimizer_scorer(runtime: DflashRuntime) -> bool:
    return runtime.prefill_mode != "off" or (
        runtime.kvflash != "off" and runtime.kvflash_policy == "drafter"
    )


def _without_unused_scorer(runtime: DflashRuntime) -> DflashRuntime:
    """Clear the scorer path once no enabled feature consumes it."""
    return runtime if _uses_optimizer_scorer(runtime) else replace(runtime, prefill_drafter="")


def _capacity_fallbacks(
    runtime: DflashRuntime,
    *,
    architecture: str,
) -> tuple[tuple[DflashRuntime, CapacityAdjustment], ...]:
    """Return progressively smaller optional stacks, in performance order.

    Installed optional assets must never make a fitting target unrunnable.
    Placement gets the full recommended stack first; only when that has no
    valid device plan do we remove independent optional workloads. Qwen dense
    can retain bounded KV residency without the 1.2 GB scorer by switching to
    its validated QK policy.
    """
    candidates: list[tuple[DflashRuntime, CapacityAdjustment]] = []
    current = runtime

    if current.prefill_mode != "off":
        defaults = DflashRuntime()
        current = _without_unused_scorer(
            replace(
                current,
                prefill_mode="off",
                prefill_keep_ratio=defaults.prefill_keep_ratio,
                prefill_threshold=defaults.prefill_threshold,
                prefill_drafter=(
                    current.prefill_drafter
                    if current.kvflash != "off" and current.kvflash_policy == "drafter"
                    else ""
                ),
            )
        )
        candidates.append((current, "pflash"))

    if current.kvflash != "off" and current.kvflash_policy == "drafter":
        if architecture == "qwen35":
            current = replace(current, kvflash_policy="qk")
            action: CapacityAdjustment = "kvflash-qk"
        else:
            current = replace(current, kvflash="off")
            action = "kvflash-off"
        current = _without_unused_scorer(current)
        candidates.append((current, action))

    if current.speculative_decode:
        current = replace(current, speculative_decode=False, lazy=False)
        candidates.append((current, "dflash"))

    return tuple(candidates)


def automatic_plan(
    cfg: Config,
    *,
    optimizer_drafter_available: bool | None = None,
) -> OptimizationPlan:
    """Resolve the recommended profile for the selected model and primary GPU.

    Automatic mode favors exact/full-cache execution whenever it fits. PFlash
    is enabled only for its validated Qwen3.6 dense profile. KVFlash is enabled
    only under real memory pressure and only on qwen-family paths with a tuned
    scorer. Spark is enabled only for supported MoE models that would otherwise
    leave too little VRAM headroom. Advanced mode can override these choices.
    """
    from lucebox import download as download_mod

    runtime = runtime_from_host(cfg.host)
    preset = download_mod.PRESETS.get(cfg.model.preset)
    if optimizer_drafter_available is None:
        optimizer_drafter_available = download_mod.optimizer_drafter_installed(cfg)

    if preset is None:
        dflash = OptimizationDecision(
            "DFlash",
            False,
            False,
            "choose a model first; it activates when that model has a matching draft",
        )
        unavailable = "choose a model first"
        placement = automatic_placement(
            cfg,
            runtime,
            object(),
            has_draft=False,
            optimizer_drafter_available=optimizer_drafter_available,
        )
        return OptimizationPlan(
            runtime=runtime,
            model_name=cfg.model.preset or "not selected",
            placement=placement,
            dflash=dflash,
            pflash=OptimizationDecision("PFlash", False, False, unavailable),
            kvflash=OptimizationDecision("KVFlash", False, False, unavailable),
            spark=OptimizationDecision("Spark", False, False, unavailable),
        )

    runtime = replace(runtime, max_ctx=min(runtime.max_ctx, preset.native_context))
    vram_known = cfg.host.vram_gb > 0
    headroom = cfg.host.vram_gb - preset.approx_total_gb if vram_known else 0
    if vram_known:
        runtime = _cap_exact_context(
            runtime,
            headroom_gb=headroom,
            arch=preset.architecture,
        )

    has_draft = draft_available(cfg, preset)
    runtime = replace(runtime, speculative_decode=has_draft)
    if has_draft:
        dflash_reason = "matching speculative draft is available for this model"
    elif preset.speculator_dir:
        dflash_reason = "optional model speculator is not installed"
    elif preset.has_draft:
        dflash_reason = "matching speculative draft is not installed"
    else:
        dflash_reason = "no compatible speculative draft is published for this model"
    dflash = OptimizationDecision("DFlash", has_draft, has_draft, dflash_reason)

    long_context = runtime.max_ctx >= 32768

    # PFlash automatic mode is deliberately narrow: this is the profile with
    # end-to-end long-context quality validation. Other architectures remain
    # unavailable until the engine exposes a compatible compression path.
    pflash_supported = preset.architecture == "qwen35"
    pflash_available = pflash_supported and long_context
    pflash_profile = pflash_available and vram_known
    pflash_enabled = pflash_profile and optimizer_drafter_available
    if pflash_enabled:
        pflash_reason = "long prompts use the installed scorer above 32K tokens"
        runtime = replace(
            runtime,
            prefill_mode="auto",
            prefill_keep_ratio=0.10,
            prefill_threshold=32768,
            prefill_drafter=download_mod.optimizer_drafter_container_path(),
        )
    elif pflash_profile:
        pflash_reason = "shared 1.2 GB scorer is not installed"
    elif not pflash_supported:
        pflash_reason = "this model architecture has no PFlash compression path"
    elif not long_context:
        pflash_reason = "the safe context for this GPU is below PFlash's long-prompt threshold"
    else:
        pflash_reason = "the conservative automatic profile keeps full prefill for this model"
    pflash = OptimizationDecision(
        "PFlash",
        pflash_enabled,
        pflash_available,
        pflash_reason,
    )

    # Bounded KV residency changes long-context retrieval semantics, so it is
    # not a blanket speed toggle. Use it only when the selected stack leaves
    # less than 5 GB for KV/compute and a tuned qwen-family policy is available.
    kvflash_profile = (
        vram_known
        and long_context
        and headroom < 5
        and preset.architecture in {"qwen35", "qwen35moe"}
    )
    kvflash_enabled = kvflash_profile and (
        optimizer_drafter_available or preset.architecture == "qwen35"
    )
    if kvflash_enabled:
        policy: Literal["drafter", "qk"] = "drafter" if optimizer_drafter_available else "qk"
        runtime = replace(runtime, kvflash="auto", kvflash_policy=policy)
        if optimizer_drafter_available and not runtime.prefill_drafter:
            runtime = replace(
                runtime,
                prefill_drafter=download_mod.optimizer_drafter_container_path(),
            )
        kvflash_reason = f"only {headroom} GB VRAM headroom; {policy} policy bounds KV residency"
    elif kvflash_profile:
        kvflash_reason = "memory pressure detected, but no quality-safe scorer is installed"
    elif not long_context:
        kvflash_reason = "the selected context does not need bounded KV residency"
    elif preset.architecture not in {"qwen35", "qwen35moe"}:
        kvflash_reason = "cross-tokenizer residency remains an Advanced choice"
    else:
        kvflash_reason = "full KV cache fits; exact full-cache execution is preferred"
    kvflash = OptimizationDecision(
        "KVFlash",
        kvflash_enabled,
        long_context,
        kvflash_reason,
    )

    spark_available = preset.architecture in {"laguna", "qwen35moe"}
    spark_pressure = spark_available and vram_known and headroom < 6
    # Cold experts live in host RAM. A 20–22 GB preset plus the OS, runtime,
    # KV spill, and working buffers is not a safe automatic fit below 32 GB.
    # Advanced mode can still let an informed operator opt in.
    spark_host_ready = cfg.host.ram_gb >= 32
    spark_enabled = spark_pressure and spark_host_ready
    runtime = replace(runtime, spark=spark_enabled)
    if spark_enabled:
        spark_reason = f"MoE weights leave {headroom} GB headroom; expert residency self-tunes"
    elif spark_pressure and cfg.host.ram_gb <= 0:
        spark_reason = "GPU memory is tight, but host RAM is unknown; automatic offload stays off"
    elif spark_pressure:
        spark_reason = (
            f"GPU memory is tight, but {cfg.host.ram_gb} GB host RAM is below "
            "the 32 GB automatic offload floor"
        )
    elif spark_available and vram_known:
        spark_reason = "the model fits the primary GPU; all-GPU execution is faster and simpler"
    elif spark_available:
        spark_reason = "GPU memory is unknown; automatic mode will not assume offload"
    else:
        spark_reason = "this is not a Spark-compatible MoE architecture"
    spark = OptimizationDecision("Spark", spark_enabled, spark_available, spark_reason)

    needs_optimizer_drafter = not optimizer_drafter_available and (
        pflash_profile or (kvflash_profile and preset.architecture == "qwen35moe")
    )
    placement = automatic_placement(
        cfg,
        runtime,
        preset,
        has_draft=has_draft,
        optimizer_drafter_available=optimizer_drafter_available,
    )
    capacity_adjustments: set[CapacityAdjustment] = set()
    if not placement.runnable:
        for fallback_runtime, adjustment in _capacity_fallbacks(
            runtime,
            architecture=preset.architecture,
        ):
            capacity_adjustments.add(adjustment)
            placement = automatic_placement(
                cfg,
                fallback_runtime,
                preset,
                has_draft=has_draft and fallback_runtime.speculative_decode,
                optimizer_drafter_available=optimizer_drafter_available,
            )
            runtime = fallback_runtime
            if placement.runnable:
                break

    adjusted = placement.optimization_runtime
    if dflash.enabled and not adjusted.speculative_decode:
        reason = (
            "disabled because the draft would exceed the safe GPU memory budget"
            if "dflash" in capacity_adjustments
            else "disabled because the selected target placement has no compatible draft path"
        )
        dflash = OptimizationDecision(
            "DFlash",
            False,
            dflash.available,
            reason,
        )
    if pflash.enabled and adjusted.prefill_mode == "off":
        reason = (
            "disabled because the scorer would exceed the safe GPU memory budget"
            if "pflash" in capacity_adjustments
            else "disabled because the selected target placement has no compatible compression path"
        )
        pflash = OptimizationDecision(
            "PFlash",
            False,
            pflash.available,
            reason,
        )
    if kvflash.enabled and adjusted.kvflash == "off":
        reason = (
            "disabled because its scorer would exceed the safe GPU memory budget"
            if "kvflash-off" in capacity_adjustments
            else "disabled because the selected target placement has no compatible KV path"
        )
        kvflash = OptimizationDecision(
            "KVFlash",
            False,
            kvflash.available,
            reason,
        )
    elif "kvflash-qk" in capacity_adjustments:
        kvflash = OptimizationDecision(
            "KVFlash",
            True,
            kvflash.available,
            "memory is tight; QK policy avoids a separate scorer allocation",
        )
    if spark.enabled and not adjusted.spark:
        spark = OptimizationDecision(
            "Spark",
            False,
            spark.available,
            "disabled because Automatic does not compose Spark with target layer splitting",
        )
    if needs_optimizer_drafter:
        # Do not ask a first-time user to download 1.2 GB only to discard the
        # scorer during capacity fallback. The scorer-present branch cannot
        # recurse again because its availability makes ``needs_*`` false.
        scorer_plan = automatic_plan(cfg, optimizer_drafter_available=True)
        needs_optimizer_drafter = scorer_plan.placement.runnable and _uses_optimizer_scorer(
            scorer_plan.runtime
        )
    return OptimizationPlan(
        runtime=adjusted,
        model_name=preset.name,
        placement=placement,
        dflash=dflash,
        pflash=pflash,
        kvflash=kvflash,
        spark=spark,
        needs_optimizer_drafter=needs_optimizer_drafter,
    )
