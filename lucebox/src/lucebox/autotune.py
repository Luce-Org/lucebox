"""Safe, explainable optimization planning for a Lucebox machine.

The planner combines host facts, the selected model preset, and locally
installed optimization assets. It intentionally does not claim to benchmark
the workload: every automatic decision is a conservative rule backed by a
known engine capability, and the CLI prints the reason for every on/off choice.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

from lucebox.capabilities import (
    FeatureCapability,
    KvFlashCapability,
    ModelOptimizationProfile,
    Qualification,
    model_profile,
)
from lucebox.placement import PlacementPlan, automatic_placement
from lucebox.topology import from_config
from lucebox.types import Config, DflashRuntime, HostFacts


@dataclass(frozen=True, slots=True)
class OptimizationDecision:
    """One user-visible optimization decision and its explanation."""

    name: str
    enabled: bool
    available: bool
    reason: str
    qualification: Qualification = Qualification.QUALIFIED


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
    prefill_alternative: OptimizationDecision | None = None
    prefill_strategy: str = "not selected"
    decode_strategy: str = "not selected"
    kv_strategy: str = "not selected"
    needs_optimizer_drafter: bool = False

    @property
    def decisions(self) -> tuple[OptimizationDecision, ...]:
        core = (self.dflash, self.pflash, self.kvflash, self.spark)
        if self.prefill_alternative is None:
            return core
        return (*core, self.prefill_alternative)

    @property
    def active_names(self) -> tuple[str, ...]:
        return tuple(item.name for item in self.decisions if item.enabled)

    @property
    def phase_strategies(self) -> tuple[tuple[str, str], ...]:
        """Resolved implementation for each performance-critical phase."""
        return (
            ("Prefill", self.prefill_strategy),
            ("Decode", self.decode_strategy),
            ("KV cache", self.kv_strategy),
        )


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
    profile: ModelOptimizationProfile,
    backend: str,
) -> DflashRuntime:
    """Keep exact-cache families inside conservative primary-GPU headroom.

    Models with a qualified automatic bounded-residency path are handled by
    the planner below. Other models keep exact cache semantics in Automatic
    mode and receive a smaller context when weights leave little working room.
    """
    kvflash = profile.kvflash
    if kvflash is not None and kvflash.feature.automatic_on(backend):
        return runtime
    if headroom_gb < 4:
        return replace(runtime, max_ctx=min(runtime.max_ctx, 8192))
    if headroom_gb < 7:
        return replace(runtime, max_ctx=min(runtime.max_ctx, 32768))
    return runtime


CapacityAdjustment = Literal[
    "pflash",
    "kvflash-scorerless",
    "kvflash-off",
    "dflash",
]


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
    kvflash: KvFlashCapability | None,
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
        scorerless_policy = kvflash.scorerless_policy if kvflash is not None else None
        if scorerless_policy is not None:
            current = replace(current, kvflash_policy=scorerless_policy)
            action: CapacityAdjustment = "kvflash-scorerless"
        else:
            current = replace(current, kvflash="off")
            action = "kvflash-off"
        current = _without_unused_scorer(current)
        candidates.append((current, action))

    if current.speculative_decode:
        current = replace(current, speculative_decode=False, lazy=False)
        candidates.append((current, "dflash"))

    return tuple(candidates)


def _selected_backend(cfg: Config) -> str:
    """Return the backend of the runtime's primary device."""
    primary = from_config(cfg).primary
    if primary is not None:
        return primary.backend
    # Config snapshots predating the inventory fields can still carry the
    # selected generic vendor. Placement remains blocked when the configured
    # image and inventory disagree; this fallback is only for feature support
    # reporting and migration of those snapshots.
    if cfg.host.gpu_vendor == "nvidia":
        return "cuda"
    if cfg.host.gpu_vendor == "amd":
        return "hip"
    return ""


def _qualification(feature: FeatureCapability | None, backend: str) -> Qualification:
    if feature is None:
        return Qualification.UNAVAILABLE
    return feature.qualification_on(backend)


def _preview_reason(feature: FeatureCapability, backend: str) -> str:
    backend_label = "CUDA" if backend == "cuda" else "HIP"
    return (
        f"{feature.label} is available as a {backend_label} preview; "
        "Automatic keeps the qualified baseline"
    )


def automatic_plan(
    cfg: Config,
    *,
    optimizer_drafter_available: bool | None = None,
) -> OptimizationPlan:
    """Resolve the recommended profile for the selected model and primary GPU.

    Automatic mode favors exact/full-cache execution whenever it fits. The
    model contract limits each feature to legal, qualified backends; measured
    memory pressure decides whether a qualified feature is useful. Device
    placement is then resolved independently from the detected topology.
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
            Qualification.UNAVAILABLE,
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
            pflash=OptimizationDecision(
                "PFlash", False, False, unavailable, Qualification.UNAVAILABLE
            ),
            kvflash=OptimizationDecision(
                "KVFlash", False, False, unavailable, Qualification.UNAVAILABLE
            ),
            spark=OptimizationDecision(
                "Spark", False, False, unavailable, Qualification.UNAVAILABLE
            ),
        )

    profile = model_profile(preset.name)
    if profile.architecture != preset.architecture:
        raise ValueError(
            f"optimization profile for {preset.name!r} targets {profile.architecture!r}, "
            f"but the model catalog declares {preset.architecture!r}"
        )
    backend = _selected_backend(cfg)
    runtime = replace(runtime, max_ctx=min(runtime.max_ctx, preset.native_context))
    vram_known = cfg.host.vram_gb > 0
    headroom = cfg.host.vram_gb - preset.approx_total_gb if vram_known else 0
    if vram_known:
        runtime = _cap_exact_context(
            runtime,
            headroom_gb=headroom,
            profile=profile,
            backend=backend,
        )

    has_draft = draft_available(cfg, preset)
    decode_feature = profile.speculative_decode
    decode_qualification = _qualification(decode_feature, backend)
    decode_supported = decode_feature is not None and decode_feature.available_on(backend)
    dflash_available = has_draft and decode_supported
    dflash_enabled = (
        dflash_available
        and decode_feature is not None
        and decode_feature.automatic_on(backend)
    )
    runtime = replace(runtime, speculative_decode=dflash_enabled)
    if dflash_available and not dflash_enabled and decode_feature is not None:
        dflash_reason = _preview_reason(decode_feature, backend)
    elif dflash_enabled:
        dflash_reason = "matching speculative draft is available for this model"
    elif has_draft and not decode_supported:
        dflash_reason = "the installed draft is not supported by the selected backend"
    elif preset.speculator_dir:
        dflash_reason = "optional model speculator is not installed"
    elif preset.has_draft:
        dflash_reason = "matching speculative draft is not installed"
    else:
        dflash_reason = "no compatible speculative draft is published for this model"
    dflash = OptimizationDecision(
        decode_feature.label if decode_feature is not None else "DFlash",
        dflash_enabled,
        dflash_available,
        dflash_reason,
        decode_qualification,
    )

    alternative_capability = profile.deepseek_prefill
    if alternative_capability is None:
        prefill_alternative = None
    else:
        alternative_feature = alternative_capability.feature
        alternative_qualification = alternative_feature.qualification_on(backend)
        alternative_available = alternative_feature.available_on(backend)
        if alternative_available:
            alternative_reason = _preview_reason(alternative_feature, backend)
        else:
            alternative_reason = (
                "this approximate prefill path is unavailable on the selected backend"
            )
        prefill_alternative = OptimizationDecision(
            alternative_feature.label,
            False,
            alternative_available,
            alternative_reason,
            alternative_qualification,
        )

    long_context = runtime.max_ctx >= 32768

    pflash_capability = profile.pflash
    pflash_feature = pflash_capability.feature if pflash_capability is not None else None
    pflash_qualification = _qualification(pflash_feature, backend)
    pflash_supported = (
        pflash_feature is not None and pflash_feature.available_on(backend)
    )
    pflash_available = (
        pflash_supported
        and pflash_capability is not None
        and runtime.max_ctx >= pflash_capability.minimum_context
    )
    pflash_profile = (
        pflash_available
        and vram_known
        and pflash_feature is not None
        and pflash_feature.automatic_on(backend)
    )
    pflash_enabled = pflash_profile and optimizer_drafter_available
    if pflash_enabled:
        pflash_reason = "long prompts use the installed scorer above 32K tokens"
        assert pflash_capability is not None
        runtime = replace(
            runtime,
            prefill_mode="auto",
            prefill_keep_ratio=pflash_capability.keep_ratio,
            prefill_threshold=pflash_capability.minimum_context,
            prefill_drafter=download_mod.optimizer_drafter_container_path(),
        )
    elif pflash_profile:
        pflash_reason = "shared 1.2 GB scorer is not installed"
    elif pflash_available and pflash_feature is not None:
        pflash_reason = _preview_reason(pflash_feature, backend)
    elif not pflash_supported:
        pflash_reason = (
            f"this model has no PFlash production path; {profile.prefill_baseline} remains active"
        )
    elif not long_context:
        pflash_reason = "the safe context for this GPU is below PFlash's long-prompt threshold"
    else:
        pflash_reason = "the conservative automatic profile keeps full prefill for this model"
    pflash = OptimizationDecision(
        "PFlash",
        pflash_enabled,
        pflash_available,
        pflash_reason,
        pflash_qualification,
    )

    # Bounded residency affects long-context retrieval semantics, so it is
    # activated only by a model contract and only under real memory pressure.
    kvflash_capability = profile.kvflash
    kvflash_feature = (
        kvflash_capability.feature if kvflash_capability is not None else None
    )
    kvflash_qualification = _qualification(kvflash_feature, backend)
    kvflash_supported = (
        kvflash_feature is not None and kvflash_feature.available_on(backend)
    )
    kvflash_available = (
        kvflash_supported
        and kvflash_capability is not None
        and runtime.max_ctx >= kvflash_capability.minimum_context
    )
    kvflash_profile = (
        vram_known
        and kvflash_available
        and kvflash_capability is not None
        and kvflash_feature is not None
        and kvflash_feature.automatic_on(backend)
        and headroom < kvflash_capability.pressure_headroom_gb
    )
    kvflash_policy = None
    if kvflash_profile and kvflash_capability is not None:
        kvflash_policy = (
            kvflash_capability.preferred_policy
            if optimizer_drafter_available
            else kvflash_capability.scorerless_policy
        )
    kvflash_enabled = kvflash_policy is not None
    if kvflash_enabled:
        assert kvflash_policy is not None
        runtime = replace(runtime, kvflash="auto", kvflash_policy=kvflash_policy)
        if kvflash_policy == "drafter" and not runtime.prefill_drafter:
            runtime = replace(
                runtime,
                prefill_drafter=download_mod.optimizer_drafter_container_path(),
            )
        kvflash_reason = (
            f"only {headroom} GB VRAM headroom; {kvflash_policy} policy bounds KV residency"
        )
    elif kvflash_profile:
        kvflash_reason = "memory pressure detected, but no quality-safe scorer is installed"
    elif kvflash_available and kvflash_feature is not None:
        if kvflash_feature.qualification_on(backend) is Qualification.PREVIEW:
            kvflash_reason = _preview_reason(kvflash_feature, backend)
        else:
            kvflash_reason = "full KV cache fits; exact full-cache execution is preferred"
    elif kvflash_capability is not None and runtime.max_ctx < kvflash_capability.minimum_context:
        kvflash_reason = "the selected context does not need bounded KV residency"
    else:
        kvflash_reason = (
            f"this model uses {profile.kv_baseline} instead of generic KVFlash"
        )
    kvflash = OptimizationDecision(
        "KVFlash",
        kvflash_enabled,
        kvflash_available,
        kvflash_reason,
        kvflash_qualification,
    )

    spark_capability = profile.spark
    spark_feature = spark_capability.feature if spark_capability is not None else None
    spark_qualification = _qualification(spark_feature, backend)
    spark_available = spark_feature is not None and spark_feature.available_on(backend)
    spark_pressure = (
        spark_available
        and vram_known
        and spark_capability is not None
        and headroom < spark_capability.pressure_headroom_gb
    )
    # Cold experts live in host RAM. A 20–22 GB preset plus the OS, runtime,
    # KV spill, and working buffers is not a safe automatic fit below 32 GB.
    # Advanced mode can still let an informed operator opt in.
    minimum_host_ram = (
        spark_capability.minimum_host_ram_gb if spark_capability is not None else 0
    )
    spark_host_ready = cfg.host.ram_gb >= minimum_host_ram
    spark_enabled = (
        spark_pressure
        and spark_host_ready
        and spark_feature is not None
        and spark_feature.automatic_on(backend)
    )
    runtime = replace(runtime, spark=spark_enabled)
    if spark_enabled:
        spark_reason = f"MoE weights leave {headroom} GB headroom; expert residency self-tunes"
    elif spark_pressure and spark_feature is not None and not spark_feature.automatic_on(backend):
        spark_reason = _preview_reason(spark_feature, backend)
    elif spark_pressure and cfg.host.ram_gb <= 0:
        spark_reason = "GPU memory is tight, but host RAM is unknown; automatic offload stays off"
    elif spark_pressure:
        spark_reason = (
            f"GPU memory is tight, but {cfg.host.ram_gb} GB host RAM is below "
            f"the {minimum_host_ram} GB automatic offload floor"
        )
    elif spark_available and vram_known:
        spark_reason = "the model fits the primary GPU; all-GPU execution is faster and simpler"
    elif spark_available:
        spark_reason = "GPU memory is unknown; automatic mode will not assume offload"
    else:
        spark_reason = "this is not a Spark-compatible MoE architecture"
    spark = OptimizationDecision(
        "Spark",
        spark_enabled,
        spark_available,
        spark_reason,
        spark_qualification,
    )

    needs_optimizer_drafter = not optimizer_drafter_available and (
        pflash_profile
        or (
            kvflash_profile
            and kvflash_capability is not None
            and kvflash_capability.scorerless_policy is None
        )
    )
    placement = automatic_placement(
        cfg,
        runtime,
        preset,
        has_draft=dflash_enabled,
        optimizer_drafter_available=optimizer_drafter_available,
    )
    capacity_adjustments: set[CapacityAdjustment] = set()
    if not placement.runnable:
        for fallback_runtime, adjustment in _capacity_fallbacks(
            runtime,
            kvflash=kvflash_capability,
        ):
            capacity_adjustments.add(adjustment)
            placement = automatic_placement(
                cfg,
                fallback_runtime,
                preset,
                has_draft=dflash_enabled and fallback_runtime.speculative_decode,
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
            dflash.name,
            False,
            dflash.available,
            reason,
            dflash.qualification,
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
            pflash.qualification,
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
            kvflash.qualification,
        )
    elif "kvflash-scorerless" in capacity_adjustments:
        kvflash = OptimizationDecision(
            "KVFlash",
            True,
            kvflash.available,
            f"memory is tight; {adjusted.kvflash_policy} policy avoids a separate scorer allocation",
            kvflash.qualification,
        )
    if spark.enabled and not adjusted.spark:
        spark = OptimizationDecision(
            "Spark",
            False,
            spark.available,
            "disabled because Automatic does not compose Spark with target layer splitting",
            spark.qualification,
        )
    if (
        prefill_alternative is not None
        and prefill_alternative.available
        and alternative_capability is not None
    ):
        alternative_placement = automatic_placement(
            cfg,
            replace(adjusted, ds4_prefill=alternative_capability.mode),
            preset,
            has_draft=adjusted.speculative_decode,
            optimizer_drafter_available=optimizer_drafter_available,
        )
        if not alternative_placement.runnable:
            prefill_alternative = OptimizationDecision(
                prefill_alternative.name,
                False,
                False,
                alternative_placement.reason,
                prefill_alternative.qualification,
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
        prefill_alternative=prefill_alternative,
        prefill_strategy=(pflash.name if pflash.enabled else profile.prefill_baseline),
        decode_strategy=(dflash.name if dflash.enabled else profile.decode_baseline),
        kv_strategy=(
            f"{kvflash.name} ({adjusted.kvflash_policy})"
            if kvflash.enabled
            else profile.kv_baseline
        ),
        needs_optimizer_drafter=needs_optimizer_drafter,
    )
