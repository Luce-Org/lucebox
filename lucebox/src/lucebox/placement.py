"""Capability-checked accelerator placement for Lucebox inference.

Layer splitting is a capacity tool, not an automatic speed claim: the engine
runs contiguous layer groups sequentially and crosses a host/IPC boundary.
When a model fits the faster primary GPU, keeping the target monolithic is the
default. Secondary devices are used for an independently useful workload
(draft/scorer or Spark cold experts), or when splitting is required to make a
model runnable.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

from lucebox.topology import GpuDevice, HardwareTopology, from_config
from lucebox.types import Config, DflashRuntime, PlacementRuntime


@dataclass(frozen=True, slots=True)
class ArchitecturePlacementCapabilities:
    layer_split: bool
    remote_draft: bool
    draft_on_layer_split: bool
    pflash_on_layer_split: bool
    expert_offload: bool


# Mirrors server/src/common/model_capabilities.h for the product-level paths
# the packaged planner can actually launch. A parity test keeps this table
# explicit and reviewable when an engine architecture changes.
ARCHITECTURE_CAPABILITIES: dict[str, ArchitecturePlacementCapabilities] = {
    "qwen35": ArchitecturePlacementCapabilities(True, True, True, True, False),
    "qwen35moe": ArchitecturePlacementCapabilities(False, False, False, False, True),
    "laguna": ArchitecturePlacementCapabilities(True, False, False, False, True),
    "gemma4": ArchitecturePlacementCapabilities(True, False, False, False, False),
    "deepseek4": ArchitecturePlacementCapabilities(True, False, False, False, False),
    "qwen3": ArchitecturePlacementCapabilities(False, False, False, False, False),
}


@dataclass(frozen=True, slots=True)
class PlacementOption:
    key: str
    label: str
    available: bool
    recommended: bool
    reason: str


@dataclass(frozen=True, slots=True)
class PlacementPlan:
    runtime: PlacementRuntime
    optimization_runtime: DflashRuntime
    summary: str
    reason: str
    runnable: bool
    topology: HardwareTopology
    options: tuple[PlacementOption, ...]


def _artifact_size_gb(path: Path, fallback: float) -> float:
    try:
        size = path.stat().st_size / (1024**3)
    except OSError:
        return fallback
    # Unit tests and interrupted downloads create tiny placeholders. They do
    # not describe the memory footprint of a loaded model.
    return size if size >= 0.25 else fallback


def _model_sizes(cfg: Config, preset: object, has_draft: bool) -> tuple[float, float]:
    target_file = cfg.model.target_file or str(getattr(preset, "target_file", ""))
    target_fallback = float(getattr(preset, "approx_target_gb", 0.0))
    target_gb = _artifact_size_gb(cfg.models_dir / target_file, target_fallback)

    draft_gb = 0.0
    if has_draft:
        draft_file = cfg.model.draft_file or getattr(preset, "draft_file", None)
        draft_fallback = float(getattr(preset, "approx_draft_gb", 0.0))
        if draft_file:
            draft_gb = _artifact_size_gb(cfg.models_dir / "draft" / str(draft_file), draft_fallback)
        else:
            draft_gb = draft_fallback
    return target_gb, draft_gb


def _primary_capacity(device: GpuDevice) -> float:
    # Discrete GPUs need less host-side reserve; UMA devices share their pool
    # with the OS and CPU engine buffers and therefore keep a wider margin.
    reserve = 8.0 if device.unified_memory else 2.0
    return max(0.0, float(device.effective_memory_gb) - reserve)


def _secondary_score(device: GpuDevice, primary: GpuDevice) -> tuple[int, int, int, int]:
    return (
        device.backend == primary.backend,
        not device.unified_memory,
        device.physical_vram_gb,
        device.effective_memory_gb,
    )


def _best_secondary(topology: HardwareTopology) -> GpuDevice | None:
    if topology.primary is None or not topology.companions:
        return None
    primary = topology.primary
    return max(
        topology.companions,
        key=lambda device: _secondary_score(device, primary),
    )


def _cross_backend_ready(
    cfg: Config,
    primary: GpuDevice,
    secondary: GpuDevice,
) -> bool:
    """Whether the installed paired runtime supports this backend direction.

    The first packaged heterogeneous contract is deliberately directional:
    a CUDA server drives a HIP companion daemon on RTX + Strix systems. A
    future HIP-main/CUDA-daemon package can extend the host facts without the
    planner accidentally claiming that today's binaries support it.
    """
    return cfg.host.hybrid_runtime and primary.backend == "cuda" and secondary.backend == "hip"


def _split_devices(
    cfg: Config,
    topology: HardwareTopology,
    target_gb: float,
    primary_capacity: float,
) -> tuple[tuple[GpuDevice, ...], tuple[float, ...]]:
    """Choose the smallest ordered device set that makes the target fit."""
    primary = topology.primary
    if primary is None:
        return (), ()

    local = sorted(
        (device for device in topology.companions if device.backend == primary.backend),
        key=lambda device: _secondary_score(device, primary),
        reverse=True,
    )
    remote = sorted(
        (
            device
            for device in topology.companions
            if device.backend != primary.backend and _cross_backend_ready(cfg, primary, device)
        ),
        key=lambda device: _secondary_score(device, primary),
        reverse=True,
    )
    selected = [primary]
    capacities = [primary_capacity]
    total = primary_capacity
    for device in (*local, *remote):
        if total >= target_gb:
            break
        capacity = _primary_capacity(device)
        if capacity <= 0.0:
            continue
        selected.append(device)
        capacities.append(capacity)
        total += capacity
    if len(selected) < 2 or total < target_gb:
        return (), ()

    # Assign only the portion needed from the final device, then normalize.
    # A small floor prevents whole-layer rounding from creating an empty shard.
    remaining = target_gb
    allocations: list[float] = []
    for capacity in capacities:
        allocation = min(capacity, remaining)
        allocations.append(max(allocation, target_gb * 0.05))
        remaining -= allocation
    total_allocation = sum(allocations)
    weights = tuple(round(value / total_allocation, 4) for value in allocations)
    # Make the serialized values sum exactly to one after rounding.
    weights = (*weights[:-1], round(1.0 - sum(weights[:-1]), 4))
    return tuple(selected), weights


def _single_option(primary: GpuDevice, fits: bool) -> PlacementOption:
    return PlacementOption(
        key="single",
        label="Primary GPU",
        available=fits,
        recommended=fits,
        reason=(
            f"the selected stack fits {primary.name}; avoiding a layer/IPC boundary is faster"
            if fits
            else f"the target does not fit the safe memory budget on {primary.name}"
        ),
    )


def automatic_placement(
    cfg: Config,
    runtime: DflashRuntime,
    preset: object,
    *,
    has_draft: bool,
    optimizer_drafter_available: bool,
) -> PlacementPlan:
    """Resolve a safe placement and every user-visible alternative."""
    topology = from_config(cfg)
    primary = topology.primary
    if primary is None:
        return PlacementPlan(
            runtime=PlacementRuntime(),
            optimization_runtime=runtime,
            summary="No accelerator selected",
            reason="choose a CUDA or ROCm backend after a supported GPU is detected",
            runnable=False,
            topology=topology,
            options=(),
        )

    architecture = str(getattr(preset, "architecture", ""))
    capabilities = ARCHITECTURE_CAPABILITIES.get(
        architecture,
        ArchitecturePlacementCapabilities(False, False, False, False, False),
    )
    secondary = _best_secondary(topology)
    target_gb, draft_gb = _model_sizes(cfg, preset, has_draft)
    scorer_gb = (
        1.2
        if optimizer_drafter_available
        and (
            runtime.prefill_mode != "off"
            or (runtime.kvflash != "off" and runtime.kvflash_policy == "drafter")
        )
        else 0.0
    )
    primary_capacity = _primary_capacity(primary)
    target_fits = target_gb <= primary_capacity
    stack_gb = target_gb + draft_gb + scorer_gb
    # Spark's explicit purpose is to make an MoE target fit by leaving cold
    # experts in host memory. Its load-time allocator remains the final source
    # of truth for the exact byte budget.
    stack_fits = stack_gb <= primary_capacity or (runtime.spark and capabilities.expert_offload)

    options: list[PlacementOption] = [_single_option(primary, stack_fits)]
    base = PlacementRuntime(mode="single", target_device=primary.placement_name)
    if secondary is None:
        if stack_fits:
            return PlacementPlan(
                runtime=base,
                optimization_runtime=runtime,
                summary=primary.name,
                reason="one accelerator is available and the selected stack fits its safe budget",
                runnable=True,
                topology=topology,
                options=tuple(options),
            )
        return PlacementPlan(
            runtime=base,
            optimization_runtime=runtime,
            summary=primary.name,
            reason="the selected target exceeds this GPU's safe memory budget",
            runnable=False,
            topology=topology,
            options=tuple(options),
        )

    same_backend = primary.backend == secondary.backend
    cross_backend_ready = _cross_backend_ready(cfg, primary, secondary)
    draft_work_available = (has_draft and runtime.speculative_decode) or (
        optimizer_drafter_available and runtime.prefill_mode != "off"
    )
    draft_offload_available = draft_work_available and (
        same_backend or (cross_backend_ready and capabilities.remote_draft)
    )
    options.append(
        PlacementOption(
            key="draft-offload",
            label="Secondary draft/scorer",
            available=draft_offload_available,
            recommended=draft_offload_available and not stack_fits and target_fits,
            reason=(
                "moves the independent draft/scorer workload off the primary GPU"
                if draft_offload_available
                else (
                    "the selected model has no installed draft/scorer workload"
                    if not draft_work_available
                    else "the engine/runtime cannot execute this draft across the backend boundary"
                )
            ),
        )
    )

    split_primary_capacity = primary_capacity
    if capabilities.draft_on_layer_split and runtime.speculative_decode:
        split_primary_capacity -= draft_gb
    if capabilities.pflash_on_layer_split:
        split_primary_capacity -= scorer_gb
    split_devices, split_weights = _split_devices(
        cfg,
        topology,
        target_gb,
        max(0.0, split_primary_capacity),
    )
    split_available = capabilities.layer_split and bool(split_devices)
    options.append(
        PlacementOption(
            key="layer-split",
            label="Target layer split",
            available=split_available,
            recommended=split_available and not target_fits,
            reason=(
                "capacity mode for a target that does not fit the primary GPU"
                if split_available
                else (
                    f"{architecture or 'this model'} has no engine layer-split path"
                    if not capabilities.layer_split
                    else "a compatible runtime or enough combined safe memory is unavailable"
                )
            ),
        )
    )

    remote_experts_available = (
        runtime.spark and capabilities.expert_offload and (same_backend or cross_backend_ready)
    )
    options.append(
        PlacementOption(
            key="remote-experts",
            label="Secondary Spark experts",
            available=remote_experts_available,
            recommended=remote_experts_available,
            reason=(
                "runs Spark cold experts on the secondary accelerator instead of CPU"
                if remote_experts_available
                else "available only when Spark is active and a compatible IPC runtime is installed"
            ),
        )
    )

    if remote_experts_available:
        placement = PlacementRuntime(
            mode="heterogeneous",
            target_device=primary.placement_name,
            remote_expert_device=secondary.placement_name,
        )
        return PlacementPlan(
            runtime=placement,
            optimization_runtime=runtime,
            summary=f"{primary.name} target + {secondary.name} Spark experts",
            reason="MoE memory pressure activated Spark; the secondary GPU handles cold experts",
            runnable=True,
            topology=topology,
            options=tuple(options),
        )

    if stack_fits:
        return PlacementPlan(
            runtime=base,
            optimization_runtime=runtime,
            summary=primary.name,
            reason=(
                f"the full stack fits the faster primary; {secondary.name} remains available "
                "because a sequential target split would reduce throughput"
            ),
            runnable=True,
            topology=topology,
            options=tuple(options),
        )

    if target_fits and draft_offload_available:
        mixed = not same_backend
        placement = PlacementRuntime(
            mode="heterogeneous" if mixed else "draft-offload",
            target_device=primary.placement_name,
            draft_device=secondary.placement_name,
            remote_draft=mixed,
        )
        return PlacementPlan(
            runtime=placement,
            optimization_runtime=runtime,
            summary=f"{primary.name} target + {secondary.name} draft/scorer",
            reason="the target fits the primary, but its optional draft/scorer needs separate memory",
            runnable=True,
            topology=topology,
            options=tuple(options),
        )

    if not target_fits and split_available:
        mixed = len({device.backend for device in split_devices}) > 1
        adjusted = runtime
        if not capabilities.draft_on_layer_split:
            adjusted = replace(adjusted, speculative_decode=False)
        if not capabilities.pflash_on_layer_split:
            adjusted = replace(
                adjusted,
                prefill_mode="off",
                prefill_drafter="",
            )
        # Spark and target layer splitting are distinct placement systems. No
        # architecture currently validates composing both in Automatic mode.
        adjusted = replace(adjusted, spark=False, spark_vram_gb=0.0)
        placement = PlacementRuntime(
            mode="heterogeneous" if mixed else "layer-split",
            target_devices=tuple(device.placement_name for device in split_devices),
            target_layer_split=split_weights,
            remote_target_shard=mixed,
            # P2P is topology-specific and cannot be inferred from a shared
            # backend name. The safe host-staged boundary is the automatic
            # default; an explicit expert benchmark may still opt into P2P.
            peer_access=False,
        )
        device_names = " + ".join(device.name for device in split_devices)
        return PlacementPlan(
            runtime=placement,
            optimization_runtime=adjusted,
            summary=f"{device_names} target split",
            reason="capacity mode: the target does not fit the primary GPU alone",
            runnable=True,
            topology=topology,
            options=tuple(options),
        )

    return PlacementPlan(
        runtime=base,
        optimization_runtime=runtime,
        summary=primary.name,
        reason=(
            "the target does not fit the primary and no validated secondary-device "
            "placement is available"
        ),
        runnable=False,
        topology=topology,
        options=tuple(options),
    )
