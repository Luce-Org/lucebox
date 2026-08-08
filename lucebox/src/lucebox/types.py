"""Shared dataclasses passed between modules.

HostFacts is populated from the LUCEBOX_HOST_* env vars set by lucebox.sh.
Config is what we serialize to/from .lucebox/config.toml. Both are frozen so
mistakes (e.g. mutating a config after autotune wrote it) fail loudly.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

Variant = str
CtkStatus = Literal["runtime", "cdi", "installed-unwired", "none"]


def default_models_dir() -> Path:
    """Resolve the default models directory under the XDG Base Directory spec.

    $XDG_DATA_HOME (default ~/.local/share) is the conventional location for
    user-specific data files on Linux + macOS. Lucebox nests its model store
    under that so downloads live alongside other per-user app data instead
    of cluttering $HOME directly. The host wrapper bind-mounts this path
    into the container so paths line up in and out of the image.
    """
    base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / "lucebox" / "models"


GpuVendor = Literal["nvidia", "amd", "none"]
PlacementMode = Literal[
    "single",
    "draft-offload",
    "layer-split",
    "heterogeneous",
]


@dataclass(frozen=True, slots=True)
class HostFacts:
    """Probed once by lucebox.sh, passed in via env vars. Single source of
    truth on the Python side — we never reprobe (we can't see host /proc)."""

    nproc: int = 0
    ram_gb: int = 0
    gpu_vendor: GpuVendor = "none"
    has_nvidia_gpu: bool = False
    has_amd_gpu: bool = False
    gpu_name: str = ""
    gpu_count: int = 0
    vram_gb: int = 0
    # NVIDIA compute capability without the dot ("120") or AMD gfx target
    # ("gfx1151").  The historical ``gpu_sm`` name stays in the serialized
    # contract for compatibility with HOST_INFO consumers.
    gpu_sm: str = ""
    driver_version: str = ""  # NVIDIA driver, e.g. "595.71.05"
    driver_major: int = 0
    rocm_version: str = ""  # AMD ROCm userspace, e.g. "7.2.4"
    has_kfd: bool = False  # /dev/kfd exists and is accessible to the user
    has_dri: bool = False  # at least one accessible /dev/dri/renderD* node
    has_systemd: bool = False
    is_wsl: bool = False
    has_docker: bool = False
    docker_version: str = ""
    ctk: CtkStatus = "none"
    # Per-vendor inventory is kept in addition to the selected accelerator.
    # On an RTX + Strix host the generic ``gpu_*`` fields describe the RTX,
    # while these fields retain the AMD companion for placement planning. On
    # an R9700 + Strix host the AMD CSV contains both devices.
    nvidia_gpu_name: str = ""
    nvidia_gpu_count: int = 0
    nvidia_vram_gb: int = 0
    nvidia_gpu_arch: str = ""
    nvidia_gpu_list_csv: str = ""
    nvidia_unified_memory: bool = False
    amd_gpu_name: str = ""
    amd_gpu_count: int = 0
    amd_vram_gb: int = 0
    amd_gpu_arch: str = ""
    amd_gpu_list_csv: str = ""
    # The first packaged cross-vendor runtime is a CUDA server paired with a
    # HIP companion daemon. The host wrapper proves that contract before
    # setting this fact; the planner never assumes a CUDA image can execute a
    # HIP child process.
    hybrid_runtime: bool = False

    def __post_init__(self) -> None:
        """Reject malformed persisted host snapshots.

        ``Literal`` annotations help static type-checkers, but TOML is runtime
        input and can still contain arbitrary strings.  Failing here keeps an
        invalid snapshot from silently selecting the wrong accelerator path.
        """
        if self.gpu_vendor not in {"nvidia", "amd", "none"}:
            raise ValueError(f"gpu_vendor must be nvidia, amd, or none; got {self.gpu_vendor!r}")
        if self.ctk not in {"runtime", "cdi", "installed-unwired", "none"}:
            raise ValueError(
                f"ctk must be runtime, cdi, installed-unwired, or none; got {self.ctk!r}"
            )


_PLACEMENT_DEVICE_RE = re.compile(r"^(cuda|hip):([0-9]+)$")


def _placement_backend(device: str) -> str:
    match = _PLACEMENT_DEVICE_RE.fullmatch(device)
    if not match:
        raise ValueError(f"placement device must use cuda:N or hip:N syntax; got {device!r}")
    return match.group(1)


@dataclass(frozen=True, slots=True)
class PlacementRuntime:
    """Validated accelerator placement passed atomically to the engine.

    Paths to backend IPC binaries deliberately do not live in user config.
    They are installation details resolved by the host wrapper after it has
    verified the executable and its backend runtime. These fields describe
    only the portable execution intent.
    """

    mode: PlacementMode = "single"
    target_device: str = ""
    target_devices: tuple[str, ...] = ()
    target_layer_split: tuple[float, ...] = ()
    draft_device: str = ""
    remote_draft: bool = False
    remote_target_shard: bool = False
    peer_access: bool = False
    remote_expert_device: str = ""

    def __post_init__(self) -> None:
        if self.mode not in {
            "single",
            "draft-offload",
            "layer-split",
            "heterogeneous",
        }:
            raise ValueError(f"unknown placement mode {self.mode!r}")
        if self.target_device and self.target_devices:
            raise ValueError("target_device and target_devices are mutually exclusive")

        devices = self.target_devices or ((self.target_device,) if self.target_device else ())
        if not devices and self.mode != "single":
            raise ValueError("non-single placement requires a target device")
        if len(set(devices)) != len(devices):
            raise ValueError("target devices must be unique")
        backends = tuple(_placement_backend(device) for device in devices)
        if self.draft_device:
            draft_backend = _placement_backend(self.draft_device)
        else:
            draft_backend = backends[0] if backends else ""
        if self.remote_expert_device:
            _placement_backend(self.remote_expert_device)

        if self.target_layer_split:
            if len(self.target_layer_split) != len(self.target_devices):
                raise ValueError("target_layer_split must contain one weight per target device")
            if len(self.target_layer_split) < 2:
                raise ValueError("target layer split requires at least two devices")
            if any(
                not math.isfinite(weight) or weight <= 0.0 for weight in self.target_layer_split
            ):
                raise ValueError("target layer split weights must be finite and positive")
        elif len(self.target_devices) > 1:
            raise ValueError("multiple target devices require target_layer_split")

        mixed_target = len(set(backends)) > 1
        if mixed_target and not self.remote_target_shard:
            raise ValueError("mixed-backend target placement requires remote_target_shard")
        if self.remote_target_shard and not mixed_target:
            raise ValueError("remote_target_shard requires mixed-backend target placement")
        mixed_draft = bool(draft_backend and backends and draft_backend != backends[0])
        if mixed_draft and not self.remote_draft:
            raise ValueError("mixed-backend draft placement requires remote_draft")
        if self.remote_draft and not mixed_draft:
            raise ValueError("remote_draft requires mixed-backend draft placement")
        if self.draft_device and self.draft_device in devices:
            raise ValueError("draft_device must differ from the target device")
        if self.remote_expert_device and self.remote_expert_device in devices:
            raise ValueError("remote_expert_device must differ from target devices")
        if self.peer_access and (len(backends) < 2 or len(set(backends)) != 1):
            raise ValueError("peer_access requires a same-backend target layer split")
        if self.mode == "single" and (
            len(devices) > 1
            or self.draft_device
            or self.remote_expert_device
            or self.remote_draft
            or self.remote_target_shard
        ):
            raise ValueError("single placement cannot contain secondary-device settings")
        if self.mode == "draft-offload" and not self.draft_device:
            raise ValueError("draft-offload placement requires draft_device")
        if self.mode == "layer-split" and not self.target_layer_split:
            raise ValueError("layer-split placement requires target_layer_split")
        if self.mode == "heterogeneous" and not (
            self.remote_draft or self.remote_target_shard or self.remote_expert_device
        ):
            raise ValueError("heterogeneous placement requires a cross-device workload")

    @property
    def uses_multiple_devices(self) -> bool:
        return bool(len(self.target_devices) > 1 or self.draft_device or self.remote_expert_device)

    @property
    def target_backend(self) -> str:
        devices = self.target_devices or ((self.target_device,) if self.target_device else ())
        return _placement_backend(devices[0]) if devices else ""

    @property
    def requires_hybrid_runtime(self) -> bool:
        """Whether one process must launch a daemon from another backend."""
        if self.remote_draft or self.remote_target_shard:
            return True
        if not self.remote_expert_device or not self.target_backend:
            return False
        return _placement_backend(self.remote_expert_device) != self.target_backend


@dataclass(frozen=True, slots=True)
class DflashRuntime:
    """Typed inference and optimization settings persisted under ``[dflash]``.

    The name is historical: the same native server owns DFlash speculative
    decode, PFlash prefill compression, KVFlash bounded residency, and Spark
    MoE offload. Keeping their launch settings together gives the optimizer a
    single atomic profile to write and the container one explicit contract to
    consume.
    """

    speculative_decode: bool = True
    budget: int = 22
    max_ctx: int = 16384
    lazy: bool = False
    # Exact turn-boundary snapshots. Eight slots bound memory at long context
    # while covering the common harness/session set; 0 remains an explicit
    # opt-out and Advanced mode can raise the cap deliberately.
    prefix_cache_slots: int = 8
    # Exact full-prompt snapshots, keyed by the raw prompt. PFlash can use
    # these to skip both rescoring and target prefill when a request repeats.
    prefill_cache_slots: int = 0
    cache_type_k: str = ""
    cache_type_v: str = ""
    prefill_mode: Literal["off", "auto", "always"] = "off"
    prefill_keep_ratio: float = 0.05
    prefill_threshold: int = 32000
    prefill_drafter: str = ""
    kvflash: str = "off"
    kvflash_policy: Literal["drafter", "lru", "qk"] = "drafter"
    kvflash_tau: int = 64
    spark: bool = False
    spark_vram_gb: float = 0.0
    # DeepSeek4's architecture-specific prefill implementation. ``exact`` is
    # the quality-safe default. ``dense`` and ``sparse`` are approximate,
    # monolithic-HIP preview paths and are never selected silently.
    ds4_prefill: Literal["exact", "dense", "sparse"] = "exact"
    # Optional operator override for the phase-1 reasoning cap. ``None`` lets
    # the selected model card choose its own safe value; one global DeepSeek
    # default would silently truncate Qwen/Gemma reasoning workloads.
    think_max: int | None = None
    # Flash-attention sliding-window on full-attention layers. 0 = full
    # attention (server default). On gemma4's hybrid iSWA the full-attn
    # layers grow KV linearly with max_ctx; a sparse fa_window keeps
    # decode compute bounded on long prompts without changing the KV
    # footprint. Passed through to the server's `--fa-window <N>`
    # flag (see server/src/server/server_main.cpp).
    fa_window: int = 0
    # Soft-close thinking termination dial (PR #326 in lucebox-hub).
    # Lets the AR loop force </think> early when the close-token logit
    # comes within this probability ratio of the chosen-token logit.
    # Range [0.0, 1.0]; 0.0 = disabled (byte-identical to pre-change
    # behaviour). 0.5 = close when close-token prob >= 0.5 * chosen-token
    # prob; 0.9 = aggressive. Qwen3.5/3.6 AR path only in v1. Surfaced
    # to the server via DFLASH_THINK_SOFT_CLOSE_MIN_RATIO →
    # --think-soft-close-min-ratio.
    think_soft_close_min_ratio: float = 0.0
    # Diagnostic: when True, surface --debug-thinking-logits to the
    # server CLI via DFLASH_DEBUG_THINKING_LOGITS=1, producing one
    # stderr line per thinking AR step recording the close-vs-chosen
    # logit gap. Used to fit a sliding-ratio curve from real trajectory
    # data. Heavy stderr (one line per thinking token across all
    # in-flight requests); leave off in production.
    debug_thinking_logits: bool = False

    def __post_init__(self) -> None:
        """Validate the bounded tuning knobs before they reach the server."""
        if self.budget < 0:
            raise ValueError(f"budget must be zero or positive; got {self.budget!r}")
        if self.max_ctx <= 0:
            raise ValueError(f"max_ctx must be positive; got {self.max_ctx!r}")
        if self.prefix_cache_slots < 0:
            raise ValueError(
                f"prefix_cache_slots must be zero or positive; got {self.prefix_cache_slots!r}"
            )
        if self.prefill_cache_slots < 0:
            raise ValueError(
                f"prefill_cache_slots must be zero or positive; got {self.prefill_cache_slots!r}"
            )
        if self.prefill_cache_slots > 0 and (
            self.prefix_cache_slots + self.prefill_cache_slots > 63
        ):
            raise ValueError(
                "prefix_cache_slots + prefill_cache_slots must not exceed 63 "
                "when the exact prefill cache is enabled"
            )
        if not 0.0 < self.prefill_keep_ratio <= 1.0:
            raise ValueError(
                "prefill_keep_ratio must be in the interval (0.0, 1.0]; "
                f"got {self.prefill_keep_ratio!r}"
            )
        if not 0.0 <= self.think_soft_close_min_ratio <= 1.0:
            raise ValueError(
                "think_soft_close_min_ratio must be in the interval [0.0, 1.0]; "
                f"got {self.think_soft_close_min_ratio!r}"
            )
        if self.kvflash not in {"off", "auto"}:
            try:
                pool_tokens = int(self.kvflash)
            except ValueError as exc:
                raise ValueError(
                    f"kvflash must be off, auto, or a positive token count; got {self.kvflash!r}"
                ) from exc
            if pool_tokens <= 0:
                raise ValueError(f"kvflash token count must be positive; got {self.kvflash!r}")
        if self.kvflash_policy not in {"drafter", "lru", "qk"}:
            raise ValueError(
                f"kvflash_policy must be drafter, lru, or qk; got {self.kvflash_policy!r}"
            )
        if self.kvflash_tau <= 0:
            raise ValueError(f"kvflash_tau must be positive; got {self.kvflash_tau!r}")
        if not math.isfinite(self.spark_vram_gb) or self.spark_vram_gb < 0.0:
            raise ValueError(
                "spark_vram_gb must be finite and zero (automatic) or positive; "
                f"got {self.spark_vram_gb!r}"
            )
        if self.ds4_prefill not in {"exact", "dense", "sparse"}:
            raise ValueError(
                "ds4_prefill must be exact, dense, or sparse; "
                f"got {self.ds4_prefill!r}"
            )
        if self.prefill_threshold <= 0:
            raise ValueError(f"prefill_threshold must be positive; got {self.prefill_threshold!r}")
        if self.think_max is not None and self.think_max < 0:
            raise ValueError(f"think_max must be zero or positive; got {self.think_max!r}")
        if self.fa_window < 0:
            raise ValueError(f"fa_window must be zero or positive; got {self.fa_window!r}")
        if self.kvflash != "off" and self.fa_window > 0:
            raise ValueError("kvflash and fa_window are mutually exclusive")


@dataclass(frozen=True, slots=True)
class ModelMeta:
    """Which preset the operator picked at configure/download time.

    Persisted under ``[model]`` in config.toml so `lucebox serve` can
    pass ``DFLASH_TARGET=/opt/lucebox-hub/server/models/<file>`` and
    ``DFLASH_DRAFT`` for the draft GGUF (when one is published for the
    preset). The entrypoint's "multiple candidate GGUFs" branch never
    has to guess which one to load.

    ``target_file`` and ``draft_file`` are advanced overrides — when set
    they win over the preset's registry default. Empty strings mean
    "fall back to the registry value for [model] preset, then to the
    entrypoint's autodetect".
    """

    preset: str = ""
    target_file: str = ""
    draft_file: str = ""


@dataclass(frozen=True, slots=True)
class Config:
    """The whole config.toml, materialized."""

    variant: Variant = "cuda12"
    image: str = "ghcr.io/luce-org/lucebox-hub"
    container_name: str = "lucebox"
    port: int = 8080
    models_dir: Path = field(default_factory=default_models_dir)
    dflash: DflashRuntime = field(default_factory=DflashRuntime)
    placement: PlacementRuntime = field(default_factory=PlacementRuntime)
    host: HostFacts = field(default_factory=HostFacts)
    model: ModelMeta = field(default_factory=ModelMeta)
