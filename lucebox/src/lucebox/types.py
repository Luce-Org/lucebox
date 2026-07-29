"""Shared dataclasses passed between modules.

HostFacts is populated from the LUCEBOX_HOST_* env vars set by lucebox.sh.
Config is what we serialize to/from .lucebox/config.toml. Both are frozen so
mistakes (e.g. mutating a config after autotune wrote it) fail loudly.
"""

from __future__ import annotations

import math
import os
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
    prefix_cache_slots: int = 0
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
    # Phase-1 (thinking) cap when a request opts into thinking. Default mirrors
    # antirez/ds4 ds4_eval.c: think_max_tokens = max_tokens - hard_limit_reply
    # budget = 16000 - 512 = 15488. The server's own hardcoded default is 10000.
    think_max: int = 15488
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
        if self.prefix_cache_slots < 0 or self.prefill_cache_slots < 0:
            raise ValueError("cache slot counts must be zero or positive")
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
        if self.prefill_threshold <= 0:
            raise ValueError(f"prefill_threshold must be positive; got {self.prefill_threshold!r}")
        if self.think_max < 0:
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
    host: HostFacts = field(default_factory=HostFacts)
    model: ModelMeta = field(default_factory=ModelMeta)
