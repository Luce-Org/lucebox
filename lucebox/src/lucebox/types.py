"""Shared dataclasses passed between modules.

HostFacts is populated from the LUCEBOX_HOST_* env vars set by lucebox.sh.
Config is what we serialize to/from .lucebox/config.toml. Both are frozen so
mistakes (e.g. mutating a config after autotune wrote it) fail loudly.
"""

from __future__ import annotations

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
    gpu_name: str = ""
    gpu_count: int = 0
    vram_gb: int = 0
    gpu_sm: str = ""  # e.g. "120" — matches docker-bake arch lists
    driver_version: str = ""  # e.g. "595.71.05"
    driver_major: int = 0
    has_systemd: bool = False
    is_wsl: bool = False
    has_docker: bool = False
    docker_version: str = ""
    ctk: CtkStatus = "none"


@dataclass(frozen=True, slots=True)
class DflashRuntime:
    """The DFLASH_* knobs as typed values. Serialized under [dflash] in TOML
    and emitted as -e DFLASH_FOO=bar args to docker run.

    The 11 fields below (budget through prefill_drafter) form the strict
    allowlist mirrored by lucebench's snapshot config.json — keep both
    in lockstep. ``think_max`` is a separate phase-1 thinking cap that
    isn't part of the runtime snapshot allowlist (it's per-request, not
    per-server).
    """

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
    # Phase-1 (thinking) cap when a request opts into thinking. Default mirrors
    # antirez/ds4 ds4_eval.c: think_max_tokens = max_tokens - hard_limit_reply
    # budget = 16000 - 512 = 15488. The server's own hardcoded default is 10000.
    think_max: int = 15488


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
