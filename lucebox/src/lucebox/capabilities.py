"""Product capability contracts for supported models and engine architectures.

This module deliberately contains *static* facts only: which optimization an
engine/model pair can run, which backends have been qualified, and which
policies are legal. Hardware inventory, memory pressure, and device placement
remain runtime decisions owned by :mod:`lucebox.autotune` and
:mod:`lucebox.placement`.

Keeping those concerns separate prevents a common class of planner bugs: a
model name must never imply a particular GPU layout, and detecting a large GPU
must never make an unsupported model feature legal.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Literal

Backend = Literal["cuda", "hip"]
KvFlashPolicy = Literal["drafter", "lru", "qk"]


class Qualification(StrEnum):
    """Product support level for one model/backend implementation."""

    UNAVAILABLE = "unavailable"
    PREVIEW = "preview"
    QUALIFIED = "qualified"


@dataclass(frozen=True, slots=True)
class BackendSupport:
    """Qualification of the same implementation on CUDA and HIP."""

    cuda: Qualification
    hip: Qualification

    def for_backend(self, backend: str) -> Qualification:
        if backend == "cuda":
            return self.cuda
        if backend == "hip":
            return self.hip
        return Qualification.UNAVAILABLE

    def available_on(self, backend: str) -> bool:
        return self.for_backend(backend) is not Qualification.UNAVAILABLE

    def qualified_on(self, backend: str) -> bool:
        return self.for_backend(backend) is Qualification.QUALIFIED


QUALIFIED_BOTH = BackendSupport(Qualification.QUALIFIED, Qualification.QUALIFIED)
PREVIEW_BOTH = BackendSupport(Qualification.PREVIEW, Qualification.PREVIEW)
HIP_PREVIEW = BackendSupport(Qualification.UNAVAILABLE, Qualification.PREVIEW)


@dataclass(frozen=True, slots=True)
class FeatureCapability:
    """A launchable optimization and its automatic-mode policy."""

    label: str
    support: BackendSupport
    automatic: bool = True

    def __post_init__(self) -> None:
        if not self.label.strip():
            raise ValueError("feature label must not be empty")

    def available_on(self, backend: str) -> bool:
        return self.support.available_on(backend)

    def automatic_on(self, backend: str) -> bool:
        return self.automatic and self.support.qualified_on(backend)

    def qualification_on(self, backend: str) -> Qualification:
        return self.support.for_backend(backend)


@dataclass(frozen=True, slots=True)
class PFlashCapability:
    feature: FeatureCapability
    minimum_context: int = 32_768
    keep_ratio: float = 0.10

    def __post_init__(self) -> None:
        if self.minimum_context <= 0:
            raise ValueError("PFlash minimum context must be positive")
        if not 0.0 < self.keep_ratio <= 1.0:
            raise ValueError("PFlash keep ratio must be in (0.0, 1.0]")


@dataclass(frozen=True, slots=True)
class KvFlashCapability:
    feature: FeatureCapability
    policies: tuple[KvFlashPolicy, ...]
    preferred_policy: KvFlashPolicy
    scorerless_policy: KvFlashPolicy | None
    minimum_context: int = 32_768
    pressure_headroom_gb: float = 5.0

    def __post_init__(self) -> None:
        if not self.policies:
            raise ValueError("KVFlash must expose at least one legal policy")
        if self.preferred_policy not in self.policies:
            raise ValueError("preferred KVFlash policy must be legal for the model")
        if self.scorerless_policy is not None and self.scorerless_policy not in self.policies:
            raise ValueError("scorerless KVFlash policy must be legal for the model")
        if self.minimum_context <= 0:
            raise ValueError("KVFlash minimum context must be positive")
        if self.pressure_headroom_gb < 0.0:
            raise ValueError("KVFlash pressure headroom must not be negative")


@dataclass(frozen=True, slots=True)
class SparkCapability:
    feature: FeatureCapability
    pressure_headroom_gb: float = 6.0
    minimum_host_ram_gb: int = 32

    def __post_init__(self) -> None:
        if self.pressure_headroom_gb < 0.0:
            raise ValueError("Spark pressure headroom must not be negative")
        if self.minimum_host_ram_gb <= 0:
            raise ValueError("Spark minimum host RAM must be positive")


@dataclass(frozen=True, slots=True)
class DeepSeekPrefillCapability:
    """Approximate DeepSeek4 prefill mode exposed by the native server."""

    feature: FeatureCapability
    mode: Literal["dense", "sparse"]


@dataclass(frozen=True, slots=True)
class ModelOptimizationProfile:
    """The optimization contract for one downloadable model preset.

    The three baseline labels make phase coverage explicit even when a model
    does not use a branded Lucebox optimization. For example, DeepSeek uses
    native MLA-compressed KV state rather than generic KVFlash.
    """

    preset: str
    architecture: str
    prefill_baseline: str
    decode_baseline: str
    kv_baseline: str
    speculative_decode: FeatureCapability | None = None
    pflash: PFlashCapability | None = None
    kvflash: KvFlashCapability | None = None
    spark: SparkCapability | None = None
    deepseek_prefill: DeepSeekPrefillCapability | None = None

    def __post_init__(self) -> None:
        if not self.preset.strip() or not self.architecture.strip():
            raise ValueError("model profile preset and architecture must not be empty")
        if not all(label.strip() for label in self.phase_baselines):
            raise ValueError("every model profile must name all three baseline strategies")

    @property
    def phase_baselines(self) -> tuple[str, str, str]:
        return (self.prefill_baseline, self.decode_baseline, self.kv_baseline)


@dataclass(frozen=True, slots=True)
class ArchitecturePlacementCapabilities:
    """Engine placement operations that are legal for one architecture."""

    layer_split: bool
    remote_draft: bool
    draft_on_layer_split: bool
    pflash_on_layer_split: bool
    expert_offload: bool


# Mirrors server/src/common/model_capabilities.h. Placement imports this table
# rather than maintaining a second architecture registry.
ARCHITECTURE_CAPABILITIES: Mapping[str, ArchitecturePlacementCapabilities] = (
    MappingProxyType(
        {
            "qwen35": ArchitecturePlacementCapabilities(True, True, True, True, False),
            "qwen35moe": ArchitecturePlacementCapabilities(False, False, False, False, True),
            "laguna": ArchitecturePlacementCapabilities(True, False, False, False, True),
            "gemma4": ArchitecturePlacementCapabilities(True, False, False, False, False),
            "deepseek4": ArchitecturePlacementCapabilities(True, False, False, False, False),
            "qwen3": ArchitecturePlacementCapabilities(False, False, False, False, False),
        }
    )
)

_DFLASH = FeatureCapability("DFlash", QUALIFIED_BOTH)
_DSPARK = FeatureCapability("DSpark", QUALIFIED_BOTH)
_PFLASH = FeatureCapability("PFlash", QUALIFIED_BOTH)
_KVFLASH = FeatureCapability("KVFlash", QUALIFIED_BOTH)
_KVFLASH_PREVIEW = FeatureCapability("KVFlash", PREVIEW_BOTH, automatic=False)
_SPARK = FeatureCapability("Spark", QUALIFIED_BOTH)


MODEL_OPTIMIZATION_PROFILES: Mapping[str, ModelOptimizationProfile] = MappingProxyType(
    {
        "qwen3.6-27b": ModelOptimizationProfile(
            preset="qwen3.6-27b",
            architecture="qwen35",
            prefill_baseline="Exact prefill",
            decode_baseline="Autoregressive decode",
            kv_baseline="Full KV cache",
            speculative_decode=_DFLASH,
            pflash=PFlashCapability(_PFLASH),
            kvflash=KvFlashCapability(
                _KVFLASH,
                policies=("drafter", "qk", "lru"),
                preferred_policy="drafter",
                scorerless_policy="qk",
            ),
        ),
        "gemma-4-26b": ModelOptimizationProfile(
            preset="gemma-4-26b",
            architecture="gemma4",
            prefill_baseline="Exact prefill",
            decode_baseline="Autoregressive decode",
            kv_baseline="Full KV cache",
            speculative_decode=_DFLASH,
            kvflash=KvFlashCapability(
                _KVFLASH_PREVIEW,
                policies=("drafter", "lru"),
                preferred_policy="drafter",
                scorerless_policy="lru",
            ),
        ),
        "gemma-4-31b": ModelOptimizationProfile(
            preset="gemma-4-31b",
            architecture="gemma4",
            prefill_baseline="Exact prefill",
            decode_baseline="Autoregressive decode",
            kv_baseline="Full KV cache",
            speculative_decode=_DFLASH,
            kvflash=KvFlashCapability(
                _KVFLASH_PREVIEW,
                policies=("drafter", "lru"),
                preferred_policy="drafter",
                scorerless_policy="lru",
            ),
        ),
        "laguna-xs.2": ModelOptimizationProfile(
            preset="laguna-xs.2",
            architecture="laguna",
            prefill_baseline="Exact sparse-attention prefill",
            decode_baseline="Autoregressive decode",
            kv_baseline="Full hybrid-attention KV cache",
            speculative_decode=_DFLASH,
            # The cross-tokenizer path launches, but remains opt-in until its
            # retrieval/quality qualification is complete on both backends.
            kvflash=KvFlashCapability(
                _KVFLASH_PREVIEW,
                policies=("drafter", "lru"),
                preferred_policy="drafter",
                scorerless_policy="lru",
            ),
            spark=SparkCapability(_SPARK),
        ),
        "qwen3.6-moe": ModelOptimizationProfile(
            preset="qwen3.6-moe",
            architecture="qwen35moe",
            prefill_baseline="Exact prefill",
            decode_baseline="Autoregressive decode",
            kv_baseline="Full KV cache",
            kvflash=KvFlashCapability(
                _KVFLASH,
                policies=("drafter", "lru"),
                preferred_policy="drafter",
                scorerless_policy=None,
            ),
            spark=SparkCapability(_SPARK),
        ),
        "deepseek-v4-flash": ModelOptimizationProfile(
            preset="deepseek-v4-flash",
            architecture="deepseek4",
            prefill_baseline="Exact MLA prefill",
            decode_baseline="Autoregressive decode",
            kv_baseline="Native MLA-compressed KV cache",
            speculative_decode=_DSPARK,
            # Sparse DeepSeek prefill is engine-backed but approximate and
            # currently restricted to the monolithic HIP path. It is recorded
            # here so the contract is complete, but Automatic does not enable
            # it until product qualification is promoted.
            deepseek_prefill=DeepSeekPrefillCapability(
                feature=FeatureCapability(
                    "DeepSeek sparse prefill",
                    HIP_PREVIEW,
                    automatic=False,
                ),
                mode="sparse",
            ),
        ),
    }
)


def model_profile(preset: str) -> ModelOptimizationProfile:
    """Return a preset contract, failing loudly if catalog metadata drifted."""
    try:
        return MODEL_OPTIMIZATION_PROFILES[preset]
    except KeyError as exc:
        raise KeyError(f"no optimization profile is registered for preset {preset!r}") from exc


def architecture_capabilities(architecture: str) -> ArchitecturePlacementCapabilities:
    """Return legal placement operations; unknown architectures are inert."""
    return ARCHITECTURE_CAPABILITIES.get(
        architecture,
        ArchitecturePlacementCapabilities(False, False, False, False, False),
    )
