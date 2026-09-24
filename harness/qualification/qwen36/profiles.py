from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import yaml


class ProfileError(ValueError):
    pass


@dataclass(frozen=True)
class Artifact:
    path: Path
    sha256: str
    quantization: str


@dataclass(frozen=True)
class Accelerator:
    role: str
    name: str
    architecture: str


@dataclass(frozen=True)
class Workload:
    name: str
    concurrency: int
    maximum_tokens: int
    warmups: int
    repetitions: int
    require_identical: bool = False
    temperature: float = 0.0


@dataclass(frozen=True)
class QwenArProfile:
    name: str
    family: str
    recipe_id: str
    qualification_runner: Literal["qwen36_amd"]
    modality: Literal["text"]
    runner: str
    decode_mode: Literal["autoregressive"]
    feature_set: str
    server_arguments: tuple[str, ...]
    paged_attention: bool
    maximum_concurrency: int
    topology: Literal["single-device"]
    artifact: Artifact
    accelerator: Accelerator
    visible_device: str
    telemetry: Literal["amd"]
    power_profile: Literal["platform-managed"]
    server_port: int
    maximum_context: int
    prefix_cache_slots: int
    smoke_prompts: Path
    smoke_workloads: tuple[Workload, ...]
    quality_prompts: Path
    quality_maximum_tokens: int
    minimum_gold_accuracy: float
    performance_prompts: Path
    performance_workloads: tuple[Workload, ...]
    monitor_interval_seconds: float
    minimum_monitor_samples: int
    steady_window_samples: int
    maximum_rss_growth_mib: float
    maximum_health_latency_growth_fraction: float

    def artifact_path(self, models_root: Path) -> Path:
        root = models_root.resolve()
        target = (root / self.artifact.path).resolve()
        if not target.is_relative_to(root):
            raise ValueError(f"artifact path escapes the models root: {self.artifact.path}")
        return target

    def environment(self) -> dict[str, str]:
        return {"HIP_VISIBLE_DEVICES": self.visible_device}

    def snapshot(self) -> dict[str, Any]:
        def jsonable(value: Any) -> Any:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, dict):
                return {key: jsonable(item) for key, item in value.items()}
            if isinstance(value, list | tuple):
                return [jsonable(item) for item in value]
            return value

        return {"schema_version": 1, **jsonable(asdict(self))}


def _mapping(value: Any, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ProfileError(f"{field} must be a map")
    return value


def _positive_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ProfileError(f"{field} must be a positive integer")
    return value


def _non_negative_int(value: Any, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ProfileError(f"{field} must be a non-negative integer")
    return value


def _finite_float(value: Any, field: str) -> float:
    if isinstance(value, bool):
        raise ProfileError(f"{field} must be a number")
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ProfileError(f"{field} must be a number") from error
    if not math.isfinite(result):
        raise ProfileError(f"{field} must be finite")
    return result


def _sha256(value: Any, field: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ProfileError(f"{field} must be a lowercase SHA-256")
    return value


def _path(value: Any, field: str) -> Path:
    path = Path(value) if isinstance(value, str) else None
    if path is None or not value or path.is_absolute():
        raise ProfileError(f"{field} must be a non-empty relative path")
    if ".." in path.parts:
        raise ProfileError(f"{field} must be relative without parent-directory components")
    return path


def _identifier(value: Any, field: str) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", value):
        raise ProfileError(f"{field} must be a lowercase kebab-case identifier")
    return value


_RESERVED_SERVER_ARGUMENTS = {
    "--help",
    "--host",
    "--max-concurrency",
    "--max-ctx",
    "--paged-attention",
    "--port",
    "--prefix-cache-slots",
    "--version",
}
_NAMED_FEATURE_ARGUMENTS = {"specla": "--specla"}


def _server_arguments(value: Any, field: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ProfileError(f"{field} must be an array of argument tokens")
    if any(not isinstance(token, str) or not token or "\x00" in token for token in value):
        raise ProfileError(f"{field} must contain non-empty string argument tokens")
    if value and not value[0].startswith("--"):
        raise ProfileError(f"{field} must start with an option")

    options = [token.split("=", 1)[0] for token in value if token.startswith("--")]
    duplicates = sorted(option for option in set(options) if options.count(option) > 1)
    if duplicates:
        raise ProfileError(f"{field} contains duplicate options: {', '.join(duplicates)}")
    reserved = sorted(set(options) & _RESERVED_SERVER_ARGUMENTS)
    if reserved:
        raise ProfileError(f"{field} cannot override profile-owned options: {', '.join(reserved)}")
    if any(token.startswith("--") and "=" in token for token in value):
        raise ProfileError(f"{field} must use separate option and value tokens")
    return tuple(value)


def _validate_feature_arguments(feature_set: str, server_arguments: tuple[str, ...]) -> None:
    labels = set(feature_set.split("-"))
    options = {token.split("=", 1)[0] for token in server_arguments if token.startswith("--")}
    for label, option in _NAMED_FEATURE_ARGUMENTS.items():
        labeled = label in labels
        enabled = option in options
        if labeled != enabled:
            state = "requires" if labeled else "does not allow"
            raise ProfileError(
                f"feature_set {feature_set!r} {state} {option}; use a separately named profile "
                "for each feature configuration"
            )


def _workloads(values: Any, field: str) -> tuple[Workload, ...]:
    if not isinstance(values, list) or not values:
        raise ProfileError(f"{field} must be a non-empty list")
    workloads = []
    for index, value in enumerate(values):
        item = _mapping(value, f"{field}[{index}]")
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            raise ProfileError(f"{field}[{index}].name must be a non-empty string")
        require_identical = item.get("require_identical", False)
        if not isinstance(require_identical, bool):
            raise ProfileError(f"{field}[{index}].require_identical must be a boolean")
        temperature = _finite_float(item.get("temperature", 0.0), f"{field}[{index}].temperature")
        if temperature != 0:
            raise ProfileError(f"{field}[{index}] workload temperature must be zero")
        workloads.append(
            Workload(
                name=name,
                concurrency=_positive_int(item.get("concurrency"), f"{field}[{index}].concurrency"),
                maximum_tokens=_positive_int(
                    item.get("maximum_tokens"), f"{field}[{index}].maximum_tokens"
                ),
                warmups=_non_negative_int(item.get("warmups"), f"{field}[{index}].warmups"),
                repetitions=_positive_int(item.get("repetitions"), f"{field}[{index}].repetitions"),
                require_identical=require_identical,
                temperature=temperature,
            )
        )
    concurrencies = [workload.concurrency for workload in workloads]
    if len(concurrencies) != len(set(concurrencies)):
        raise ProfileError(f"{field} contains duplicate concurrency values")
    return tuple(workloads)


def load_qwen_ar_profile(path: Path, name: str) -> QwenArProfile:
    manifest = _mapping(yaml.safe_load(path.read_text()), "model manifest")
    if manifest.get("schema_version") != 2:
        raise ProfileError("model manifest schema_version must be 2")
    families = _mapping(manifest.get("families"), "families")
    profiles = _mapping(manifest.get("profiles"), "profiles")
    if name not in profiles:
        raise ProfileError(
            f"unknown production profile {name!r}; available profiles: {', '.join(sorted(profiles))}"
        )
    binding = _mapping(profiles[name], f"profiles.{name}")
    if binding.get("status") != "active":
        raise ProfileError(f"production profile {name!r} is not active")
    family = binding.get("family")
    if family not in families:
        raise ProfileError(f"production profile {name!r} has unknown family {family!r}")
    family_config = _mapping(families[family], f"families.{family}")
    recipes = _mapping(family_config.get("recipes"), f"families.{family}.recipes")
    recipe_id = binding.get("recipe")
    if not isinstance(recipe_id, str) or recipe_id not in recipes:
        raise ProfileError(f"production profile {name!r} has unknown recipe {recipe_id!r}")
    recipe = _mapping(recipes[recipe_id], f"families.{family}.recipes.{recipe_id}")
    if recipe.get("modality") != "text" or recipe.get("decode_mode") != "autoregressive":
        raise ProfileError("the active Qwen runner supports text autoregressive profiles only")
    feature_set = _identifier(recipe.get("feature_set"), "feature_set")
    if f"-{feature_set}-" not in f"-{recipe_id}-" or f"-{feature_set}-" not in f"-{name}-":
        raise ProfileError("recipe and profile must include the feature_set")
    server_arguments = _server_arguments(recipe.get("server_arguments"), "server_arguments")
    _validate_feature_arguments(feature_set, server_arguments)
    if binding.get("topology") != "single-device":
        raise ProfileError("the active Qwen runner requires single-device topology")
    devices = binding.get("devices")
    if not isinstance(devices, list) or len(devices) != 1:
        raise ProfileError("the active Qwen runner requires exactly one device")
    device = _mapping(devices[0], "devices[0]")
    device_identity = tuple(device.get(field) for field in ("role", "name", "arch"))
    if any(not isinstance(value, str) or not value.strip() for value in device_identity):
        raise ProfileError("device role, name, and arch must be non-empty strings")
    device_role, device_name, device_arch = device_identity
    qualification = _mapping(recipe.get("qualification"), "qualification")
    smoke = _mapping(qualification.get("smoke"), "qualification.smoke")
    quality = _mapping(qualification.get("quality"), "qualification.quality")
    performance = _mapping(qualification.get("performance"), "qualification.performance")
    drift = _mapping(qualification.get("drift"), "qualification.drift")
    smoke_workloads = _workloads(smoke.get("workloads"), "qualification.smoke.workloads")
    performance_workloads = _workloads(
        performance.get("workloads"), "qualification.performance.workloads"
    )
    maximum_concurrency = _positive_int(recipe.get("max_concurrency"), "max_concurrency")
    if maximum_concurrency < 2:
        raise ProfileError("max_concurrency must be at least 2")
    if (
        max(workload.concurrency for workload in smoke_workloads + performance_workloads)
        > maximum_concurrency
    ):
        raise ProfileError("workload concurrency exceeds max_concurrency")
    if {workload.concurrency for workload in smoke_workloads} != {1, maximum_concurrency}:
        raise ProfileError("smoke workloads must cover c1 and the profile maximum concurrency")
    maximum_smoke = next(
        workload for workload in smoke_workloads if workload.concurrency == maximum_concurrency
    )
    if not maximum_smoke.require_identical:
        raise ProfileError("maximum-concurrency smoke workload must require identical output")
    if qualification.get("runner") != "qwen36_amd":
        raise ProfileError("the active profile requires the qwen36_amd runner")
    if not binding.get("runner") or not recipe.get("quant"):
        raise ProfileError("runner and quant must be non-empty")
    if binding.get("required_telemetry") != "amd":
        raise ProfileError("the active Qwen profile requires AMD telemetry")
    if binding.get("power_profile") != "platform-managed":
        raise ProfileError("the active Qwen profile uses platform-managed power")
    visible_device = binding.get("visible_device")
    if not isinstance(visible_device, str) or not re.fullmatch(r"[0-9]+", visible_device):
        raise ProfileError("the active Qwen profile requires an ASCII decimal visible_device")
    if maximum_concurrency > 1 and recipe.get("paged_attention") is not True:
        raise ProfileError("concurrent AR requires paged attention")
    sample_shapes = {
        (
            workload.maximum_tokens,
            workload.warmups,
            workload.repetitions,
            workload.temperature,
        )
        for workload in performance_workloads
    }
    if len(sample_shapes) != 1:
        raise ProfileError("performance workloads may differ only by concurrency and name")
    prefix_cache_slots = _non_negative_int(
        qualification.get("prefix_cache_slots"), "qualification.prefix_cache_slots"
    )
    minimum_gold_accuracy = _finite_float(
        quality.get("minimum_gold_accuracy", -1), "quality.minimum_gold_accuracy"
    )
    monitor_interval_seconds = _finite_float(
        drift.get("monitor_interval_seconds", 0), "drift.monitor_interval_seconds"
    )
    maximum_rss_growth_mib = _finite_float(
        drift.get("maximum_rss_growth_mib", -1), "drift.maximum_rss_growth_mib"
    )
    maximum_health_latency_growth_fraction = _finite_float(
        drift.get("maximum_health_latency_growth_fraction", -1),
        "drift.maximum_health_latency_growth_fraction",
    )
    if not 0 <= minimum_gold_accuracy <= 1:
        raise ProfileError("minimum_gold_accuracy must be between zero and one")
    if (
        monitor_interval_seconds <= 0
        or maximum_rss_growth_mib < 0
        or maximum_health_latency_growth_fraction < 0
    ):
        raise ProfileError("drift interval and limits must be non-negative")
    minimum_monitor_samples = _positive_int(drift.get("minimum_samples"), "drift.minimum_samples")
    if minimum_monitor_samples < 2:
        raise ProfileError("drift.minimum_samples must be at least 2")
    server_port = _positive_int(qualification.get("server_port"), "server_port")
    if server_port > 65535:
        raise ProfileError("server_port must be at most 65535")

    return QwenArProfile(
        name=name,
        family=str(family),
        recipe_id=str(recipe_id),
        qualification_runner="qwen36_amd",
        modality="text",
        runner=str(binding.get("runner", "")),
        decode_mode="autoregressive",
        feature_set=feature_set,
        server_arguments=server_arguments,
        paged_attention=recipe.get("paged_attention") is True,
        maximum_concurrency=maximum_concurrency,
        topology="single-device",
        artifact=Artifact(
            _path(recipe.get("target_artifact"), "target_artifact"),
            _sha256(recipe.get("target_sha256"), "target_sha256"),
            str(recipe.get("quant", "")),
        ),
        accelerator=Accelerator(
            device_role, device_name, device_arch
        ),
        visible_device=visible_device,
        telemetry=str(binding.get("required_telemetry", "")),
        power_profile=str(binding.get("power_profile", "")),
        server_port=server_port,
        maximum_context=_positive_int(qualification.get("max_context"), "max_context"),
        prefix_cache_slots=prefix_cache_slots,
        smoke_prompts=_path(smoke.get("prompts"), "smoke.prompts"),
        smoke_workloads=smoke_workloads,
        quality_prompts=_path(quality.get("prompts"), "quality.prompts"),
        quality_maximum_tokens=_positive_int(
            quality.get("maximum_tokens"), "quality.maximum_tokens"
        ),
        minimum_gold_accuracy=minimum_gold_accuracy,
        performance_prompts=_path(performance.get("prompts"), "performance.prompts"),
        performance_workloads=performance_workloads,
        monitor_interval_seconds=monitor_interval_seconds,
        minimum_monitor_samples=minimum_monitor_samples,
        steady_window_samples=_positive_int(
            drift.get("steady_window_samples"), "drift.steady_window_samples"
        ),
        maximum_rss_growth_mib=maximum_rss_growth_mib,
        maximum_health_latency_growth_fraction=maximum_health_latency_growth_fraction,
    )
