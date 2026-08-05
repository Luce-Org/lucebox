"""Bounded, quality-safe calibration against a running Lucebox server.

The automatic planner owns feature selection and placement.  Calibration is
deliberately narrower: it measures the resolved plan on the actual machine and
only brackets DDTree's startup budget on backends that consume that value.
Engine-owned adaptive policies (Spark, KVFlash sizing, PFlash request policy,
and DSpark width) are observed, never duplicated here.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

import lucebox.autotune as autotune_mod
import lucebox.config as config_mod
import lucebox.download as download_mod
from lucebox import __version__
from lucebox.types import Config

SCHEMA_VERSION = 1
MIN_WINNER_GAIN = 0.05
_BUDGET_GRID = (4, 8, 16, 22, 32, 40, 64)
_BUDGET_ARCHITECTURES = frozenset({"qwen35", "qwen35moe", "laguna"})


@dataclass(frozen=True, slots=True)
class TurnMeasurement:
    """Server-reported timings for one measured request."""

    decode_tokens_per_sec: float
    prefill_tokens_per_sec: float | None
    completion_tokens: int
    cache_hit: bool
    cached_prefix_tokens: int
    prefilled_tokens: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Any) -> TurnMeasurement:
        if not isinstance(raw, dict) or not isinstance(raw.get("cache_hit"), bool):
            raise ValueError("malformed turn measurement")
        try:
            raw_prefill = raw["prefill_tokens_per_sec"]
            prefill = None if raw_prefill is None else float(raw_prefill)
            result = cls(
                decode_tokens_per_sec=float(raw["decode_tokens_per_sec"]),
                prefill_tokens_per_sec=prefill,
                completion_tokens=int(raw["completion_tokens"]),
                cache_hit=raw["cache_hit"],
                cached_prefix_tokens=int(raw["cached_prefix_tokens"]),
                prefilled_tokens=int(raw["prefilled_tokens"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("malformed turn measurement") from exc
        if (
            not math.isfinite(result.decode_tokens_per_sec)
            or result.decode_tokens_per_sec <= 0.0
            or (
                result.prefill_tokens_per_sec is not None
                and (
                    not math.isfinite(result.prefill_tokens_per_sec)
                    or result.prefill_tokens_per_sec <= 0.0
                )
            )
            or result.completion_tokens <= 0
            or result.cached_prefix_tokens < 0
            or result.prefilled_tokens < 0
        ):
            raise ValueError("invalid turn measurement")
        return result


@dataclass(frozen=True, slots=True)
class ProbeResult:
    """Comparable result for one server startup budget."""

    budget: int
    model: str
    architecture: str
    score: float
    response_signature: str
    cold: TurnMeasurement
    warm: TurnMeasurement
    server: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA_VERSION,
            "budget": self.budget,
            "model": self.model,
            "architecture": self.architecture,
            "score": self.score,
            "response_signature": self.response_signature,
            "cold": self.cold.as_dict(),
            "warm": self.warm.as_dict(),
            "server": self.server,
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> ProbeResult:
        if raw.get("schema") != SCHEMA_VERSION:
            raise ValueError("unsupported calibration result schema")
        try:
            result = cls(
                budget=int(raw["budget"]),
                model=str(raw["model"]),
                architecture=str(raw["architecture"]),
                score=float(raw["score"]),
                response_signature=str(raw["response_signature"]),
                cold=TurnMeasurement.from_dict(raw["cold"]),
                warm=TurnMeasurement.from_dict(raw["warm"]),
                server=dict(raw["server"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("malformed calibration result") from exc
        if (
            not 0 <= result.budget <= 64
            or not result.model
            or not result.architecture
            or not math.isfinite(result.score)
            or result.score <= 0.0
            or not result.response_signature
        ):
            raise ValueError("invalid calibration result")
        return result


@dataclass(frozen=True, slots=True)
class CalibrationSummary:
    """Winner plus all successfully measured candidates."""

    baseline_budget: int
    winner: ProbeResult
    results: tuple[ProbeResult, ...]
    rejected_budgets: tuple[int, ...]


def _preset(cfg: Config) -> download_mod.ModelPreset | None:
    return download_mod.PRESETS.get(cfg.model.preset)


def budget_is_tunable(cfg: Config) -> bool:
    """Whether changing ``dflash.budget`` can affect this resolved runtime."""
    preset = _preset(cfg)
    return bool(
        preset is not None
        and preset.architecture in _BUDGET_ARCHITECTURES
        and cfg.dflash.speculative_decode
        and not cfg.dflash.spark
        and 4 <= cfg.dflash.budget <= 64
        and autotune_mod.draft_available(cfg, preset)
    )


def candidate_budgets(cfg: Config) -> tuple[int, ...]:
    """Return baseline first, followed by its two nearest safe neighbours."""
    current = cfg.dflash.budget
    if not budget_is_tunable(cfg):
        return (current,)
    lower = [value for value in _BUDGET_GRID if value < current]
    upper = [value for value in _BUDGET_GRID if value > current]
    candidates = [current]
    if lower:
        candidates.append(lower[-1])
    if upper:
        candidates.append(upper[0])
    return tuple(candidates)


def apply_budget(budget: int, *, final: bool = False) -> None:
    """Atomically apply one calibration cell without changing profile ownership."""
    if not 0 <= budget <= 64:
        raise ValueError(f"calibration budget must be in [0, 64], got {budget}")
    cfg = config_mod.load()
    if cfg is None:
        raise ValueError("calibration requires an existing config.toml")
    mode = config_mod.optimization_mode()
    if mode not in {"automatic", "custom"}:
        raise ValueError("run `lucebox optimize` before calibration")
    config_mod.write_optimization_runtime(
        replace(cfg.dflash, budget=budget),
        placement=cfg.placement,
        mode=mode,
        source="calibrated" if final else "calibration-candidate",
    )


def _base_urls(cfg: Config) -> tuple[str, ...]:
    explicit = os.environ.get("LUCEBOX_CALIBRATION_URL", "").rstrip("/")
    if explicit:
        return (explicit,)
    urls = [f"http://127.0.0.1:{cfg.port}"]
    # A probe executed inside the inference container shares its network
    # namespace, where the server always listens on the internal port 8080.
    if cfg.port != 8080:
        urls.append("http://127.0.0.1:8080")
    return tuple(urls)


def _wait_for_server(
    client: httpx.Client,
    base_urls: tuple[str, ...],
    timeout_s: float,
) -> tuple[str, dict[str, Any]]:
    deadline = time.monotonic() + timeout_s
    last_error = "server did not respond"
    while time.monotonic() < deadline:
        for base_url in base_urls:
            try:
                response = client.get(f"{base_url}/props", timeout=3.0)
                response.raise_for_status()
                body = response.json()
                if isinstance(body, dict):
                    return base_url, body
                last_error = "/props did not return an object"
            except (httpx.HTTPError, json.JSONDecodeError, ValueError) as exc:
                last_error = str(exc)
        time.sleep(1.0)
    raise TimeoutError(f"Lucebox was not ready after {timeout_s:g}s: {last_error}")


def _representative_context() -> str:
    """Compactly generate a repeatable repository-shaped prefill workload."""
    return "\n".join(
        f"src/worker_{index:02d}.py: parse request, validate job {index}, "
        f"write an atomic result, and preserve cancellation state."
        for index in range(12)
    )


_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a UTF-8 source file from the repository.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": "Run a named, already-installed test target.",
            "parameters": {
                "type": "object",
                "properties": {"target": {"type": "string"}},
                "required": ["target"],
                "additionalProperties": False,
            },
        },
    },
]


def _request_body(
    messages: list[dict[str, Any]],
    model: str,
    max_tokens: int,
    *,
    include_tools: bool,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": model or "lucebox",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "seed": 20260801,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if include_tools:
        body["tools"] = _TOOLS
        # Keep the representative tool schema in the prompt while making the
        # measured completion deterministic and directly comparable across
        # candidate runtimes.  Tool execution itself is outside calibration.
        body["tool_choice"] = "none"
    return body


def _assistant_turn(
    payload: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return a replayable assistant message and a stable quality signature."""
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ValueError("chat response has no choice")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ValueError("chat response has no assistant message")

    parts = [message.get("reasoning_content"), message.get("content")]
    text = "\n".join(part for part in parts if isinstance(part, str) and part.strip()).strip()
    replay: dict[str, Any] = {"role": "assistant", "content": text or None}
    normalized_calls: list[dict[str, Any]] = []
    raw_calls = message.get("tool_calls", [])
    if raw_calls is None:
        raw_calls = []
    if not isinstance(raw_calls, list):
        raise ValueError("chat response has malformed tool calls")
    replay_calls: list[dict[str, Any]] = []
    for index, raw_call in enumerate(raw_calls):
        if not isinstance(raw_call, dict):
            raise ValueError("chat response has malformed tool calls")
        raw_function = raw_call.get("function")
        if not isinstance(raw_function, dict):
            raise ValueError("chat response has malformed tool calls")
        name = raw_function.get("name")
        arguments = raw_function.get("arguments", "")
        if not isinstance(name, str) or not name:
            raise ValueError("chat response has a nameless tool call")
        if not isinstance(arguments, str):
            arguments = json.dumps(arguments, sort_keys=True, separators=(",", ":"))
        try:
            normalized_arguments: Any = json.loads(arguments)
        except json.JSONDecodeError:
            normalized_arguments = arguments.strip()
        call_id = raw_call.get("id")
        if not isinstance(call_id, str) or not call_id:
            call_id = f"calibration_call_{index}"
        replay_calls.append(
            {
                "id": call_id,
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        )
        normalized_calls.append(
            {
                "type": "function",
                "function": {"name": name, "arguments": normalized_arguments},
            }
        )
    if replay_calls:
        replay["tool_calls"] = replay_calls
    if not text and not replay_calls:
        raise ValueError("chat response was empty")
    return replay, {"text": text, "tool_calls": normalized_calls}


def _continue_conversation(
    messages: list[dict[str, Any]],
    assistant: dict[str, Any],
    user_content: str,
) -> list[dict[str, Any]]:
    """Append a turn, simulating any requested tool result without executing it."""
    continued = [*messages, assistant]
    calls = assistant.get("tool_calls", [])
    if isinstance(calls, list):
        for call in calls:
            if isinstance(call, dict) and isinstance(call.get("id"), str):
                continued.append(
                    {
                        "role": "tool",
                        "tool_call_id": call["id"],
                        "content": '{"status":"skipped","reason":"calibration"}',
                    }
                )
    continued.append({"role": "user", "content": user_content})
    return continued


def _measurement(payload: dict[str, Any]) -> TurnMeasurement:
    usage = payload.get("usage")
    if not isinstance(usage, dict):
        raise ValueError("chat response has no usage object")
    timings = usage.get("timings")
    if not isinstance(timings, dict):
        raise ValueError("server does not expose usage.timings")
    try:
        decode_tps = float(timings["decode_tokens_per_sec"])
        prefill_ms = float(timings["prefill_ms"])
        prefilled = int(timings["prefilled_tokens"])
        completion = int(usage["completion_tokens"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("usage.timings is incomplete") from exc
    if not math.isfinite(decode_tps) or decode_tps <= 0.0 or completion <= 0:
        raise ValueError("server returned a non-positive decode measurement")
    prefill_tps = None
    if prefill_ms > 0.0 and prefilled > 0:
        prefill_tps = prefilled * 1000.0 / prefill_ms
    return TurnMeasurement(
        decode_tokens_per_sec=decode_tps,
        prefill_tokens_per_sec=prefill_tps,
        completion_tokens=completion,
        cache_hit=bool(timings.get("cache_hit", False)),
        cached_prefix_tokens=int(timings.get("cached_prefix_tokens", 0)),
        prefilled_tokens=prefilled,
    )


def _chat(
    client: httpx.Client,
    base_url: str,
    body: dict[str, Any],
    timeout_s: float,
) -> dict[str, Any]:
    response = client.post(
        f"{base_url}/v1/chat/completions",
        json=body,
        timeout=timeout_s,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("chat endpoint did not return an object")
    return payload


def probe(
    cfg: Config,
    expected_budget: int,
    *,
    ready_timeout_s: float = 600.0,
    request_timeout_s: float = 300.0,
    client: httpx.Client | None = None,
    base_urls: tuple[str, ...] | None = None,
) -> ProbeResult:
    """Measure cold prefill, decode, and warm multi-turn prefix reuse."""
    owned_client = client is None
    active_client = client or httpx.Client(trust_env=False)
    try:
        base_url, props = _wait_for_server(
            active_client,
            base_urls or _base_urls(cfg),
            ready_timeout_s,
        )
        speculative = props.get("speculative")
        if budget_is_tunable(cfg):
            actual = speculative.get("ddtree_budget") if isinstance(speculative, dict) else None
            if actual != expected_budget:
                raise ValueError(
                    f"server started with DDTree budget {actual!r}, expected {expected_budget}"
                )
        capabilities = props.get("capabilities")
        include_tools = bool(
            isinstance(capabilities, dict) and capabilities.get("tools_supported")
        )

        # Warm kernels and allocator paths without polluting the measured prefix.
        _chat(
            active_client,
            base_url,
            _request_body(
                [{"role": "user", "content": "Reply with only the word ready."}],
                cfg.model.preset,
                8,
                include_tools=include_tools,
            ),
            request_timeout_s,
        )

        first_messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "You are a concise senior coding assistant. Do not call tools for this "
                    "calibration request. Preserve behavior, cancellation, and atomic writes."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Here is a synthetic repository index:\n"
                    f"{_representative_context()}\n\n"
                    "Write a short Python function that atomically records a completed job. "
                    "Return code plus one sentence."
                ),
            },
        ]
        first = _chat(
            active_client,
            base_url,
            _request_body(
                first_messages,
                cfg.model.preset,
                64,
                include_tools=include_tools,
            ),
            request_timeout_s,
        )
        first_message, first_signature = _assistant_turn(first)
        second_messages = _continue_conversation(
            first_messages,
            first_message,
            "Now add idempotency and keep the answer under 12 lines.",
        )
        second = _chat(
            active_client,
            base_url,
            _request_body(
                second_messages,
                cfg.model.preset,
                64,
                include_tools=include_tools,
            ),
            request_timeout_s,
        )
        second_message, second_signature = _assistant_turn(second)
        third_messages = _continue_conversation(
            second_messages,
            second_message,
            "Finally add type hints without changing the behavior.",
        )
        third = _chat(
            active_client,
            base_url,
            _request_body(
                third_messages,
                cfg.model.preset,
                64,
                include_tools=include_tools,
            ),
            request_timeout_s,
        )
        _, third_signature = _assistant_turn(third)
        cold = _measurement(first)
        warm = _measurement(third)
        score = 2.0 / (
            1.0 / cold.decode_tokens_per_sec + 1.0 / warm.decode_tokens_per_sec
        )
        signature_body = json.dumps(
            [first_signature, second_signature, third_signature],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        signature = hashlib.sha256(signature_body.encode()).hexdigest()
        raw_model_props = props.get("model")
        model_props: dict[str, Any] = (
            raw_model_props if isinstance(raw_model_props, dict) else {}
        )
        preset = _preset(cfg)
        return ProbeResult(
            budget=expected_budget,
            model=cfg.model.preset,
            architecture=str(
                model_props.get("arch") or (preset.architecture if preset else "")
            ),
            score=score,
            response_signature=signature,
            cold=cold,
            warm=warm,
            server={
                "build_info": props.get("build_info"),
                "runtime": props.get("runtime"),
                "speculative": speculative,
                "pflash": props.get("pflash"),
                "prefix_cache": props.get("prefix_cache"),
            },
        )
    except httpx.HTTPError as exc:
        raise ValueError(f"server request failed: {exc}") from exc
    finally:
        if owned_client:
            active_client.close()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    try:
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def write_probe(path: Path, result: ProbeResult) -> None:
    _atomic_json(path, result.as_dict())


def read_probe(path: Path) -> ProbeResult:
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read calibration result {path}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"calibration result {path} is not an object")
    return ProbeResult.from_dict(raw)


def calibration_record_path() -> Path:
    return config_mod.default_config_path().with_name("calibration.json")


def _artifact_stat(cfg: Config, relative: Path) -> dict[str, Any]:
    logical = cfg.models_dir / relative
    container = Path("/opt/lucebox-hub/server/models") / relative
    stat = None
    for candidate in (logical, container):
        try:
            stat = candidate.stat()
            break
        except OSError:
            continue
    if stat is None:
        return {"file": relative.as_posix(), "present": False}
    return {
        "file": relative.as_posix(),
        "present": True,
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }


def _fingerprint_payload(cfg: Config) -> dict[str, Any]:
    preset = _preset(cfg)
    artifacts: list[dict[str, Any]] = []
    target = cfg.model.target_file or (preset.target_file if preset else "")
    draft = cfg.model.draft_file or (preset.draft_file if preset else "")
    if target:
        artifacts.append(_artifact_stat(cfg, Path(target)))
    if draft:
        artifacts.append(_artifact_stat(cfg, Path("draft") / draft))
    if preset is not None and preset.speculator_dir:
        root = Path("draft") / preset.speculator_dir
        artifacts.extend(_artifact_stat(cfg, root / name) for name in preset.speculator_files)
    host = cfg.host
    return {
        "cli_version": __version__,
        "variant": cfg.variant,
        "image": cfg.image,
        "model": asdict(cfg.model),
        "runtime": asdict(cfg.dflash),
        "placement": asdict(cfg.placement),
        "host": {
            "nproc": host.nproc,
            "ram_gb": host.ram_gb,
            "gpu_name": host.gpu_name,
            "gpu_count": host.gpu_count,
            "gpu_sm": host.gpu_sm,
            "vram_gb": host.vram_gb,
            "driver_version": host.driver_version,
            "rocm_version": host.rocm_version,
            "is_wsl": host.is_wsl,
            "nvidia_gpu_name": host.nvidia_gpu_name,
            "nvidia_gpu_count": host.nvidia_gpu_count,
            "nvidia_vram_gb": host.nvidia_vram_gb,
            "nvidia_gpu_arch": host.nvidia_gpu_arch,
            "nvidia_gpu_list_csv": host.nvidia_gpu_list_csv,
            "amd_gpu_name": host.amd_gpu_name,
            "amd_gpu_count": host.amd_gpu_count,
            "amd_vram_gb": host.amd_vram_gb,
            "amd_gpu_arch": host.amd_gpu_arch,
            "amd_gpu_list_csv": host.amd_gpu_list_csv,
            "hybrid_runtime": host.hybrid_runtime,
        },
        "artifacts": artifacts,
    }


def fingerprint(cfg: Config) -> str:
    payload = json.dumps(_fingerprint_payload(cfg), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def current_record(cfg: Config) -> dict[str, Any] | None:
    try:
        raw = json.loads(calibration_record_path().read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(raw, dict) or raw.get("schema") != SCHEMA_VERSION:
        return None
    return raw if raw.get("fingerprint") == fingerprint(cfg) else None


def finish(
    result_dir: Path,
    baseline_budget: int,
    *,
    cfg: Config | None = None,
) -> CalibrationSummary:
    """Select a quality-equivalent winner, persist it, and cache the evidence."""
    results = tuple(
        read_probe(path) for path in sorted(result_dir.glob("budget-*.json"))
    )
    if not results:
        raise ValueError("calibration produced no successful measurements")
    baseline = next((item for item in results if item.budget == baseline_budget), None)
    if baseline is None:
        raise ValueError("baseline calibration measurement failed")
    if not math.isfinite(baseline.score) or baseline.score <= 0.0:
        raise ValueError("baseline calibration score is invalid")

    equivalent = tuple(
        item
        for item in results
        if item.model == baseline.model
        and item.response_signature == baseline.response_signature
        and item.warm.cache_hit == baseline.warm.cache_hit
        and item.warm.cached_prefix_tokens == baseline.warm.cached_prefix_tokens
        and math.isfinite(item.score)
        and item.score > 0.0
    )
    rejected = tuple(item.budget for item in results if item not in equivalent)
    fastest = max(equivalent, key=lambda item: item.score)
    winner = (
        fastest
        if fastest.score >= baseline.score * (1.0 + MIN_WINNER_GAIN)
        else baseline
    )
    apply_budget(winner.budget, final=True)
    live_cfg = cfg or config_mod.load()
    if live_cfg is None:  # pragma: no cover - apply_budget already proved this
        raise RuntimeError("config disappeared while finishing calibration")
    final_cfg = replace(
        live_cfg,
        dflash=replace(live_cfg.dflash, budget=winner.budget),
    )
    record = {
        "schema": SCHEMA_VERSION,
        "captured_at": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "fingerprint": fingerprint(final_cfg),
        "baseline_budget": baseline_budget,
        "winner_budget": winner.budget,
        "minimum_gain": MIN_WINNER_GAIN,
        "results": [item.as_dict() for item in results],
        "rejected_budgets": list(rejected),
    }
    _atomic_json(calibration_record_path(), record)
    (result_dir / "winner").write_text(f"{winner.budget}\n")
    return CalibrationSummary(
        baseline_budget=baseline_budget,
        winner=winner,
        results=results,
        rejected_budgets=rejected,
    )
