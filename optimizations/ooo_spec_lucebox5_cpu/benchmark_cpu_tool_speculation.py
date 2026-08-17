#!/usr/bin/env python3
"""Qualify and benchmark a disjoint Strix CPU speculative-tool lane.

The qualification phase runs the official model server without tool
speculation, pins the real sparse-compute tool to reserved physical cores, and
measures sequential versus overlapped execution. It emits a qualified engine
profile only if model output, DS4 activity, tool results, CPU isolation, and
the slowdown gate all pass.

The native phase then measures the engine's exact-call commit path against the
same strong sequential CPU baseline. Wrong predictions are cancelled and
their private results must never be exposed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import statistics
import subprocess
import time
import urllib.request
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import urlsplit, urlunsplit


TOOL_NAME = "benchmark_cpu_sparse"
PROTOCOL = "dflash.tool-speculation.v1"
NATIVE_PREDICTION_SOURCE = "native-qwen3"
SPARSE_ROWS = 4096
SPARSE_NONZEROS_PER_ROW = 16
SPARSE_THREADS = 2
SPARSE_SEED = 731


def parse_cpu_list(value: str) -> list[int]:
    cpus: set[int] = set()
    for item in value.split(","):
        if not item:
            raise argparse.ArgumentTypeError("CPU list contains an empty item")
        if "-" in item:
            parts = item.split("-")
            if len(parts) != 2 or not all(part.isdigit() for part in parts):
                raise argparse.ArgumentTypeError(f"invalid CPU range: {item}")
            first, last = map(int, parts)
            if first > last:
                raise argparse.ArgumentTypeError(f"invalid CPU range: {item}")
            cpus.update(range(first, last + 1))
        elif item.isdigit():
            cpus.add(int(item))
        else:
            raise argparse.ArgumentTypeError(f"invalid CPU id: {item}")
    if not cpus:
        raise argparse.ArgumentTypeError("CPU list must not be empty")
    return sorted(cpus)


def compact_cpu_list(cpus: Iterable[int]) -> str:
    return ",".join(str(cpu) for cpu in cpus)


def expected_arguments(
    rows: int,
    nonzeros_per_row: int,
    iterations: int,
    threads: int,
    seed: int,
) -> dict[str, int]:
    if (
        rows != SPARSE_ROWS
        or nonzeros_per_row != SPARSE_NONZEROS_PER_ROW
        or threads != SPARSE_THREADS
        or seed != SPARSE_SEED
    ):
        raise ValueError("this qualification binary has a fixed sparse shape")
    # Keep the generated tool call intentionally short. The deterministic
    # benchmark binary owns the qualified sparse shape; only work duration is
    # request-dependent, matching real tools with a compact identifier.
    return {"iterations": iterations}


def tool_definition() -> dict[str, Any]:
    properties = {"iterations": {"type": "integer"}}
    return {
        "name": TOOL_NAME,
        "parameters": {
            "type": "object",
            "properties": properties,
            "required": list(properties),
            "additionalProperties": False,
        },
    }


def request_body(
    arguments: dict[str, int],
    max_tokens: int,
    *,
    prediction: dict[str, int] | None,
    automatic_prediction: bool = False,
    tool_choice: str | None = None,
) -> dict[str, Any]:
    compact = json.dumps(arguments, separators=(",", ":"))
    prompt = f"Return only this JSON object and nothing else: {compact}"
    body: dict[str, Any] = {
        "model": "dflash",
        "stream": False,
        "max_tokens": max_tokens,
        "temperature": 0,
        "messages": [
            {
                "role": "user",
                "content": prompt,
            }
        ],
        "tools": [tool_definition()],
        "automatic_tool_speculation": automatic_prediction,
    }
    if tool_choice is not None:
        body["tool_choice"] = tool_choice
    if prediction is not None:
        body["tool_speculation"] = {
            "call": {"name": TOOL_NAME, "arguments": prediction},
            "confidence": 1.0,
        }
    return body


def normalize_tool_call(result: dict[str, Any]) -> dict[str, Any] | None:
    message = result.get("choices", [{}])[0].get("message", {})
    tool_calls = message.get("tool_calls") or []
    if len(tool_calls) == 1:
        function = tool_calls[0].get("function") or {}
        arguments = function.get("arguments", "{}")
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return None
        return {"name": function.get("name"), "arguments": arguments}
    content = message.get("content")
    if not isinstance(content, str) or not content:
        return None
    bracket_call = re.fullmatch(r'\["([^"]+)"\]\((\{.*\})\)', content)
    if bracket_call:
        try:
            return {
                "name": bracket_call.group(1),
                "arguments": json.loads(bracket_call.group(2)),
            }
        except json.JSONDecodeError:
            return None
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        return None
    if not isinstance(parsed, dict):
        return None
    function = parsed.get("function", parsed.get("name"))
    if isinstance(function, dict):
        name = function.get("name")
        arguments = function.get("arguments", function.get("parameters"))
    else:
        name = function
        arguments = parsed.get(
            "params",
            parsed.get(
                "parameters",
                parsed.get("arguments", parsed.get("function_args")),
            ),
        )
        if (
            arguments is None
            and isinstance(parsed.get("parameter"), str)
            and "parameter_value" in parsed
        ):
            # DeepSeek may serialize a one-argument native call as a compact
            # name/value envelope. It is semantically the same function call.
            arguments = {parsed["parameter"]: parsed["parameter_value"]}
        if arguments is None and isinstance(name, str):
            arguments = {
                key: value
                for key, value in parsed.items()
                if key not in {"function", "name", "type"}
            }
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            return None
    if not isinstance(name, str) or not isinstance(arguments, dict):
        return None
    return {"name": name, "arguments": arguments}


def post_json(
    url: str, body: dict[str, Any], timeout: float
) -> tuple[dict[str, Any], float]:
    request = urllib.request.Request(
        url,
        data=json.dumps(body, separators=(",", ":")).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.load(response)
    if not isinstance(result, dict):
        raise RuntimeError("model response is not a JSON object")
    return result, (time.perf_counter() - started) * 1000.0


def get_json(url: str, timeout: float) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        result = json.load(response)
    if not isinstance(result, dict):
        raise RuntimeError(f"expected a JSON object from {url}")
    return result


def props_url(completion_url: str) -> str:
    parsed = urlsplit(completion_url)
    return urlunsplit((parsed.scheme, parsed.netloc, "/props", "", ""))


def observation(result: dict[str, Any], wall_ms: float) -> dict[str, Any]:
    call = normalize_tool_call(result)
    usage = result.get("usage") or {}
    timings = usage.get("timings") or {}
    message = result.get("choices", [{}])[0].get("message", {})
    content = message.get("content") or ""
    canonical_call = json.dumps(call, sort_keys=True, separators=(",", ":"))
    return {
        "request_wall_ms": wall_ms,
        "model_compute_ms": float(timings.get("prefill_ms", 0.0))
        + float(timings.get("decode_ms", 0.0)),
        "prefill_ms": float(timings.get("prefill_ms", 0.0)),
        "decode_ms": float(timings.get("decode_ms", 0.0)),
        "decode_tokens_per_sec": float(
            timings.get("decode_tokens_per_sec", 0.0)
        ),
        "completion_tokens": int(usage.get("completion_tokens", 0)),
        "accept_rate": float(usage.get("accept_rate", 0.0)),
        "tool_call": call,
        "tool_call_sha256": hashlib.sha256(canonical_call.encode()).hexdigest(),
        "assistant_content_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "speculation": result.get("dflash_tool_speculation"),
    }


def post_model(
    url: str,
    arguments: dict[str, int],
    max_tokens: int,
    timeout: float,
    *,
    prediction: dict[str, int] | None = None,
    automatic_prediction: bool = False,
    tool_choice: str | None = None,
) -> dict[str, Any]:
    result, wall_ms = post_json(
        url,
        request_body(
            arguments,
            max_tokens,
            prediction=prediction,
            automatic_prediction=automatic_prediction,
            tool_choice=tool_choice,
        ),
        timeout,
    )
    return observation(result, wall_ms)


def executor_request(
    arguments: dict[str, int], cpus: list[int], request_id: str
) -> dict[str, Any]:
    return {
        "protocol": PROTOCOL,
        "request_id": request_id,
        "mode": "authoritative-benchmark",
        "resource_percentage": 100,
        "accelerator_relation": "non_accelerator",
        "cpu_affinity": cpus,
        "cpu_affinity_isolated": True,
        "call": {"name": TOOL_NAME, "arguments": arguments},
    }


def start_executor(
    binary: Path,
    arguments: dict[str, int],
    cpus: list[int],
    request_id: str,
) -> dict[str, Any]:
    environment = os.environ.copy()
    environment["DFLASH_TOOL_SPECULATION_CPU_AFFINITY"] = compact_cpu_list(cpus)

    def pin_child() -> None:
        os.sched_setaffinity(0, set(cpus))

    started = time.perf_counter()
    process = subprocess.Popen(
        [str(binary), "--dflash-tool-spec-v1"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
        preexec_fn=pin_child,
    )
    assert process.stdin is not None
    process.stdin.write(
        json.dumps(
            executor_request(arguments, cpus, request_id),
            separators=(",", ":"),
        )
        + "\n"
    )
    process.stdin.close()
    process.stdin = None
    return {"process": process, "started": started}


def finish_executor(handle: dict[str, Any], timeout: float) -> dict[str, Any]:
    process: subprocess.Popen[str] = handle["process"]
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        stdout, stderr = process.communicate(timeout=5)
        raise RuntimeError("CPU executor timed out")
    wall_ms = (time.perf_counter() - float(handle["started"])) * 1000.0
    if process.returncode != 0:
        raise RuntimeError(
            f"CPU executor exited {process.returncode}: {stderr.strip()}"
        )
    try:
        envelope = json.loads(stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError(f"CPU executor returned invalid JSON: {stdout!r}") from error
    if not isinstance(envelope, dict) or not envelope.get("ok"):
        raise RuntimeError(f"CPU executor rejected request: {envelope!r}")
    result = envelope.get("result")
    if not isinstance(result, dict):
        raise RuntimeError("CPU executor result is not an object")
    return {"wall_ms": wall_ms, "result": result}


def stop_executor(handle: dict[str, Any]) -> None:
    process: subprocess.Popen[str] = handle["process"]
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5.0)


def run_executor(
    binary: Path,
    arguments: dict[str, int],
    cpus: list[int],
    timeout: float,
    request_id: str,
) -> dict[str, Any]:
    return finish_executor(
        start_executor(binary, arguments, cpus, request_id), timeout
    )


def expected_call(arguments: dict[str, int]) -> dict[str, Any]:
    return {"name": TOOL_NAME, "arguments": arguments}


def validate_model_call(row: dict[str, Any], arguments: dict[str, int]) -> None:
    expected = expected_call(arguments)
    if row["tool_call"] != expected:
        raise RuntimeError(
            f"model emitted {row['tool_call']!r}, expected {expected!r}"
        )


def percentile(values: Iterable[float], quantile: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("cannot take percentile of an empty sequence")
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def bootstrap_speedup_ci(
    pairs: list[dict[str, Any]], resamples: int, seed: int
) -> list[float]:
    generator = random.Random(seed)
    ratios = []
    for _ in range(resamples):
        sample = [pairs[generator.randrange(len(pairs))] for _ in pairs]
        control = statistics.median(
            float(pair["control"]["task_ms"]) for pair in sample
        )
        speculative = statistics.median(
            float(pair["speculative"]["task_ms"]) for pair in sample
        )
        ratios.append(control / speculative)
    return [percentile(ratios, 0.025), percentile(ratios, 0.975)]


def read_process_affinity(pid: int) -> list[int]:
    return sorted(os.sched_getaffinity(pid))


def calibrate(
    args: argparse.Namespace,
) -> tuple[dict[str, int], dict[str, Any]]:
    probe_arguments = expected_arguments(
        args.rows,
        args.nonzeros_per_row,
        max(1, args.initial_iterations),
        args.threads,
        args.tool_seed,
    )
    model_samples = []
    for _ in range(args.calibration_model_samples):
        row = post_model(
            args.url, probe_arguments, args.max_tokens, args.timeout
        )
        validate_model_call(row, probe_arguments)
        model_samples.append(row)
    target_ms = statistics.median(
        float(row["request_wall_ms"]) for row in model_samples
    )

    iterations = args.initial_iterations
    calibration_steps = []
    for step in range(args.calibration_steps):
        arguments = expected_arguments(
            args.rows,
            args.nonzeros_per_row,
            iterations,
            args.threads,
            args.tool_seed,
        )
        samples = [
            run_executor(
                args.binary,
                arguments,
                args.tool_cpus,
                args.timeout,
                f"calibrate-{step}-{sample}",
            )
            for sample in range(args.calibration_tool_samples)
        ]
        observed_ms = statistics.median(
            float(sample["wall_ms"]) for sample in samples
        )
        calibration_steps.append(
            {
                "iterations": iterations,
                "tool_wall_p50_ms": observed_ms,
                "samples": samples,
            }
        )
        if observed_ms <= 0:
            raise RuntimeError("CPU executor calibration returned zero time")
        ratio = target_ms / observed_ms
        if 0.97 <= ratio <= 1.03:
            break
        iterations = max(1, round(iterations * ratio))

    final_arguments = expected_arguments(
        args.rows,
        args.nonzeros_per_row,
        iterations,
        args.threads,
        args.tool_seed,
    )
    return final_arguments, {
        "target_model_request_p50_ms": target_ms,
        "model_samples": model_samples,
        "steps": calibration_steps,
        "selected_iterations": iterations,
    }


def run_direct_control(
    args: argparse.Namespace,
    arguments: dict[str, int],
    label: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    model = post_model(args.url, arguments, args.max_tokens, args.timeout)
    tool = run_executor(
        args.binary, arguments, args.tool_cpus, args.timeout, f"{label}-tool"
    )
    validate_model_call(model, arguments)
    return {
        "mode": "control",
        "task_ms": (time.perf_counter() - started) * 1000.0,
        "model": model,
        "tool": tool,
    }


def run_direct_overlap(
    args: argparse.Namespace,
    arguments: dict[str, int],
    label: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    handle = start_executor(args.binary, arguments, args.tool_cpus, label)
    try:
        model = post_model(args.url, arguments, args.max_tokens, args.timeout)
        tool = finish_executor(handle, args.timeout)
    except BaseException:
        stop_executor(handle)
        raise
    validate_model_call(model, arguments)
    return {
        "mode": "speculative",
        "task_ms": (time.perf_counter() - started) * 1000.0,
        "model": model,
        "tool": tool,
    }


def run_direct_miss(
    args: argparse.Namespace,
    arguments: dict[str, int],
    label: str,
) -> dict[str, Any]:
    wrong = dict(arguments)
    wrong["iterations"] = max(1, arguments["iterations"] - 1)
    if wrong["iterations"] == arguments["iterations"]:
        wrong["iterations"] += 1
    started = time.perf_counter()
    private = start_executor(args.binary, wrong, args.tool_cpus, f"{label}-wrong")
    model = post_model(args.url, arguments, args.max_tokens, args.timeout)
    stop_executor(private)
    authoritative = run_executor(
        args.binary,
        arguments,
        args.tool_cpus,
        args.timeout,
        f"{label}-authoritative",
    )
    validate_model_call(model, arguments)
    return {
        "mode": "miss",
        "task_ms": (time.perf_counter() - started) * 1000.0,
        "model": model,
        "authoritative_tool": authoritative,
        "private_result_exposed": False,
    }


def qualify(args: argparse.Namespace) -> None:
    model_affinity = read_process_affinity(args.model_pid)
    overlap = sorted(set(model_affinity).intersection(args.tool_cpus))
    if overlap:
        raise SystemExit(f"model/tool CPU affinity overlaps: {overlap}")
    if args.threads > len(args.tool_cpus):
        raise SystemExit("tool threads exceed reserved logical CPUs")

    arguments, calibration = calibrate(args)
    for warmup in range(args.warmups):
        run_direct_control(args, arguments, f"warmup-control-{warmup}")
        run_direct_overlap(args, arguments, f"warmup-overlap-{warmup}")

    generator = random.Random(args.seed)
    pairs = []
    for pair_index in range(args.pairs):
        order = ["control", "speculative"]
        generator.shuffle(order)
        rows: dict[str, dict[str, Any]] = {}
        for arm in order:
            rows[arm] = (
                run_direct_control(
                    args, arguments, f"pair-{pair_index}-control"
                )
                if arm == "control"
                else run_direct_overlap(
                    args, arguments, f"pair-{pair_index}-speculative"
                )
            )
        pairs.append({"pair_index": pair_index, "arm_order": order, **rows})
        print(
            json.dumps(
                {
                    "phase": "qualify",
                    "pair": pair_index + 1,
                    "control_ms": round(rows["control"]["task_ms"], 3),
                    "overlap_ms": round(rows["speculative"]["task_ms"], 3),
                    "speedup": round(
                        rows["control"]["task_ms"]
                        / rows["speculative"]["task_ms"],
                        3,
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    misses = [
        run_direct_miss(args, arguments, f"miss-{index}")
        for index in range(args.miss_samples)
    ]
    controls = [pair["control"] for pair in pairs]
    speculative = [pair["speculative"] for pair in pairs]
    control_task = statistics.median(row["task_ms"] for row in controls)
    speculative_task = statistics.median(
        row["task_ms"] for row in speculative
    )
    control_model = statistics.median(
        row["model"]["model_compute_ms"] for row in controls
    )
    speculative_model = statistics.median(
        row["model"]["model_compute_ms"] for row in speculative
    )
    miss_model = statistics.median(
        row["model"]["model_compute_ms"] for row in misses
    )
    slowdown_percent = 100.0 * (
        max(speculative_model, miss_model) / control_model - 1.0
    )
    expected_checksum = controls[0]["tool"]["result"]["checksum"]
    canonical_model_identity = {
        (
            row["model"]["tool_call_sha256"],
            row["model"]["assistant_content_sha256"],
            row["model"]["completion_tokens"],
        )
        for row in controls + speculative + misses
    }
    all_tools_equal = all(
        row["tool"]["result"]["checksum"] == expected_checksum
        and row["tool"]["result"]["cpu_affinity"] == args.tool_cpus
        for row in controls + speculative
    ) and all(
        row["authoritative_tool"]["result"]["checksum"] == expected_checksum
        for row in misses
    )
    ds4_active = all(
        row["model"]["accept_rate"] > 0
        for row in controls + speculative + misses
    )
    speedup = control_task / speculative_task
    checks = {
        "disjoint_cpu_affinity": not overlap,
        "identical_model_outputs": len(canonical_model_identity) == 1,
        "identical_tool_outputs": all_tools_equal,
        "ds4_active": ds4_active,
        "model_slowdown": slowdown_percent <= args.max_model_slowdown_percent,
        "direct_speedup": speedup >= args.min_qualification_speedup,
        "private_miss_result_hidden": all(
            not row["private_result_exposed"] for row in misses
        ),
    }
    passed = all(checks.values())
    profile = {
        "profile_status": "qualified" if passed else "rejected",
        "executor": "child_process_cpu_affinity",
        "profile_kind": "disjoint_strix_cpu_sparse_compute",
        "qualification": {
            "host": "lucebox5",
            "model_cpu_affinity": model_affinity,
            "tool_cpu_affinity": args.tool_cpus,
            "checks": checks,
        },
        "path_summary": {
            "100": {
                "accelerator_relation": "non_accelerator",
                "decode_interference_qualified": passed,
                "hit": {
                    "control_task_mean_ms": statistics.fmean(
                        row["task_ms"] for row in controls
                    ),
                    "speculative_task_mean_ms": statistics.fmean(
                        row["task_ms"] for row in speculative
                    ),
                    "model_slowdown_percent": slowdown_percent,
                },
                "miss": {
                    "control_task_mean_ms": statistics.fmean(
                        row["task_ms"] for row in controls
                    ),
                    "speculative_task_mean_ms": statistics.fmean(
                        row["task_ms"] for row in misses
                    ),
                    "model_slowdown_percent": slowdown_percent,
                },
            }
        },
    }
    summary = {
        "pairs": len(pairs),
        "control_task_p50_ms": control_task,
        "overlap_task_p50_ms": speculative_task,
        "direct_exact_hit_speedup": speedup,
        "control_model_compute_p50_ms": control_model,
        "overlap_model_compute_p50_ms": speculative_model,
        "model_compute_slowdown_percent": slowdown_percent,
        "control_tool_wall_p50_ms": statistics.median(
            row["tool"]["wall_ms"] for row in controls
        ),
        "overlap_tool_wall_p50_ms": statistics.median(
            row["tool"]["wall_ms"] for row in speculative
        ),
        "miss_task_p50_ms": statistics.median(
            row["task_ms"] for row in misses
        ),
        "median_accept_rate": statistics.median(
            row["model"]["accept_rate"]
            for row in controls + speculative + misses
        ),
        "checks": checks,
        "passed": passed,
    }
    report = {
        "phase": "qualification",
        "host": "lucebox5",
        "config": report_config(args, arguments),
        "model_pid": args.model_pid,
        "model_cpu_affinity": model_affinity,
        "tool_cpu_affinity": args.tool_cpus,
        "calibration": calibration,
        "summary": summary,
        "profile": profile,
        "pairs": pairs,
        "misses": misses,
    }
    write_report(args.output, report)
    if passed:
        write_report(args.profile_output, profile)
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    if not passed:
        raise SystemExit("CPU-lane qualification failed")


def native_control(
    args: argparse.Namespace,
    arguments: dict[str, int],
    label: str,
) -> dict[str, Any]:
    direct = run_direct_control(args, arguments, label)
    tool_result = direct["tool"]["result"]
    return {
        "mode": "control",
        "task_ms": direct["task_ms"],
        **direct["model"],
        "tool_wall_ms": direct["tool"]["wall_ms"],
        "tool_compute_ms": float(tool_result["compute_ms"]),
        "tool_checksum": tool_result["checksum"],
        "tool_cpu_affinity": tool_result["cpu_affinity"],
    }


def native_speculative(
    args: argparse.Namespace,
    arguments: dict[str, int],
    *,
    prediction: dict[str, int] | None = None,
) -> dict[str, Any]:
    row = post_model(
        args.url,
        arguments,
        args.max_tokens,
        args.timeout,
        prediction=prediction or arguments,
    )
    validate_model_call(row, arguments)
    metadata = row["speculation"] if isinstance(row["speculation"], dict) else {}
    tool_result = metadata.get("result", {})
    return {
        "mode": "speculative",
        "task_ms": row["request_wall_ms"],
        **row,
        "tool_wall_ms": float(metadata.get("executor_wall_ms", math.nan)),
        "tool_compute_ms": float(tool_result.get("compute_ms", math.nan)),
        "tool_checksum": tool_result.get("checksum"),
        "tool_cpu_affinity": tool_result.get("cpu_affinity"),
    }


def native_qwen_control(
    args: argparse.Namespace,
    arguments: dict[str, int],
    label: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    model = post_model(
        args.url,
        arguments,
        args.max_tokens,
        args.timeout,
        automatic_prediction=False,
        tool_choice="required",
    )
    tool = run_executor(
        args.binary,
        arguments,
        args.tool_cpus,
        args.timeout,
        f"{label}-tool",
    )
    validate_model_call(model, arguments)
    result = tool["result"]
    return {
        "mode": "control",
        "task_ms": (time.perf_counter() - started) * 1000.0,
        **model,
        "tool_wall_ms": float(tool["wall_ms"]),
        "tool_compute_ms": float(result["compute_ms"]),
        "tool_checksum": result["checksum"],
        "tool_cpu_affinity": result["cpu_affinity"],
        "prediction_hit": False,
        "predictor_wall_ms": 0.0,
    }


def native_qwen_speculative(
    args: argparse.Namespace,
    arguments: dict[str, int],
    label: str,
) -> dict[str, Any]:
    started = time.perf_counter()
    model = post_model(
        args.url,
        arguments,
        args.max_tokens,
        args.timeout,
        automatic_prediction=True,
        tool_choice="required",
    )
    validate_model_call(model, arguments)
    metadata = model["speculation"] if isinstance(
        model["speculation"], dict
    ) else {}
    prediction_hit = metadata.get("status") == "hit"
    if prediction_hit:
        tool_result = metadata.get("result")
        if not isinstance(tool_result, dict):
            raise RuntimeError("automatic hit did not expose a tool result")
        tool_wall_ms = float(metadata.get("executor_wall_ms", math.nan))
    else:
        # This is the real miss path: discard the private speculative result,
        # then execute the authoritative model call normally.
        fallback = run_executor(
            args.binary,
            arguments,
            args.tool_cpus,
            args.timeout,
            f"{label}-fallback",
        )
        tool_result = fallback["result"]
        tool_wall_ms = float(fallback["wall_ms"])
    return {
        "mode": "qwen_speculative",
        "task_ms": (time.perf_counter() - started) * 1000.0,
        **model,
        "tool_wall_ms": tool_wall_ms,
        "tool_compute_ms": float(tool_result["compute_ms"]),
        "tool_checksum": tool_result["checksum"],
        "tool_cpu_affinity": tool_result["cpu_affinity"],
        "prediction_hit": prediction_hit,
        "predictor_wall_ms": float(metadata.get("predictor_wall_ms", 0.0)),
        "prediction_source": metadata.get("prediction_source"),
        "prediction_status": metadata.get("status"),
        "prediction_reason": metadata.get("reason"),
    }


def summarize_native(
    pairs: list[dict[str, Any]], resamples: int, seed: int
) -> dict[str, Any]:
    controls = [pair["control"] for pair in pairs]
    speculative = [pair["speculative"] for pair in pairs]
    control_task = statistics.median(row["task_ms"] for row in controls)
    speculative_task = statistics.median(
        row["task_ms"] for row in speculative
    )
    control_model = statistics.median(
        row["model_compute_ms"] for row in controls
    )
    speculative_model = statistics.median(
        row["model_compute_ms"] for row in speculative
    )
    control_tool = statistics.median(
        row["tool_compute_ms"] for row in controls
    )
    speculative_tool = statistics.median(
        row["tool_compute_ms"] for row in speculative
    )
    paired_speedups = [
        float(pair["control"]["task_ms"])
        / float(pair["speculative"]["task_ms"])
        for pair in pairs
    ]
    return {
        "pairs": len(pairs),
        "control_task_p50_ms": control_task,
        "control_task_p95_ms": percentile(
            (row["task_ms"] for row in controls), 0.95
        ),
        "control_task_max_ms": max(row["task_ms"] for row in controls),
        "speculative_task_p50_ms": speculative_task,
        "speculative_task_p95_ms": percentile(
            (row["task_ms"] for row in speculative), 0.95
        ),
        "speculative_task_max_ms": max(
            row["task_ms"] for row in speculative
        ),
        "exact_hit_speedup": control_task / speculative_task,
        "paired_speedup_p05": percentile(paired_speedups, 0.05),
        "paired_speedup_min": min(paired_speedups),
        "exact_hit_speedup_bootstrap_95ci": bootstrap_speedup_ci(
            pairs, resamples, seed
        ),
        "task_latency_reduction_percent": 100.0
        * (control_task - speculative_task)
        / control_task,
        "control_model_compute_p50_ms": control_model,
        "speculative_model_compute_p50_ms": speculative_model,
        "model_compute_slowdown_percent": 100.0
        * (speculative_model / control_model - 1.0),
        "control_tool_compute_p50_ms": control_tool,
        "speculative_tool_compute_p50_ms": speculative_tool,
        "tool_compute_slowdown_percent": 100.0
        * (speculative_tool / control_tool - 1.0),
        "latency_match_ratio": min(control_model, control_tool)
        / max(control_model, control_tool),
        "ideal_zero_interference_speedup_ceiling": (
            control_model + control_tool
        )
        / max(control_model, control_tool),
        "median_decode_tokens_per_sec": statistics.median(
            row["decode_tokens_per_sec"] for row in controls + speculative
        ),
        "median_accept_rate": statistics.median(
            row["accept_rate"] for row in controls + speculative
        ),
        "native_hits": sum(
            isinstance(row["speculation"], dict)
            and row["speculation"].get("status") == "hit"
            for row in speculative
        ),
        "all_calls_identical": all(
            pair["control"]["tool_call_sha256"]
            == pair["speculative"]["tool_call_sha256"]
            for pair in pairs
        ),
        "all_model_outputs_identical": all(
            pair["control"]["assistant_content_sha256"]
            == pair["speculative"]["assistant_content_sha256"]
            and pair["control"]["completion_tokens"]
            == pair["speculative"]["completion_tokens"]
            for pair in pairs
        ),
        "all_tool_outputs_equivalent": all(
            pair["control"]["tool_checksum"]
            == pair["speculative"]["tool_checksum"]
            and pair["control"]["tool_cpu_affinity"]
            == pair["speculative"]["tool_cpu_affinity"]
            for pair in pairs
        ),
    }


def native(args: argparse.Namespace) -> None:
    props = get_json(props_url(args.url), args.timeout)
    tool_props = props.get("tool_speculation")
    if not isinstance(tool_props, dict) or not tool_props.get("enabled"):
        raise SystemExit("server tool speculation is not enabled")
    expected_props = {
        "execution_mode": "child_process_cpu_affinity",
        "profile_status": "qualified",
        "compute_isolation": "disjoint_cpu_affinity",
        "cpu_affinity_isolated": True,
        "preserves_token_speculation": True,
    }
    for key, expected in expected_props.items():
        if tool_props.get(key) != expected:
            raise SystemExit(
                f"server tool_speculation.{key}={tool_props.get(key)!r}, "
                f"expected {expected!r}"
            )
    if tool_props.get("tool_cpu_affinity") != args.tool_cpus:
        raise SystemExit("server tool CPU affinity differs from benchmark")
    model_affinity = tool_props.get("model_cpu_affinity")
    if not isinstance(model_affinity, list) or set(model_affinity) & set(
        args.tool_cpus
    ):
        raise SystemExit("server model/tool CPU affinity is not disjoint")

    arguments = expected_arguments(
        args.rows,
        args.nonzeros_per_row,
        args.iterations,
        args.threads,
        args.tool_seed,
    )
    for warmup in range(args.warmups):
        native_control(args, arguments, f"native-warm-control-{warmup}")
        row = native_speculative(args, arguments)
        if (row["speculation"] or {}).get("status") != "hit":
            raise RuntimeError("native speculative warmup did not commit")

    generator = random.Random(args.seed)
    pairs = []
    for pair_index in range(args.pairs):
        order = ["control", "speculative"]
        generator.shuffle(order)
        rows: dict[str, dict[str, Any]] = {}
        for arm in order:
            rows[arm] = (
                native_control(
                    args, arguments, f"native-pair-{pair_index}-control"
                )
                if arm == "control"
                else native_speculative(args, arguments)
            )
        pairs.append({"pair_index": pair_index, "arm_order": order, **rows})
        print(
            json.dumps(
                {
                    "phase": "native",
                    "pair": pair_index + 1,
                    "control_ms": round(rows["control"]["task_ms"], 3),
                    "speculative_ms": round(
                        rows["speculative"]["task_ms"], 3
                    ),
                    "speedup": round(
                        rows["control"]["task_ms"]
                        / rows["speculative"]["task_ms"],
                        3,
                    ),
                    "status": (rows["speculative"]["speculation"] or {}).get(
                        "status"
                    ),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    wrong = dict(arguments)
    wrong["iterations"] = max(1, arguments["iterations"] - 1)
    if wrong["iterations"] == arguments["iterations"]:
        wrong["iterations"] += 1
    miss = native_speculative(args, arguments, prediction=wrong)
    miss_metadata = miss["speculation"] or {}
    miss_check = {
        "passed": miss_metadata.get("status") == "miss"
        and miss_metadata.get("reason") == "invocation_mismatch"
        and "result" not in miss_metadata
        and all(
            miss["assistant_content_sha256"]
            == pair["control"]["assistant_content_sha256"]
            and miss["completion_tokens"]
            == pair["control"]["completion_tokens"]
            for pair in pairs
        ),
        "status": miss_metadata.get("status"),
        "reason": miss_metadata.get("reason"),
        "private_result_exposed": "result" in miss_metadata,
    }
    summary = summarize_native(pairs, args.bootstrap_resamples, args.seed)
    correctness_passed = (
        summary["native_hits"] == args.pairs
        and summary["all_calls_identical"]
        and summary["all_model_outputs_identical"]
        and summary["all_tool_outputs_equivalent"]
        and miss_check["passed"]
        and all(
            pair[arm]["accept_rate"] > 0
            for pair in pairs
            for arm in ("control", "speculative")
        )
    )
    ci_low = summary["exact_hit_speedup_bootstrap_95ci"][0]
    checks = {
        "correctness": correctness_passed,
        "strong_sequential_baseline": True,
        "exact_hit_speedup": summary["exact_hit_speedup"] >= args.min_speedup,
        "speedup_ci_low": ci_low >= args.min_speedup_ci_low,
        "model_slowdown": summary["model_compute_slowdown_percent"]
        <= args.max_model_slowdown_percent,
    }
    production_gate = {
        "passed": all(checks.values()),
        "checks": checks,
        "thresholds": {
            "min_exact_hit_speedup": args.min_speedup,
            "min_speedup_ci_low": args.min_speedup_ci_low,
            "max_model_slowdown_percent": args.max_model_slowdown_percent,
        },
    }
    report = {
        "phase": "native_engine",
        "host": "lucebox5",
        "config": report_config(args, arguments),
        "server_snapshot": {
            "runtime": props.get("runtime"),
            "speculative": props.get("speculative"),
            "tool_speculation": tool_props,
        },
        "methodology": {
            "control": "model request followed by the identical CPU-pinned sparse tool",
            "speculative": "engine starts the identical CPU-pinned sparse tool before DS4 generation",
            "commit": "result exposed only after exact canonical call match",
            "pairing": "randomized arm order within every warm pair",
        },
        "correctness_passed": correctness_passed,
        "production_gate": production_gate,
        "miss_check": miss_check,
        "summary": summary,
        "pairs": pairs,
    }
    write_report(args.output, report)
    print(
        json.dumps(
            {
                "correctness_passed": correctness_passed,
                "production_gate": production_gate,
                "miss_check": miss_check,
                "summary": summary,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    if not production_gate["passed"]:
        raise SystemExit("native CPU tool-speculation production gate failed")


def native_qwen(args: argparse.Namespace) -> None:
    props = get_json(props_url(args.url), args.timeout)
    tool_props = props.get("tool_speculation")
    if not isinstance(tool_props, dict) or not tool_props.get("enabled"):
        raise SystemExit("server tool speculation is not enabled")
    if not tool_props.get("automatic_prediction_enabled"):
        raise SystemExit("server automatic Qwen prediction is not enabled")
    if tool_props.get("execution_mode") != "child_process_cpu_affinity":
        raise SystemExit("automatic benchmark requires the isolated CPU executor")
    if tool_props.get("tool_cpu_affinity") != args.tool_cpus:
        raise SystemExit("server tool CPU affinity differs from benchmark")

    arguments = expected_arguments(
        args.rows,
        args.nonzeros_per_row,
        args.iterations,
        args.threads,
        args.tool_seed,
    )
    for warmup in range(args.warmups):
        native_qwen_control(
            args, arguments, f"qwen-warm-{warmup}-control"
        )
        native_qwen_speculative(
            args, arguments, f"qwen-warm-{warmup}-speculative"
        )

    generator = random.Random(args.seed)
    pairs: list[dict[str, Any]] = []
    for pair_index in range(args.pairs):
        order = ["control", "speculative"]
        generator.shuffle(order)
        rows: dict[str, dict[str, Any]] = {}
        for arm in order:
            rows[arm] = (
                native_qwen_control(
                    args, arguments, f"qwen-{pair_index}-control"
                )
                if arm == "control"
                else native_qwen_speculative(
                    args, arguments, f"qwen-{pair_index}-speculative"
                )
            )
        pairs.append({"pair_index": pair_index, "arm_order": order, **rows})
        print(json.dumps({
            "phase": "native-qwen",
            "pair": pair_index + 1,
            "control_ms": round(rows["control"]["task_ms"], 3),
            "speculative_ms": round(rows["speculative"]["task_ms"], 3),
            "speedup": round(
                rows["control"]["task_ms"] /
                rows["speculative"]["task_ms"], 3),
            "prediction_status": rows["speculative"]["prediction_status"],
            "predictor_ms": round(
                rows["speculative"]["predictor_wall_ms"], 3),
        }, sort_keys=True), flush=True)

    summary = summarize_native(pairs, args.bootstrap_resamples, args.seed)
    speculative = [pair["speculative"] for pair in pairs]
    hits = sum(row["prediction_hit"] for row in speculative)
    summary.update({
        "qwen_prediction_hits": hits,
        "qwen_prediction_hit_rate": hits / len(speculative),
        "qwen_predictor_p50_ms": statistics.median(
            row["predictor_wall_ms"] for row in speculative
        ),
        "qwen_predictor_p95_ms": percentile(
            (row["predictor_wall_ms"] for row in speculative), 0.95
        ),
        "qwen_prediction_source_valid": all(
            row["prediction_source"] == NATIVE_PREDICTION_SOURCE
            for row in speculative
        ),
    })
    checks = {
        "all_model_outputs_identical": summary["all_model_outputs_identical"],
        "all_calls_identical": summary["all_calls_identical"],
        "all_tool_outputs_equivalent": summary["all_tool_outputs_equivalent"],
        "ds4_active": summary["median_accept_rate"] > 0,
        "prediction_source": summary["qwen_prediction_source_valid"],
        "prediction_hit_rate":
            summary["qwen_prediction_hit_rate"] >= args.min_prediction_hit_rate,
        "speedup": summary["exact_hit_speedup"] >= args.min_speedup,
        "speedup_ci_low":
            summary["exact_hit_speedup_bootstrap_95ci"][0] >=
            args.min_speedup_ci_low,
        "speedup_p05":
            summary["paired_speedup_p05"] >= args.min_speedup_p05,
        "model_slowdown": summary["model_compute_slowdown_percent"] <=
            args.max_model_slowdown_percent,
    }
    report = {
        "phase": "native_qwen_engine",
        "host": "lucebox5",
        "config": report_config(args, arguments),
        "methodology": {
            "control": "DS4 generation, then the authoritative CPU-pinned tool",
            "speculative": "Qwen3-0.6B predicts the call; the engine launches the private CPU-pinned tool before DS4 finishes",
            "prediction_cost": "included in speculative request wall time",
            "miss_cost": "included by running the authoritative tool after every miss",
            "commit": "exact canonical function name and arguments",
            "semantic_token_injection": False,
        },
        "server_snapshot": {"tool_speculation": tool_props},
        "production_gate": {
            "passed": all(checks.values()),
            "checks": checks,
            "thresholds": {
                "min_speedup": args.min_speedup,
                "min_speedup_ci_low": args.min_speedup_ci_low,
                "min_speedup_p05": args.min_speedup_p05,
                "min_prediction_hit_rate": args.min_prediction_hit_rate,
                "max_model_slowdown_percent":
                    args.max_model_slowdown_percent,
            },
        },
        "summary": summary,
        "pairs": pairs,
    }
    write_report(args.output, report)
    print(json.dumps({
        "production_gate": report["production_gate"],
        "summary": summary,
    }, indent=2, sort_keys=True), flush=True)
    if not report["production_gate"]["passed"]:
        raise SystemExit("automatic Qwen tool-speculation gate failed")


def report_config(
    args: argparse.Namespace, arguments: dict[str, int]
) -> dict[str, Any]:
    return {
        "url": args.url,
        "binary": str(args.binary.resolve()),
        "tool_arguments": arguments,
        "fixed_sparse_shape": {
            "rows": args.rows,
            "nonzeros_per_row": args.nonzeros_per_row,
            "threads": args.threads,
            "seed": args.tool_seed,
        },
        "tool_cpus": args.tool_cpus,
        "max_tokens": args.max_tokens,
        "pairs": args.pairs,
        "warmups": args.warmups,
        "seed": args.seed,
    }


def write_report(path: Path | None, value: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--url", default="http://127.0.0.1:18145/v1/chat/completions"
    )
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--tool-cpus", type=parse_cpu_list, required=True)
    parser.add_argument("--rows", type=int, default=4096)
    parser.add_argument("--nonzeros-per-row", type=int, default=16)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--tool-seed", type=int, default=731)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--pairs", type=int, default=20)
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--seed", type=int, default=814)
    parser.add_argument("--max-model-slowdown-percent", type=float, default=5.0)
    parser.add_argument("--output", type=Path)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="phase", required=True)

    qualify_parser = subparsers.add_parser("qualify")
    add_common_arguments(qualify_parser)
    qualify_parser.add_argument("--model-pid", type=int, required=True)
    qualify_parser.add_argument("--initial-iterations", type=int, default=20)
    qualify_parser.add_argument("--calibration-steps", type=int, default=5)
    qualify_parser.add_argument("--calibration-model-samples", type=int, default=3)
    qualify_parser.add_argument("--calibration-tool-samples", type=int, default=2)
    qualify_parser.add_argument("--miss-samples", type=int, default=5)
    qualify_parser.add_argument("--min-qualification-speedup", type=float, default=1.70)
    qualify_parser.add_argument("--profile-output", type=Path, required=True)

    native_parser = subparsers.add_parser("native")
    add_common_arguments(native_parser)
    native_parser.add_argument("--iterations", type=int, required=True)
    native_parser.add_argument("--bootstrap-resamples", type=int, default=20_000)
    native_parser.add_argument("--min-speedup", type=float, default=1.80)
    native_parser.add_argument("--min-speedup-ci-low", type=float, default=1.70)

    qwen_parser = subparsers.add_parser("native-qwen")
    add_common_arguments(qwen_parser)
    qwen_parser.add_argument("--iterations", type=int, required=True)
    qwen_parser.add_argument("--bootstrap-resamples", type=int, default=20_000)
    qwen_parser.add_argument("--min-speedup", type=float, default=1.60)
    qwen_parser.add_argument("--min-speedup-ci-low", type=float, default=1.50)
    qwen_parser.add_argument("--min-speedup-p05", type=float, default=1.25)
    qwen_parser.add_argument("--min-prediction-hit-rate", type=float, default=0.90)

    args = parser.parse_args()
    if not args.binary.is_file():
        parser.error(f"executor binary does not exist: {args.binary}")
    positive = [
        args.rows,
        args.nonzeros_per_row,
        args.threads,
        args.max_tokens,
        args.pairs,
        args.warmups,
        args.timeout,
    ]
    if any(value <= 0 for value in positive):
        parser.error("counts and timeouts must be positive")
    if args.tool_seed < 0 or args.max_model_slowdown_percent < 0:
        parser.error("seed and slowdown threshold must be non-negative")
    if args.phase == "qualify":
        if (
            args.model_pid <= 0
            or args.initial_iterations <= 0
            or args.calibration_steps <= 0
            or args.calibration_model_samples <= 0
            or args.calibration_tool_samples <= 0
            or args.miss_samples <= 0
            or args.min_qualification_speedup <= 1.0
        ):
            parser.error("qualification settings must be positive")
        qualify(args)
    else:
        if (
            args.iterations <= 0
            or args.bootstrap_resamples <= 0
            or args.min_speedup <= 1.0
            or args.min_speedup_ci_low <= 1.0
            or args.min_speedup_ci_low > args.min_speedup
        ):
            parser.error("native benchmark settings are invalid")
        if args.phase == "native-qwen":
            if (
                not 0.0 <= args.min_prediction_hit_rate <= 1.0
                or args.min_speedup_p05 <= 1.0
                or args.min_speedup_p05 > args.min_speedup
            ):
                parser.error("native Qwen benchmark settings are invalid")
            native_qwen(args)
        else:
            native(args)


if __name__ == "__main__":
    main()
