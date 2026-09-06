#!/usr/bin/env python3

from __future__ import annotations

import datetime as dt
import math
from pathlib import Path
from typing import Any

from harness.qualification.qwen36.profiles import QwenArProfile
from harness.qualification.qwen36.qualify import ROOT, git_commit, sha256


def _samples(report: dict[str, Any], name: str) -> list[float]:
    raw_values = [run.get(name) for case in report.get("cases", []) for run in case.get("runs", [])]
    if not raw_values:
        raise ValueError(f"generation report contains no {name} samples")
    values = []
    for raw in raw_values:
        if isinstance(raw, bool) or not isinstance(raw, int | float):
            raise ValueError(f"generation report requires positive finite {name} samples")
        value = float(raw)
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"generation report requires positive finite {name} samples")
        values.append(value)
    return values


def _validate_usage_throughput(report: dict[str, Any]) -> None:
    for case in report.get("cases", []):
        if not isinstance(case, dict):
            raise ValueError("generation report case evidence must be an object")
        for run in case.get("runs", []):
            if not isinstance(run, dict):
                raise ValueError("generation report run evidence must be an object")
            completion_tokens = run.get("completion_tokens")
            elapsed_s = run.get("elapsed_s")
            tok_s = run.get("tok_s")
            if (
                isinstance(completion_tokens, bool)
                or not isinstance(completion_tokens, int)
                or completion_tokens <= 0
            ):
                raise ValueError("generation report requires positive completion_tokens")
            if (
                isinstance(elapsed_s, bool)
                or not isinstance(elapsed_s, int | float)
                or not math.isfinite(elapsed_s)
                or elapsed_s <= 0
            ):
                raise ValueError("generation report requires positive finite elapsed_s")
            if (
                isinstance(tok_s, bool)
                or not isinstance(tok_s, int | float)
                or not math.isfinite(tok_s)
                or tok_s <= 0
            ):
                raise ValueError("generation report requires positive finite tok_s")
            expected = completion_tokens / elapsed_s
            if not math.isclose(tok_s, expected, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError("generation report tok_s differs from completion_tokens / elapsed_s")


def _prompt_path(report: dict[str, Any]) -> Path:
    value = report.get("prompts")
    if not isinstance(value, str) or not value:
        raise ValueError("generation report must identify its prompt corpus")
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def from_generation_reports(
    reports: list[dict[str, Any]],
    *,
    profile: QwenArProfile,
    target: Path,
    environment: dict[str, Any],
    qualification_subject: dict[str, Any],
) -> dict[str, Any]:
    if not reports:
        raise ValueError("at least one generation report is required")
    by_concurrency: dict[int, dict[str, Any]] = {}
    for report in reports:
        if report.get("schema_version") != 2:
            raise ValueError("AR performance requires generation report schema version 2")
        concurrency = report.get("concurrency")
        if isinstance(concurrency, bool) or not isinstance(concurrency, int) or concurrency <= 0:
            raise ValueError("generation report concurrency must be positive")
        if concurrency in by_concurrency:
            raise ValueError(f"duplicate concurrency report: {concurrency}")
        if report.get("summary", {}).get("status") != "pass":
            raise ValueError("AR performance requires passing generation reports")
        if any(
            run.get("token_count_source") != "usage"
            for case in report.get("cases", [])
            for run in case.get("runs", [])
        ):
            raise ValueError("AR performance requires server-reported token counts")
        _validate_usage_throughput(report)
        by_concurrency[concurrency] = report

    prompt_paths = {_prompt_path(report).resolve() for report in reports}
    max_tokens = {report.get("max_tokens") for report in reports}
    temperatures = {report.get("temperature") for report in reports}
    if len(prompt_paths) != 1 or len(max_tokens) != 1 or len(temperatures) != 1:
        raise ValueError("generation reports use different prompts or decoding parameters")
    prompt_path = prompt_paths.pop()
    if not prompt_path.is_file():
        raise ValueError(f"generation prompt corpus is missing: {prompt_path}")
    try:
        prompt_corpus = str(prompt_path.relative_to(ROOT))
    except ValueError:
        prompt_corpus = str(prompt_path)

    sample_plan = {workload.concurrency: workload for workload in profile.performance_workloads}
    if set(by_concurrency) != set(sample_plan):
        raise ValueError("generation report concurrency set differs from the profile")
    for concurrency, report in by_concurrency.items():
        workload = sample_plan[concurrency]
        observed = (
            report.get("warmups"),
            report.get("repeats"),
            report.get("max_tokens"),
            report.get("temperature"),
        )
        integer_fields = observed[:3]
        temperature = observed[3]
        if (
            any(isinstance(value, bool) or not isinstance(value, int) for value in integer_fields)
            or isinstance(temperature, bool)
            or not isinstance(temperature, int | float)
            or not math.isfinite(temperature)
        ):
            raise ValueError(f"generation report sample plan has invalid types at c{concurrency}")
        expected = (
            workload.warmups,
            workload.repetitions,
            workload.maximum_tokens,
            workload.temperature,
        )
        if observed != expected:
            raise ValueError(f"generation report sample plan differs at c{concurrency}")
    metrics: dict[str, Any] = {}
    for concurrency, report in sorted(by_concurrency.items()):
        metrics[f"aggregate_tok_s_c{concurrency}"] = {
            "direction": "higher",
            "unit": "tok/s",
            "samples": _samples(report, "tok_s"),
        }
        metrics[f"batch_latency_s_c{concurrency}"] = {
            "direction": "lower",
            "unit": "s",
            "samples": _samples(report, "elapsed_s"),
        }

    result = {
        "schema_version": 1,
        "profile": profile.name,
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "git_commit": git_commit(),
        "qualification_subject": qualification_subject,
        "comparison_identity": {
            "profile": {
                "name": profile.name,
                "modality": profile.modality,
                "recipe": profile.recipe_id,
                "feature_set": profile.feature_set,
            },
            "model": {
                "target": target.name,
                "target_sha256": sha256(target),
            },
            "hardware": {
                "runner": profile.runner,
                "topology": profile.topology,
                "accelerator": environment["accelerator"],
                "driver": environment["driver"],
                "kernel": environment["kernel"],
                "runtime": environment["runtime"],
                "power_profile": environment["power_profile"],
                "performance_level": environment["performance_level"],
                "toolchain": {
                    "compiler": environment["compiler"],
                    "cmake": environment["cmake"],
                },
            },
            "run": {
                "warmups": profile.performance_workloads[0].warmups,
                "repetitions": profile.performance_workloads[0].repetitions,
                "build_type": environment["build_type"],
                "build_flags": sorted(
                    argument
                    for argument in environment.get("configure_arguments", [])
                    if argument.startswith("-D")
                ),
                "environment": {
                    "decode_mode": profile.decode_mode,
                    "paged_attention": profile.paged_attention,
                    "maximum_context": profile.maximum_context,
                    "maximum_concurrency": profile.maximum_concurrency,
                    "prefix_cache_slots": profile.prefix_cache_slots,
                    "server_arguments": list(profile.server_arguments),
                    "concurrency": sorted(by_concurrency),
                    "prompt_corpus": prompt_corpus,
                    "prompt_sha256": sha256(prompt_path),
                    "max_tokens": max_tokens.pop(),
                    "temperature": temperatures.pop(),
                },
            },
        },
        "metrics": metrics,
    }
    return result
