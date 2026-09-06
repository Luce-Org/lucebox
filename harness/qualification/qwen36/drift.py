#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def _rss_kib(sample: dict[str, Any]) -> int | None:
    value = sample.get("server_process", {}).get("VmRSS")
    if not isinstance(value, str):
        return None
    try:
        return int(value.split()[0])
    except (IndexError, ValueError):
        return None


def evaluate(
    samples: list[dict[str, Any]],
    *,
    max_rss_growth_mib: float,
    max_health_latency_growth_fraction: float,
    expected_accelerator_device: str,
    minimum_samples: int = 2,
    steady_window_samples: int = 1,
) -> dict[str, Any]:
    if minimum_samples < 2 or steady_window_samples < 1:
        raise ValueError("drift sample requirements must be positive")
    required_samples = max(minimum_samples, steady_window_samples * 3)
    if len(samples) < required_samples:
        raise ValueError(f"drift gate requires at least {required_samples} monitor samples")
    failures: list[str] = []
    if any("monitor_error" in sample for sample in samples):
        failures.append("resource monitor reported an internal error")
    if any(sample.get("health_ok") is not True for sample in samples):
        failures.append("server health endpoint was unavailable during qualification")
    accelerator_samples = [sample.get("accelerator", {}) for sample in samples]
    if any(
        accelerator.get("kind") != "amd"
        or accelerator.get("device") != expected_accelerator_device
        or accelerator.get("ok") is not True
        for accelerator in accelerator_samples
    ):
        failures.append("accelerator telemetry failed during qualification")

    rss = [value for sample in samples if (value := _rss_kib(sample)) is not None]
    if len(rss) == len(samples):
        early_rss = statistics.median(rss[steady_window_samples : steady_window_samples * 2])
        late_rss = statistics.median(rss[-steady_window_samples:])
        rss_growth_mib = (late_rss - early_rss) / 1024.0
    else:
        rss_growth_mib = None
    if rss_growth_mib is None:
        failures.append("server RSS samples are unavailable")
    elif rss_growth_mib > max_rss_growth_mib:
        failures.append("server RSS growth exceeded the configured limit")

    latencies = [
        float(sample["health_latency_ms"])
        for sample in samples
        if isinstance(sample.get("health_latency_ms"), int | float)
        and not isinstance(sample.get("health_latency_ms"), bool)
    ]
    health_latency_growth_fraction = None
    if len(latencies) != len(samples):
        failures.append("health endpoint latency samples are unavailable")
    else:
        early = statistics.median(latencies[steady_window_samples : steady_window_samples * 2])
        late = statistics.median(latencies[-steady_window_samples:])
        health_latency_growth_fraction = (late - early) / early if early > 0 else None
        if health_latency_growth_fraction is None:
            failures.append("initial health endpoint latency is not positive")
        elif health_latency_growth_fraction > max_health_latency_growth_fraction:
            failures.append("health endpoint latency growth exceeded the configured limit")

    return {
        "schema_version": 1,
        "status": "fail" if failures else "pass",
        "sample_count": len(samples),
        "steady_window_samples": steady_window_samples,
        "rss_growth_mib": rss_growth_mib,
        "max_rss_growth_mib": max_rss_growth_mib,
        "accelerator_device": expected_accelerator_device,
        "health_latency_growth_fraction": health_latency_growth_fraction,
        "max_health_latency_growth_fraction": max_health_latency_growth_fraction,
        "failures": failures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Check qualification telemetry for drift.")
    parser.add_argument("monitor_log", type=Path)
    parser.add_argument("--max-rss-growth-mib", type=float, default=512.0)
    parser.add_argument("--max-health-latency-growth-fraction", type=float, default=0.50)
    parser.add_argument("--accelerator-device", required=True)
    parser.add_argument("--minimum-samples", type=int, default=2)
    parser.add_argument("--steady-window-samples", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    samples = [json.loads(line) for line in args.monitor_log.read_text().splitlines() if line]
    report = evaluate(
        samples,
        max_rss_growth_mib=args.max_rss_growth_mib,
        max_health_latency_growth_fraction=args.max_health_latency_growth_fraction,
        expected_accelerator_device=args.accelerator_device,
        minimum_samples=args.minimum_samples,
        steady_window_samples=args.steady_window_samples,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
