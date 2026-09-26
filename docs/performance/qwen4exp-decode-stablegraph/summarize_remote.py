#!/usr/bin/env python3
"""Summarize the lock-held E23 measurements on lucebox4."""

import csv
import json
import re
import statistics
from collections import Counter
from pathlib import Path


FOLLOWUP = Path("/tmp/q4exp-followup")
OUT = Path("/tmp/q4exp-decode-stablegraph")


def timing(row, key):
    return row["usage"]["timings"][key]


summary = {"decode_ab": {}, "phase_profile": {}, "prefill": {}, "runtime_trace": {}}
for variant in ("control", "candidate"):
    rows = json.loads((FOLLOWUP / f"stablegraph-{variant}.json").read_text())
    summary["decode_ab"][variant] = {}
    for depth in (2048, 16384):
        selected = [r for r in rows if r["target_depth"] == depth]
        rates = [128000.0 / timing(r, "decode_ms") for r in selected]
        summary["decode_ab"][variant][str(depth)] = {
            "n": len(rates),
            "effective_prompt_tokens": sorted({timing(r, "effective_prompt_tokens") for r in selected}),
            "median_tps": statistics.median(rates),
            "min_tps": min(rates),
            "max_tps": max(rates),
            "median_decode_ms": statistics.median(timing(r, "decode_ms") for r in selected),
        }

for depth in (2048, 16384):
    control = summary["decode_ab"]["control"][str(depth)]["median_tps"]
    candidate = summary["decode_ab"]["candidate"][str(depth)]["median_tps"]
    summary["decode_ab"]["candidate"][str(depth)]["delta_percent"] = 100 * (candidate / control - 1)

for variant in ("control", "candidate"):
    values = []
    for line in (FOLLOWUP / f"stablegraph-phase-{variant}.server.log").read_text().splitlines():
        if not re.search(r"\[qwen4exp-prof\] T=1(?: |$)", line):
            continue
        values.append(float(re.search(r"build\+alloc=([0-9.]+)ms", line).group(1)))
    assert len(values) == 508
    summary["phase_profile"][variant] = {}
    for depth, selected in ((2048, values[:254]), (16384, values[254:])):
        summary["phase_profile"][variant][str(depth)] = {
            "n": len(selected),
            "mean_build_alloc_ms": statistics.mean(selected),
            "median_build_alloc_ms": statistics.median(selected),
            "min_build_alloc_ms": min(selected),
            "max_build_alloc_ms": max(selected),
            "nonzero_calls": sum(v > 0 for v in selected),
        }

prefill = []
for line in (OUT / "prefill.txt").read_text().splitlines():
    match = re.search(r"pt=(\d+) ttft=([0-9.]+)s", line)
    if match:
        tokens, seconds = int(match.group(1)), float(match.group(2))
        prefill.append({"tokens": tokens, "ttft_s": seconds, "tps": tokens / seconds})
summary["prefill"] = {
    "n": len(prefill),
    "samples": prefill,
    "min_latency_s": min(r["ttft_s"] for r in prefill),
    "median_latency_s": statistics.median(r["ttft_s"] for r in prefill),
    "max_latency_s": max(r["ttft_s"] for r in prefill),
    "best_tps": max(r["tps"] for r in prefill),
    "median_tps": statistics.median(r["tps"] for r in prefill),
    "min_tps": min(r["tps"] for r in prefill),
}

trace_dir = FOLLOWUP / "stablegraph-runtime-trace-trace"
api_path = trace_dir / "stablegraph-runtime-trace_hip_api_trace.csv"
kernel_path = trace_dir / "stablegraph-runtime-trace_kernel_trace.csv"
api_counts = Counter()
graph_correlations = []
with api_path.open(newline="") as handle:
    for row in csv.DictReader(handle):
        api_counts[row["Function"]] += 1
        if row["Function"] == "hipGraphLaunch":
            graph_correlations.append(int(row["Correlation_Id"]))
kernel_counts = Counter()
kernel_rows = 0
with kernel_path.open(newline="") as handle:
    for row in csv.DictReader(handle):
        kernel_rows += 1
        kernel_counts[int(row["Correlation_Id"])] += 1
per_graph = Counter(kernel_counts[c] for c in graph_correlations)
summary["runtime_trace"] = {
    "generated_positions": 256,
    "t1_forward_calls": 254,
    "hip_graph_launch_calls": api_counts["hipGraphLaunch"],
    "hip_launch_kernel_calls": api_counts["hipLaunchKernel"],
    "total_kernel_trace_rows_including_prefill": kernel_rows,
    "device_kernels_per_graph_launch_distribution": dict(sorted(per_graph.items())),
    "first_graph_correlations": graph_correlations[:8],
    "raw_remote_paths": [str(api_path), str(kernel_path)],
}

(OUT / "measurement-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
