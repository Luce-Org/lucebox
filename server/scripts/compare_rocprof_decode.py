#!/usr/bin/env python3
"""Compare per-kernel decode work between two timestamped rocprof traces."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from analyze_rocprof_overlap import request_decode_windows, row_duration_in_windows


def collect(
    trace: Path, requests: Path
) -> tuple[dict[tuple[str, str], int], dict[tuple[str, str], int], int]:
    labeled = request_decode_windows(requests, include_warmup=False)
    windows = [(start, end) for start, end, _ in labeled]
    durations: dict[tuple[str, str], int] = defaultdict(int)
    counts: dict[tuple[str, str], int] = defaultdict(int)
    trace_start: int | None = None
    trace_end: int | None = None
    with trace.open(newline="") as handle:
        for row in csv.DictReader(handle):
            start = int(row["Start_Timestamp"])
            end = int(row["End_Timestamp"])
            trace_start = start if trace_start is None else min(trace_start, start)
            trace_end = end if trace_end is None else max(trace_end, end)
            duration = row_duration_in_windows(start, end, windows)
            if duration <= 0:
                continue
            key = (row["Agent_Id"], row["Kernel_Name"])
            durations[key] += duration
            counts[key] += 1
    if trace_start is None or trace_end is None:
        raise SystemExit(f"empty kernel trace: {trace}")
    request_count = sum(
        1 for start, end in windows
        if max(start, trace_start) < min(end, trace_end)
    )
    if request_count == 0:
        raise SystemExit(f"no measured request overlaps trace: {trace}")
    return durations, counts, request_count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("control_trace", type=Path)
    parser.add_argument("control_requests", type=Path)
    parser.add_argument("candidate_trace", type=Path)
    parser.add_argument("candidate_requests", type=Path)
    parser.add_argument("--top", type=int, default=20)
    args = parser.parse_args()

    control_durations, control_counts, control_n = collect(
        args.control_trace, args.control_requests
    )
    candidate_durations, candidate_counts, candidate_n = collect(
        args.candidate_trace, args.candidate_requests
    )
    agents = sorted(
        {key[0] for key in control_durations} |
        {key[0] for key in candidate_durations}
    )
    print(f"control_requests={control_n} candidate_requests={candidate_n}")
    for agent in agents:
        control_total = sum(
            duration for (key_agent, _), duration in control_durations.items()
            if key_agent == agent
        ) / control_n
        candidate_total = sum(
            duration for (key_agent, _), duration in candidate_durations.items()
            if key_agent == agent
        ) / candidate_n
        print(
            f"{agent} summed_dispatch_ms_per_request "
            f"control={control_total/1e6:.3f} "
            f"candidate={candidate_total/1e6:.3f} "
            f"delta={(candidate_total-control_total)/1e6:+.3f}"
        )

        differences: list[tuple[float, str, float, float, float, float]] = []
        names = {
            name for key_agent, name in control_durations if key_agent == agent
        } | {
            name for key_agent, name in candidate_durations if key_agent == agent
        }
        for name in names:
            key = (agent, name)
            control_ms = control_durations.get(key, 0) / control_n / 1e6
            candidate_ms = candidate_durations.get(key, 0) / candidate_n / 1e6
            control_count = control_counts.get(key, 0) / control_n
            candidate_count = candidate_counts.get(key, 0) / candidate_n
            differences.append((
                candidate_ms - control_ms, name, control_ms, candidate_ms,
                control_count, candidate_count,
            ))

        print(f"largest_added_{agent}")
        for delta, name, control_ms, candidate_ms, control_count, candidate_count in sorted(
            differences, reverse=True
        )[: args.top]:
            print(
                f"{delta:+.3f}ms/request count={control_count:.1f}->{candidate_count:.1f} "
                f"time={control_ms:.3f}->{candidate_ms:.3f} {name}"
            )
        print(f"largest_removed_{agent}")
        for delta, name, control_ms, candidate_ms, control_count, candidate_count in sorted(
            differences
        )[: args.top]:
            print(
                f"{delta:+.3f}ms/request count={control_count:.1f}->{candidate_count:.1f} "
                f"time={control_ms:.3f}->{candidate_ms:.3f} {name}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
