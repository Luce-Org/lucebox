#!/usr/bin/env python3
"""Summarize two-GPU overlap from a rocprofv3 kernel trace CSV."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path


def parse_dispatch(
    row: Mapping[str, str | None],
) -> tuple[str, int, int, str] | None:
    """Parse one rocprof dispatch, ignoring partial or malformed rows."""
    agent = row.get("Agent_Id")
    start_text = row.get("Start_Timestamp")
    end_text = row.get("End_Timestamp")
    if not agent or start_text is None or end_text is None:
        return None
    try:
        start = int(start_text)
        end = int(end_text)
    except ValueError:
        return None
    if end <= start:
        return None
    return agent, start, end, row.get("Kernel_Name") or "<unknown>"


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    ordered = sorted(intervals)
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end:
            if end > prev_end:
                merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))
    return merged


def merge_nearby_intervals(
    intervals: list[tuple[int, int]], max_gap_ns: int
) -> list[tuple[int, int]]:
    """Merge dispatch bursts separated only by launch-sized idle gaps."""
    if not intervals:
        return []
    ordered = sorted(intervals)
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + max_gap_ns:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


def clip_intervals(
    intervals: Sequence[tuple[int, int]], window_start: int, window_end: int
) -> list[tuple[int, int]]:
    """Return only the portions of intervals inside a half-open window."""
    return [
        (max(start, window_start), min(end, window_end))
        for start, end in intervals
        if end > window_start and start < window_end
    ]


def build_timeline_bursts(
    intervals_by_agent: Mapping[str, Sequence[tuple[int, int]]],
    agents: Sequence[str],
    window_start: int,
    window_end: int,
    merge_gap_ns: int,
    per_agent_limit: int,
) -> list[tuple[int, int, str]]:
    """Build a time-ordered timeline with an independent cap per agent."""
    bursts: list[tuple[int, int, str]] = []
    for agent in agents:
        clipped = clip_intervals(
            intervals_by_agent[agent], window_start, window_end
        )
        merged = merge_nearby_intervals(clipped, merge_gap_ns)
        bursts.extend(
            (start, end, agent)
            for start, end in merged[:per_agent_limit]
        )
    return sorted(bursts)


def intersect_intervals(
    left: list[tuple[int, int]], right: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    intersections: list[tuple[int, int]] = []
    i = 0
    j = 0
    while i < len(left) and j < len(right):
        start = max(left[i][0], right[j][0])
        end = min(left[i][1], right[j][1])
        if start < end:
            intersections.append((start, end))
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return intersections


def clipped_duration(
    intervals: list[tuple[int, int]], start: int, end: int
) -> int:
    total = 0
    for interval_start, interval_end in intervals:
        if interval_end <= start:
            continue
        if interval_start >= end:
            break
        total += max(0, min(interval_end, end) - max(interval_start, start))
    return total


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("kernel_trace", type=Path)
    parser.add_argument("--bin-ms", type=float, default=1000.0)
    parser.add_argument("--window-start-s", type=float, default=0.0)
    parser.add_argument("--window-end-s", type=float)
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument(
        "--timeline-max", type=int, default=0,
        help="print at most this many per-agent dispatch bursts in the window",
    )
    parser.add_argument(
        "--timeline-merge-gap-us", type=float, default=25.0,
        help="merge same-agent intervals separated by at most this gap",
    )
    args = parser.parse_args()
    float_args = [
        args.bin_ms,
        args.window_start_s,
        args.timeline_merge_gap_us,
    ]
    if args.window_end_s is not None:
        float_args.append(args.window_end_s)
    if not all(math.isfinite(value) for value in float_args):
        parser.error("floating-point options must be finite")
    if args.bin_ms <= 0 or args.window_start_s < 0:
        parser.error("bin size must be positive and window start non-negative")
    if args.window_end_s is not None and args.window_end_s < 0:
        parser.error("window end must be non-negative")
    if args.timeline_max < 0 or args.timeline_merge_gap_us < 0:
        parser.error("timeline limits and merge gap must be non-negative")

    intervals_by_agent: dict[str, list[tuple[int, int]]] = defaultdict(list)
    trace_start: int | None = None
    trace_end: int | None = None
    with args.kernel_trace.open(newline="") as handle:
        for row in csv.DictReader(handle):
            parsed = parse_dispatch(row)
            if parsed is None:
                continue
            agent, start, end, _ = parsed
            intervals_by_agent[agent].append((start, end))
            trace_start = start if trace_start is None else min(trace_start, start)
            trace_end = end if trace_end is None else max(trace_end, end)

    if trace_start is None or trace_end is None:
        raise SystemExit("kernel trace contains no positive-duration dispatches")
    agents = sorted(intervals_by_agent)
    if len(agents) != 2:
        raise SystemExit(f"expected exactly two GPU agents, found {agents}")
    merged = {agent: merge_intervals(intervals_by_agent[agent]) for agent in agents}
    overlap = intersect_intervals(merged[agents[0]], merged[agents[1]])

    window_start = trace_start + int(args.window_start_s * 1e9)
    requested_end = (
        trace_start + int(args.window_end_s * 1e9)
        if args.window_end_s is not None
        else trace_end
    )
    window_end = min(trace_end, requested_end)
    if window_end <= window_start:
        raise SystemExit("selected window is empty")
    span = window_end - window_start
    busy = {
        agent: clipped_duration(merged[agent], window_start, window_end)
        for agent in agents
    }
    overlap_ns = clipped_duration(overlap, window_start, window_end)

    duration_by_kernel: dict[str, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    count_by_kernel: dict[str, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    if args.top > 0:
        with args.kernel_trace.open(newline="") as handle:
            for row in csv.DictReader(handle):
                parsed = parse_dispatch(row)
                if parsed is None:
                    continue
                agent, dispatch_start, dispatch_end, name = parsed
                start = max(window_start, dispatch_start)
                end = min(window_end, dispatch_end)
                if end <= start:
                    continue
                duration_by_kernel[agent][name] += end - start
                count_by_kernel[agent][name] += 1

    print(
        f"window_s={(window_start-trace_start)/1e9:.3f}:"
        f"{(window_end-trace_start)/1e9:.3f} span_s={span/1e9:.3f}"
    )
    for agent in agents:
        print(
            f"{agent} busy_s={busy[agent]/1e9:.3f} "
            f"utilization={100.0*busy[agent]/span:.2f}%"
        )
    union_ns = busy[agents[0]] + busy[agents[1]] - overlap_ns
    print(
        f"both_busy_s={overlap_ns/1e9:.3f} "
        f"overlap_of_{agents[0]}={100.0*overlap_ns/max(1,busy[agents[0]]):.2f}% "
        f"overlap_of_{agents[1]}={100.0*overlap_ns/max(1,busy[agents[1]]):.2f}% "
        f"either_busy_s={union_ns/1e9:.3f}"
    )

    bin_ns = max(1, int(args.bin_ms * 1e6))
    print("bin_start_s,agent1_busy_pct,agent2_busy_pct,both_busy_pct")
    cursor = window_start
    while cursor < window_end:
        end = min(cursor + bin_ns, window_end)
        width = end - cursor
        print(
            f"{(cursor-trace_start)/1e9:.3f},"
            f"{100.0*clipped_duration(merged[agents[0]], cursor, end)/width:.2f},"
            f"{100.0*clipped_duration(merged[agents[1]], cursor, end)/width:.2f},"
            f"{100.0*clipped_duration(overlap, cursor, end)/width:.2f}"
        )
        cursor = end

    for agent in agents:
        print(f"top_kernels_{agent}")
        top = sorted(
            duration_by_kernel[agent].items(), key=lambda item: item[1], reverse=True
        )[: args.top]
        for name, duration in top:
            print(
                f"{duration/1e9:.6f}s count={count_by_kernel[agent][name]} {name}"
            )

    if args.timeline_max > 0:
        gap_ns = max(0, int(args.timeline_merge_gap_us * 1e3))
        bursts = build_timeline_bursts(
            intervals_by_agent,
            agents,
            window_start,
            window_end,
            gap_ns,
            args.timeline_max,
        )
        print(
            "timeline_start_s,duration_us,agent,"
            f"merge_gap_us={args.timeline_merge_gap_us:g}"
        )
        for start, end, agent in bursts:
            print(
                f"{(start-trace_start)/1e9:.9f},"
                f"{(end-start)/1e3:.3f},{agent}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
