#!/usr/bin/env python3
"""Summarize two-GPU overlap from a rocprofv3 kernel trace CSV."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


def merge_intervals(intervals: list[tuple[int, int]]) -> list[tuple[int, int]]:
    if not intervals:
        return []
    intervals.sort()
    merged = [intervals[0]]
    for start, end in intervals[1:]:
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
    intervals.sort()
    merged = [intervals[0]]
    for start, end in intervals[1:]:
        prev_start, prev_end = merged[-1]
        if start <= prev_end + max_gap_ns:
            merged[-1] = (prev_start, max(prev_end, end))
        else:
            merged.append((start, end))
    return merged


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


def duration_in_windows(
    intervals: list[tuple[int, int]], windows: list[tuple[int, int]]
) -> int:
    return sum(clipped_duration(intervals, start, end) for start, end in windows)


def row_duration_in_windows(
    start: int, end: int, windows: list[tuple[int, int]]
) -> int:
    total = 0
    for window_start, window_end in windows:
        if window_end <= start:
            continue
        if window_start >= end:
            break
        total += max(0, min(end, window_end) - max(start, window_start))
    return total


def request_decode_windows(
    path: Path, include_warmup: bool
) -> list[tuple[int, int, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    windows: list[tuple[int, int, str]] = []
    groups = payload.get("groups")
    if groups is None:
        groups = [{"target_context": "publication", "records": payload.get("records", [])}]
    for group in groups:
        target = group.get("target_context", "unknown")
        for record in group.get("records", []):
            if not record.get("ok") or (not include_warmup and not record.get("measured")):
                continue
            start = record.get("first_token_monotonic_ns")
            end = record.get("request_end_monotonic_ns")
            if start is None or end is None or int(end) <= int(start):
                continue
            label = f"ctx={target},index={record.get('index', '?')}"
            windows.append((int(start), int(end), label))
    if not windows:
        raise SystemExit(
            "request JSON has no usable decode timestamps; rerun with the "
            "instrumented publication client"
        )
    windows.sort()
    return windows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("kernel_trace", type=Path)
    parser.add_argument("--bin-ms", type=float, default=1000.0)
    parser.add_argument("--window-start-s", type=float, default=0.0)
    parser.add_argument("--window-end-s", type=float)
    parser.add_argument(
        "--requests-json", type=Path,
        help="select first-token-to-request-end windows from benchmark JSON",
    )
    parser.add_argument(
        "--include-warmup", action="store_true",
        help="include warmup request windows when --requests-json is used",
    )
    parser.add_argument(
        "--memory-copy-trace", type=Path,
        help="optional rocprofv3 memory-copy CSV summarized over the same windows",
    )
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
    if args.bin_ms <= 0 or args.window_start_s < 0:
        parser.error("bin size must be positive and window start non-negative")

    intervals_by_agent: dict[str, list[tuple[int, int]]] = defaultdict(list)
    trace_start: int | None = None
    trace_end: int | None = None
    with args.kernel_trace.open(newline="") as handle:
        for row in csv.DictReader(handle):
            agent = row["Agent_Id"]
            start = int(row["Start_Timestamp"])
            end = int(row["End_Timestamp"])
            if end <= start:
                continue
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
    labeled_windows: list[tuple[int, int, str]]
    if args.requests_json:
        labeled_windows = []
        for start, end, label in request_decode_windows(
            args.requests_json, args.include_warmup
        ):
            start = max(start, window_start)
            end = min(end, window_end)
            if start < end:
                labeled_windows.append((start, end, label))
        if not labeled_windows:
            raise SystemExit("no request decode window overlaps the kernel trace")
    else:
        labeled_windows = [(window_start, window_end, "trace")]
    selected_windows = merge_intervals(
        [(start, end) for start, end, _ in labeled_windows]
    )
    span = sum(end - start for start, end in selected_windows)
    busy = {
        agent: duration_in_windows(merged[agent], selected_windows)
        for agent in agents
    }
    overlap_ns = duration_in_windows(overlap, selected_windows)

    duration_by_kernel: dict[str, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    count_by_kernel: dict[str, dict[str, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    if args.top > 0:
        with args.kernel_trace.open(newline="") as handle:
            for row in csv.DictReader(handle):
                agent = row["Agent_Id"]
                start = int(row["Start_Timestamp"])
                end = int(row["End_Timestamp"])
                duration = row_duration_in_windows(start, end, selected_windows)
                if duration <= 0:
                    continue
                name = row["Kernel_Name"]
                duration_by_kernel[agent][name] += duration
                count_by_kernel[agent][name] += 1

    print(
        f"window_s={(selected_windows[0][0]-trace_start)/1e9:.3f}:"
        f"{(selected_windows[-1][1]-trace_start)/1e9:.3f} "
        f"selected_span_s={span/1e9:.3f} windows={len(selected_windows)}"
    )
    for agent in agents:
        print(
            f"{agent} busy_s={busy[agent]/1e9:.3f} "
            f"utilization={100.0*busy[agent]/span:.2f}%"
        )
    union_ns = busy[agents[0]] + busy[agents[1]] - overlap_ns
    agent0_only = busy[agents[0]] - overlap_ns
    agent1_only = busy[agents[1]] - overlap_ns
    idle_ns = span - union_ns
    print(
        f"both_busy_s={overlap_ns/1e9:.3f} "
        f"overlap_of_{agents[0]}={100.0*overlap_ns/max(1,busy[agents[0]]):.2f}% "
        f"overlap_of_{agents[1]}={100.0*overlap_ns/max(1,busy[agents[1]]):.2f}% "
        f"either_busy_s={union_ns/1e9:.3f}"
    )
    print(
        f"{agents[0]}_only_s={agent0_only/1e9:.3f} "
        f"{agents[1]}_only_s={agent1_only/1e9:.3f} "
        f"neither_busy_s={idle_ns/1e9:.3f}"
    )

    bin_ns = max(1, int(args.bin_ms * 1e6))
    print("bin_start_s,agent1_busy_pct,agent2_busy_pct,both_busy_pct")
    for selected_start, selected_end in selected_windows:
        cursor = selected_start
        while cursor < selected_end:
            end = min(cursor + bin_ns, selected_end)
            width = end - cursor
            print(
                f"{(cursor-trace_start)/1e9:.3f},"
                f"{100.0*clipped_duration(merged[agents[0]], cursor, end)/width:.2f},"
                f"{100.0*clipped_duration(merged[agents[1]], cursor, end)/width:.2f},"
                f"{100.0*clipped_duration(overlap, cursor, end)/width:.2f}"
            )
            cursor = end

    if args.requests_json:
        print("request_decode,label,start_s,span_s,agent1_busy_pct,agent2_busy_pct,both_busy_pct")
        for start, end, label in labeled_windows:
            width = end - start
            print(
                f"request_decode,{label},{(start-trace_start)/1e9:.6f},"
                f"{width/1e9:.6f},"
                f"{100.0*clipped_duration(merged[agents[0]], start, end)/width:.2f},"
                f"{100.0*clipped_duration(merged[agents[1]], start, end)/width:.2f},"
                f"{100.0*clipped_duration(overlap, start, end)/width:.2f}"
            )

    for agent in agents:
        print(f"top_kernels_{agent}")
        top = sorted(
            duration_by_kernel[agent].items(), key=lambda item: item[1], reverse=True
        )[: args.top]
        for name, duration in top:
            print(
                f"{duration/1e9:.6f}s count={count_by_kernel[agent][name]} {name}"
            )

    if args.memory_copy_trace:
        copy_duration: dict[tuple[str, str, str], int] = defaultdict(int)
        copy_count: dict[tuple[str, str, str], int] = defaultdict(int)
        with args.memory_copy_trace.open(newline="") as handle:
            for row in csv.DictReader(handle):
                start = int(row["Start_Timestamp"])
                end = int(row["End_Timestamp"])
                duration = row_duration_in_windows(start, end, selected_windows)
                if duration <= 0:
                    continue
                key = (
                    row["Direction"], row["Source_Agent_Id"],
                    row["Destination_Agent_Id"],
                )
                copy_duration[key] += duration
                copy_count[key] += 1
        print("memory_copies")
        for key, duration in sorted(
            copy_duration.items(), key=lambda item: item[1], reverse=True
        ):
            direction, source, destination = key
            print(
                f"{duration/1e6:.3f}ms count={copy_count[key]} "
                f"{direction} {source}->{destination}"
            )

    if args.timeline_max > 0:
        gap_ns = max(0, int(args.timeline_merge_gap_us * 1e3))
        bursts: list[tuple[int, int, str]] = []
        for agent in agents:
            for start, end in merge_nearby_intervals(
                intervals_by_agent[agent], gap_ns
            ):
                for window_start, window_end in selected_windows:
                    clipped_start = max(start, window_start)
                    clipped_end = min(end, window_end)
                    if clipped_start < clipped_end:
                        bursts.append((clipped_start, clipped_end, agent))
        bursts.sort()
        print(
            "timeline_start_s,duration_us,agent,"
            f"merge_gap_us={args.timeline_merge_gap_us:g}"
        )
        for start, end, agent in bursts[: args.timeline_max]:
            print(
                f"{(start-trace_start)/1e9:.9f},"
                f"{(end-start)/1e3:.3f},{agent}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
