#!/usr/bin/env python3
"""Validate and summarize paired coding-agent turn-cache benchmarks."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import re
import statistics
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
SAFE_TAG = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", help="tag or path to coding_*.json")
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--minimum-pairs", type=int, default=30)
    return parser.parse_args()


def artifact_path(value: str) -> Path:
    direct = Path(value)
    if direct.is_file():
        return direct
    if not SAFE_TAG.fullmatch(value):
        raise SystemExit("artifact tag must be a short safe identifier")
    return SCRIPT_DIR / "results" / f"coding_{value}.json"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values: list[float], probability: float) -> float:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * probability)
    return ordered[index]


def bootstrap_aggregate_ci(
    pairs: list[dict[str, dict[str, Any]]], samples: int, field: str
) -> tuple[float, float] | None:
    if not pairs or samples <= 0:
        return None
    generator = random.Random(614)
    speedups = []
    for _ in range(samples):
        draw = [pairs[generator.randrange(len(pairs))] for _ in pairs]
        control = sum(pair["control"][field] for pair in draw)
        cached = sum(pair["cache"][field] for pair in draw)
        if cached > 0:
            speedups.append(control / cached)
    if not speedups:
        return None
    return percentile(speedups, 0.025), percentile(speedups, 0.975)


def eligible_prompt_shape(result: dict[str, Any]) -> list[int] | None:
    turns = result.get("turn_log")
    if not isinstance(turns, list):
        return None
    shape: list[int] = []
    for turn in turns:
        if not isinstance(turn, dict) or turn.get("cache_eligible") is not True:
            continue
        timings = turn.get("timings")
        if not isinstance(timings, dict):
            return None
        tokens = timings.get("effective_prompt_tokens")
        if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens <= 0:
            return None
        shape.append(tokens)
    return shape


def main() -> int:
    args = parse_args()
    path = artifact_path(args.artifact)
    with path.open(encoding="utf-8") as artifact_file:
        artifact = json.load(artifact_file)
    if not isinstance(artifact, dict):
        raise SystemExit("benchmark artifact must be a JSON object")
    if artifact.get("schema") != "lucebox.agent-turn-cache-benchmark.v1":
        raise SystemExit("unsupported or missing benchmark schema")
    if args.bootstrap_samples <= 0 or args.minimum_pairs <= 0:
        raise SystemExit("sample and pair counts must be positive")

    raw_results = artifact.get("results")
    if not isinstance(raw_results, list):
        raise SystemExit("benchmark results must be a JSON array")
    grouped: dict[tuple[str, int], dict[str, dict[str, Any]]] = {}
    for result in raw_results:
        if not isinstance(result, dict):
            raise SystemExit("benchmark result is not an object")
        task = result.get("task")
        repetition = result.get("repetition")
        arm = result.get("arm")
        if (not isinstance(task, str) or
            isinstance(repetition, bool) or not isinstance(repetition, int) or
            arm not in {"control", "cache"}):
            raise SystemExit("benchmark result has an invalid task, repetition, or arm")
        if not isinstance(result.get("correct"), bool):
            raise SystemExit(f"{task} has invalid correctness data")
        arms = grouped.setdefault((task, repetition), {})
        if arm in arms:
            raise SystemExit(f"duplicate result for {task} repetition {repetition} arm {arm}")
        arms[arm] = result

    print(
        f"{'task':18} {'rep':>3} {'ctrl_pf':>8} {'cache_pf':>8} "
        f"{'pf_gain':>8} {'turn_gain':>9} {'hits':>7} {'trace':>7} {'correct':>9}"
    )
    complete_pairs = []
    incomplete_pairs = 0
    for (task, repetition), arms in sorted(grouped.items()):
        if "control" not in arms or "cache" not in arms:
            incomplete_pairs += 1
            continue
        control = arms["control"]
        cached = arms["cache"]
        for result in (control, cached):
            for field in (
                "wall_ms", "eligible_prefill_ms", "eligible_model_wall_ms"
            ):
                value = result.get(field)
                if (isinstance(value, bool) or not isinstance(value, (int, float)) or
                    not math.isfinite(value) or value < 0):
                    raise SystemExit(f"{task} has invalid {field}")
            for field in (
                "calls", "eligible_followup_turns", "agent_turn_cache_hits",
                "unexpected_agent_turn_cache_hits", "eligible_prefilled_tokens",
                "eligible_effective_prompt_tokens",
            ):
                value = result.get(field)
                if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                    raise SystemExit(f"{task} has invalid {field}")
            if result["wall_ms"] <= 0:
                raise SystemExit(f"{task} has non-positive wall_ms")
            if (result["eligible_followup_turns"] <= 0 or
                result["eligible_prefill_ms"] <= 0 or
                result["eligible_model_wall_ms"] <= 0):
                raise SystemExit(f"{task} has no measurable follow-up turn")
            if result.get("timing_records_complete") is not True:
                raise SystemExit(f"{task} has incomplete server timing telemetry")
            if not isinstance(result.get("trace_sha256"), str) or not re.fullmatch(
                r"[0-9a-f]{64}", result["trace_sha256"]
            ):
                raise SystemExit(f"{task} has invalid trace_sha256")
            if not isinstance(result.get("assistant_trace_sha256"), str) or not re.fullmatch(
                r"[0-9a-f]{64}", result["assistant_trace_sha256"]
            ):
                raise SystemExit(f"{task} has invalid assistant_trace_sha256")
        prefill_gain = (
            control["eligible_prefill_ms"] / cached["eligible_prefill_ms"]
            if cached["eligible_prefill_ms"] > 0 else 0.0
        )
        turn_gain = (
            control["eligible_model_wall_ms"] / cached["eligible_model_wall_ms"]
            if cached["eligible_model_wall_ms"] > 0 else 0.0
        )
        both_correct = control["correct"] and cached["correct"]
        trace_match = control["trace_sha256"] == cached["trace_sha256"]
        print(
            f"{task:18} {repetition:3d} {control['eligible_prefill_ms']/1000:8.2f} "
            f"{cached['eligible_prefill_ms']/1000:8.2f} {prefill_gain:8.3f} "
            f"{turn_gain:9.3f} {cached['agent_turn_cache_hits']:>2}/"
            f"{cached['eligible_followup_turns']:<4} {str(trace_match):>7} "
            f"{str(both_correct):>9}"
        )
        complete_pairs.append(arms)

    if not complete_pairs:
        print("no complete pairs; no performance claim is valid")
        return 1
    task_count = artifact.get("task_count")
    repetitions = artifact.get("repetitions")
    task_ids = artifact.get("task_ids")
    valid_task_ids = (
        isinstance(task_ids, list) and bool(task_ids) and
        all(isinstance(task, str) and SAFE_TAG.fullmatch(task) for task in task_ids) and
        len(set(task_ids)) == len(task_ids)
    )
    valid_dimensions = (
        isinstance(task_count, int) and not isinstance(task_count, bool) and
        task_count > 0 and isinstance(repetitions, int) and
        not isinstance(repetitions, bool) and repetitions > 0 and
        valid_task_ids and len(task_ids) == task_count
    )
    expected_keys = (
        {(task, repetition) for task in task_ids for repetition in range(repetitions)}
        if valid_dimensions else set()
    )
    expected_pairs = task_count * repetitions if valid_dimensions else -1
    all_correct = all(
        pair["control"].get("correct") is True and
        pair["cache"].get("correct") is True
        for pair in complete_pairs
    )
    exact_tool_trajectories = all(
        pair["control"]["trace_sha256"] == pair["cache"]["trace_sha256"]
        for pair in complete_pairs
    )
    exact_agent_transcripts = all(
        pair["control"]["assistant_trace_sha256"] ==
            pair["cache"]["assistant_trace_sha256"]
        for pair in complete_pairs
    )
    prompt_shapes_match = all(
        eligible_prompt_shape(pair["control"]) == eligible_prompt_shape(pair["cache"])
        and bool(eligible_prompt_shape(pair["control"]))
        for pair in complete_pairs
    )
    full_hit_coverage = all(
        pair["cache"]["eligible_followup_turns"] > 0 and
        pair["cache"]["agent_turn_cache_hits"] ==
            pair["cache"]["eligible_followup_turns"] and
        pair["cache"]["unexpected_agent_turn_cache_hits"] == 0 and
        pair["control"]["agent_turn_cache_hits"] == 0 and
        pair["control"]["unexpected_agent_turn_cache_hits"] == 0
        for pair in complete_pairs
    )
    reduced_prefill_work = all(
        pair["cache"]["eligible_prefilled_tokens"] <
            pair["control"]["eligible_prefilled_tokens"]
        for pair in complete_pairs
    )
    prefill_ratios = [
        pair["control"]["eligible_prefill_ms"] /
            pair["cache"]["eligible_prefill_ms"]
        for pair in complete_pairs
    ]
    turn_ratios = [
        pair["control"]["eligible_model_wall_ms"] /
            pair["cache"]["eligible_model_wall_ms"]
        for pair in complete_pairs
    ]
    loop_ratios = [
        pair["control"]["wall_ms"] / pair["cache"]["wall_ms"]
        for pair in complete_pairs
    ]
    control_prefill = sum(
        pair["control"]["eligible_prefill_ms"] for pair in complete_pairs
    )
    cache_prefill = sum(
        pair["cache"]["eligible_prefill_ms"] for pair in complete_pairs
    )
    control_turn = sum(
        pair["control"]["eligible_model_wall_ms"] for pair in complete_pairs
    )
    cache_turn = sum(
        pair["cache"]["eligible_model_wall_ms"] for pair in complete_pairs
    )
    control_loop = sum(pair["control"]["wall_ms"] for pair in complete_pairs)
    cache_loop = sum(pair["cache"]["wall_ms"] for pair in complete_pairs)
    prefill_interval = bootstrap_aggregate_ci(
        complete_pairs, args.bootstrap_samples, "eligible_prefill_ms"
    )
    turn_interval = bootstrap_aggregate_ci(
        complete_pairs, args.bootstrap_samples, "eligible_model_wall_ms"
    )
    loop_interval = bootstrap_aggregate_ci(
        complete_pairs, args.bootstrap_samples, "wall_ms"
    )
    print(
        f"complete_pairs={len(complete_pairs)} all_correct={all_correct} "
        f"exact_tool_trajectories={exact_tool_trajectories} "
        f"exact_agent_transcripts={exact_agent_transcripts} "
        f"prompt_shapes_match={prompt_shapes_match}"
    )
    print(
        f"follow-up prefill median=x{statistics.median(prefill_ratios):.3f} "
        f"aggregate=x{control_prefill/cache_prefill:.3f} "
        f"bootstrap95={prefill_interval}"
    )
    print(
        f"follow-up request median=x{statistics.median(turn_ratios):.3f} "
        f"aggregate=x{control_turn/cache_turn:.3f} bootstrap95={turn_interval}"
    )
    print(
        f"whole agent loop median=x{statistics.median(loop_ratios):.3f} "
        f"aggregate=x{control_loop/cache_loop:.3f} bootstrap95={loop_interval}"
    )
    print(
        "agent-turn cache hits="
        f"{sum(pair['cache']['agent_turn_cache_hits'] for pair in complete_pairs)}/"
        f"{sum(pair['cache']['eligible_followup_turns'] for pair in complete_pairs)}"
    )
    runtime = artifact.get("runtime") if isinstance(artifact.get("runtime"), dict) else {}
    run_id = artifact.get("run_id")
    checkout = (
        artifact.get("benchmark_checkout")
        if isinstance(artifact.get("benchmark_checkout"), dict) else {}
    )
    workspace = (
        artifact.get("workspace_revision")
        if isinstance(artifact.get("workspace_revision"), dict) else {}
    )
    server_props = (
        artifact.get("server_props")
        if isinstance(artifact.get("server_props"), dict) else {}
    )
    prefix_cache_props = (
        server_props.get("prefix_cache")
        if isinstance(server_props.get("prefix_cache"), dict) else {}
    )
    full_cache_props = (
        server_props.get("full_cache")
        if isinstance(server_props.get("full_cache"), dict) else {}
    )
    digest_fields_recorded = all(
        isinstance(artifact.get(field), str) and
        re.fullmatch(r"[0-9a-f]{64}", artifact[field])
        for field in ("workspace_sha256", "tasks_sha256", "tool_schema_sha256")
    )
    gates = {
        "complete_expected_pairs": (
            isinstance(expected_pairs, int) and expected_pairs > 0 and
            len(complete_pairs) == expected_pairs and incomplete_pairs == 0 and
            set(grouped) == expected_keys
        ),
        "minimum_pair_count": len(complete_pairs) >= args.minimum_pairs,
        "all_pairs_correct": all_correct,
        "exact_control_cache_tool_trajectories": exact_tool_trajectories,
        "exact_control_cache_agent_transcripts": exact_agent_transcripts,
        "equal_eligible_prompt_shapes": prompt_shapes_match,
        "all_eligible_cache_turns_hit": full_hit_coverage,
        "cache_reduces_prefilled_tokens": reduced_prefill_work,
        "engine_commit_recorded_and_clean": (
            isinstance(checkout.get("commit"), str) and
            bool(checkout["commit"].strip()) and
            checkout.get("dirty") is False
        ),
        "workspace_commit_recorded_and_clean": (
            isinstance(workspace.get("commit"), str) and
            bool(workspace["commit"].strip()) and
            workspace.get("dirty") is False
        ),
        "input_hashes_recorded": digest_fields_recorded,
        "run_isolation_id_recorded": (
            isinstance(run_id, str) and SAFE_TAG.fullmatch(run_id) is not None
        ),
        "server_build_recorded": (
            isinstance(runtime.get("server_build_id"), str) and
            bool(runtime["server_build_id"].strip())
        ),
        "server_capability_snapshot_recorded": bool(server_props),
        "agent_turn_cache_enabled_on_server": (
            prefix_cache_props.get("agent_turn_cache_enabled") is True
        ),
        "full_prompt_cache_disabled_for_pair_isolation": (
            isinstance(full_cache_props.get("capacity"), int) and
            not isinstance(full_cache_props.get("capacity"), bool) and
            full_cache_props["capacity"] == 0
        ),
        "hardware_recorded": (
            isinstance(runtime.get("hardware_note"), str) and
            bool(runtime["hardware_note"].strip())
        ),
        "positive_prefill_bootstrap_interval": (
            prefill_interval is not None and prefill_interval[0] > 1.0
        ),
        "positive_followup_wall_bootstrap_interval": (
            turn_interval is not None and turn_interval[0] > 1.0
        ),
        "positive_whole_loop_bootstrap_interval": (
            loop_interval is not None and loop_interval[0] > 1.0
        ),
    }
    print("publication gates:")
    for name, passed in gates.items():
        print(f"  {name}: {'PASS' if passed else 'FAIL'}")
    print(f"artifact_sha256={sha256_file(path)}")
    if not all(gates.values()):
        print("publication gate failed; no headline speedup claim is valid")
        return 1
    print("publication gate passed; report the full artifact and per-task table")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
