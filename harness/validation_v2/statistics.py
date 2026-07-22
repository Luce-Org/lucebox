"""Paired prompt-level summaries with nested technical repetitions."""

from __future__ import annotations

import math
import random
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot take a quantile of an empty sequence")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def paired_prompt_values(
    rows: Iterable[Mapping[str, Any]],
    *,
    baseline: str,
    candidate: str,
    metric: str,
    expected_repetitions: int | None = None,
    expected_prompt_ids: Iterable[str] | None = None,
) -> dict[str, tuple[list[float], list[float]]]:
    if expected_repetitions is not None and expected_repetitions < 1:
        raise ValueError("expected_repetitions must be >= 1")

    latest: dict[str, Mapping[str, Any]] = {}
    legacy_rows: list[Mapping[str, Any]] = []
    for row in rows:
        config_id = row.get("config_id")
        attempt = row.get("attempt")
        if config_id is None and attempt is None:
            legacy_rows.append(row)
            continue
        if config_id is None or attempt is None:
            raise ValueError("retry-aware rows require both config_id and attempt")
        attempt_number = int(attempt)
        if attempt_number < 1:
            raise ValueError("attempt must be >= 1")
        key = str(config_id)
        previous = latest.get(key)
        if previous is not None and int(previous["attempt"]) == attempt_number:
            raise ValueError(f"duplicate result attempt: {key}#{attempt_number}")
        if previous is None or int(previous["attempt"]) < attempt_number:
            latest[key] = row

    current_rows = [*legacy_rows, *latest.values()]
    grouped: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
    failed: set[tuple[str, str]] = set()
    for row in current_rows:
        prompt_id = str(row["prompt_id"])
        config = str(row["config"])
        if config not in {baseline, candidate}:
            continue
        repetition = row.get("repetition")
        if repetition is None:
            raise ValueError("paired analysis rows require repetition")
        repetition_number = int(repetition)
        if repetition_number < 1:
            raise ValueError("repetition must be >= 1")
        if row.get("status") != "success" or row.get(metric) is None:
            failed.add((prompt_id, config))
            continue
        value = float(row[metric])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"{metric} must be finite and positive for {prompt_id}/{config}")
        values = grouped[(prompt_id, config)]
        if repetition_number in values:
            raise ValueError(
                f"duplicate repetition for {prompt_id}/{config}: {repetition_number}"
            )
        values[repetition_number] = value

    observed_prompts = {prompt for prompt, _ in grouped} | {prompt for prompt, _ in failed}
    prompts = sorted(observed_prompts | set(expected_prompt_ids or ()))
    result: dict[str, tuple[list[float], list[float]]] = {}
    incomplete: list[str] = []
    for prompt_id in prompts:
        base_by_rep = grouped.get((prompt_id, baseline), {})
        cand_by_rep = grouped.get((prompt_id, candidate), {})
        base_reps = set(base_by_rep)
        cand_reps = set(cand_by_rep)
        if (
            (prompt_id, baseline) in failed
            or (prompt_id, candidate) in failed
            or not base_by_rep
            or not cand_by_rep
        ):
            incomplete.append(prompt_id)
            continue
        if base_reps != cand_reps:
            incomplete.append(prompt_id)
            continue
        if expected_repetitions is not None and base_reps != set(
            range(1, expected_repetitions + 1)
        ):
            incomplete.append(prompt_id)
            continue
        repetitions = sorted(base_reps)
        result[prompt_id] = (
            [base_by_rep[index] for index in repetitions],
            [cand_by_rep[index] for index in repetitions],
        )
    if incomplete:
        raise ValueError(f"paired analysis has failed/incomplete prompts: {incomplete}")
    if not result:
        raise ValueError("paired analysis has no complete prompts")
    return result


def summarize_paired(
    pairs: Mapping[str, tuple[Sequence[float], Sequence[float]]],
    *,
    bootstrap_samples: int = 10_000,
    seed: int = 20260718,
) -> dict[str, Any]:
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be >= 1")
    prompt_ids = sorted(pairs)
    ratios: dict[str, float] = {}
    differences: dict[str, float] = {}
    for prompt_id, (baseline, candidate) in pairs.items():
        base_mean = statistics.fmean(baseline)
        candidate_mean = statistics.fmean(candidate)
        if base_mean <= 0:
            raise ValueError(f"baseline mean must be positive for {prompt_id}")
        ratios[prompt_id] = candidate_mean / base_mean
        differences[prompt_id] = candidate_mean - base_mean

    rng = random.Random(seed)
    bootstrap_geomeans: list[float] = []
    bootstrap_differences: list[float] = []
    for _ in range(bootstrap_samples):
        sampled_prompts = [rng.choice(prompt_ids) for _ in prompt_ids]
        sampled_ratios: list[float] = []
        sampled_diffs: list[float] = []
        for prompt_id in sampled_prompts:
            baseline, candidate = pairs[prompt_id]
            if len(baseline) != len(candidate):
                raise ValueError(f"paired repetitions differ for {prompt_id}")
            sampled_indices = [rng.randrange(len(baseline)) for _ in baseline]
            base_mean = statistics.fmean(baseline[index] for index in sampled_indices)
            candidate_mean = statistics.fmean(candidate[index] for index in sampled_indices)
            if base_mean <= 0 or candidate_mean <= 0:
                raise ValueError(f"repetition means must be positive for {prompt_id}")
            sampled_ratios.append(candidate_mean / base_mean)
            sampled_diffs.append(candidate_mean - base_mean)
        bootstrap_geomeans.append(math.exp(statistics.fmean(math.log(x) for x in sampled_ratios)))
        bootstrap_differences.append(statistics.fmean(sampled_diffs))

    ratio_values = list(ratios.values())
    difference_values = list(differences.values())
    return {
        "independent_unit": "prompt",
        "prompt_count": len(prompt_ids),
        "technical_repetitions": {
            prompt_id: {"baseline": len(pairs[prompt_id][0]), "candidate": len(pairs[prompt_id][1])}
            for prompt_id in prompt_ids
        },
        "geometric_mean_ratio": math.exp(statistics.fmean(math.log(x) for x in ratio_values)),
        "median_ratio": statistics.median(ratio_values),
        "mean_difference": statistics.fmean(difference_values),
        "slower_prompt_count": sum(value < 1.0 for value in ratio_values),
        "ratio_ci95": [
            _quantile(bootstrap_geomeans, 0.025),
            _quantile(bootstrap_geomeans, 0.975),
        ],
        "difference_ci95": [
            _quantile(bootstrap_differences, 0.025),
            _quantile(bootstrap_differences, 0.975),
        ],
        "per_prompt_ratio": ratios,
        "per_prompt_difference": differences,
        "bootstrap": {"samples": bootstrap_samples, "seed": seed, "nested_repetitions": True},
    }
