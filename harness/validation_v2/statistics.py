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
) -> dict[str, tuple[list[float], list[float]]]:
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    failed: set[tuple[str, str]] = set()
    for row in rows:
        prompt_id = str(row["prompt_id"])
        config = str(row["config"])
        if config not in {baseline, candidate}:
            continue
        if row.get("status") != "success" or row.get(metric) is None:
            failed.add((prompt_id, config))
            continue
        grouped[(prompt_id, config)].append(float(row[metric]))

    prompts = sorted({prompt for prompt, _ in grouped} | {prompt for prompt, _ in failed})
    result: dict[str, tuple[list[float], list[float]]] = {}
    incomplete: list[str] = []
    for prompt_id in prompts:
        base = grouped.get((prompt_id, baseline), [])
        cand = grouped.get((prompt_id, candidate), [])
        if (prompt_id, baseline) in failed or (prompt_id, candidate) in failed or not base or not cand:
            incomplete.append(prompt_id)
            continue
        result[prompt_id] = (base, cand)
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
            base_rep = rng.choice(list(baseline))
            candidate_rep = rng.choice(list(candidate))
            if base_rep <= 0:
                raise ValueError(f"baseline repetition must be positive for {prompt_id}")
            sampled_ratios.append(candidate_rep / base_rep)
            sampled_diffs.append(candidate_rep - base_rep)
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
