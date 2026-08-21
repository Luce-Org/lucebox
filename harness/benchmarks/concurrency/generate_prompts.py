#!/usr/bin/env python3
"""Generate deterministic prompt manifests for concurrency runs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "server" / "scripts"))
from bench_he import PROMPTS  # noqa: E402

RAGGED_PROFILES = {
    "short": (250, 350, 450, 550),
    "medium": (650, 850, 1150, 1350),
    "long": (2000, 2600, 3400, 4000),
}

DEFAULT_CLIENT_LEVELS = (2, 4, 8, 16)

WORD_BANK = (
    "systems engineers compare latency throughput scheduling memory kernels queues "
    "batches requests tokens caches pages attention arithmetic bandwidth occupancy "
    "profiling measurement fairness reproducibility workloads concurrency admission "
    "prefill decoding evidence tradeoffs implementation validation production service"
).split()


def parse_client_levels(value: str) -> tuple[int, ...]:
    try:
        levels = tuple(int(item) for item in value.split(","))
    except ValueError as exc:
        raise ValueError("client levels must be comma-separated integers") from exc
    if not levels or any(level < 1 for level in levels):
        raise ValueError("client levels must be positive")
    if len(set(levels)) != len(levels):
        raise ValueError("client levels must be distinct")
    return levels


def cohort_targets(strata: tuple[int, ...], clients: int) -> list[int]:
    if clients < 1:
        raise ValueError("clients must be positive")
    if len(strata) != 4 or sum(strata) % len(strata):
        raise ValueError("ragged profiles require four strata with an integer mean")
    mean = sum(strata) // len(strata)
    cycles, remainder = divmod(clients, len(strata))
    targets = list(strata) * cycles
    if remainder == 1:
        targets.append(mean)
    elif remainder == 2:
        targets.extend((strata[0], strata[-1]))
    elif remainder == 3:
        targets.extend((strata[0], mean, strata[-1]))
    if len(targets) != clients or sum(targets) != clients * mean:
        raise ValueError("profile strata must be symmetric around their mean")
    return targets


def prompt_text(profile: str, cohort: str, index: int, target_words: int) -> str:
    prefix = (
        f"Ragged benchmark {profile} cohort {cohort} request {index}. "
        "Write a structured engineering analysis of the following observations, "
        "including assumptions, likely bottlenecks, and a concise conclusion."
    ).split()
    words = list(prefix)
    cursor = (index * 7 + target_words) % len(WORD_BANK)
    while len(words) < target_words:
        words.append(WORD_BANK[cursor % len(WORD_BANK)])
        cursor += 1
    return " ".join(words[:target_words])


def build_ragged_records(
    profile: str, client_levels: tuple[int, ...] = DEFAULT_CLIENT_LEVELS,
) -> list[dict[str, object]]:
    strata = RAGGED_PROFILES[profile]
    if not client_levels or any(level < 1 for level in client_levels):
        raise ValueError("client levels must be positive")
    if len(set(client_levels)) != len(client_levels):
        raise ValueError("client levels must be distinct")
    records: list[dict[str, object]] = []
    for clients in client_levels:
        cohort = f"c{clients}"
        cohort_offset = len(records)
        for cohort_index, target in enumerate(cohort_targets(strata, clients)):
            records.append({
                "id": f"{profile}-{cohort}-{cohort_index:04d}",
                "cohort": cohort,
                "cohort_clients": clients,
                "cohort_index": cohort_index,
                "cohort_offset": cohort_offset,
                "stratum": strata.index(target) if target in strata else "mean",
                "target_words": target,
                "prompt": prompt_text(profile, cohort, cohort_index, target),
            })
    return records


def build_raw_human_eval_records() -> list[dict[str, object]]:
    return [
        {
            "id": f"he_raw_{index:02d}",
            "suite": "he-raw",
            "name": name,
            "prompt": prompt,
            "max_tokens": 128,
        }
        for index, (name, prompt) in enumerate(PROMPTS, 1)
    ]


def build_records(
    profile: str, client_levels: tuple[int, ...] = DEFAULT_CLIENT_LEVELS,
) -> list[dict[str, object]]:
    if profile == "he-raw":
        return build_raw_human_eval_records()
    return build_ragged_records(profile, client_levels)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile", choices=["he-raw", *sorted(RAGGED_PROFILES)], required=True
    )
    parser.add_argument(
        "--clients", default=",".join(map(str, DEFAULT_CLIENT_LEVELS)),
        help="comma-separated, distinct concurrency levels for disjoint cohorts",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        parser.error(f"refusing to overwrite {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    try:
        client_levels = parse_client_levels(args.clients)
    except ValueError as exc:
        parser.error(str(exc))
    records = build_records(args.profile, client_levels)
    args.out.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in records),
        encoding="utf-8",
    )
    print(f"wrote {len(records)} prompts to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
