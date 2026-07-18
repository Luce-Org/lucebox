"""Pre-generate stable validation-v2 experiment cells."""

from __future__ import annotations

import argparse
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .records import canonical_json, sha256_json


def load_prompt_ids(path: Path) -> list[str]:
    prompt_ids: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            prompt_id = str(raw.get("id") or raw.get("name") or "")
            if not prompt_id:
                raise ValueError(f"prompt at {path}:{line_number} has no id/name")
            prompt_ids.append(prompt_id)
    if len(prompt_ids) != len(set(prompt_ids)):
        raise ValueError("prompt IDs must be unique")
    if not prompt_ids:
        raise ValueError("prompt corpus is empty")
    return prompt_ids


def build_cells(
    *,
    prompt_ids: Iterable[str],
    platform: str,
    environment_hash: str,
    configurations: Iterable[Mapping[str, Any]],
    repetitions: int,
    phase: str,
) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for raw_config in configurations:
        config = dict(raw_config)
        config_name = str(config["name"])
        config_hash = sha256_json(config)
        for prompt_id in prompt_ids:
            for repetition in range(1, repetitions + 1):
                config_id = f"{phase}:{platform}:{config_name}:{prompt_id}:r{repetition}"
                cells.append(
                    {
                        "config_id": config_id,
                        "config": config,
                        "config_hash": config_hash,
                        "environment_hash": environment_hash,
                        "phase": phase,
                        "platform": platform,
                        "prompt_id": prompt_id,
                        "repetition": repetition,
                    }
                )
    return cells


def write_cells(path: Path, cells: Iterable[Mapping[str, Any]]) -> str:
    rows = list(cells)
    config_ids = [str(row["config_id"]) for row in rows]
    if len(config_ids) != len(set(config_ids)):
        raise ValueError("generated matrix has duplicate config IDs")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
    return sha256_json(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--configurations", type=Path, required=True)
    parser.add_argument("--platform", required=True, choices=("dual-v100s", "rtx3090"))
    parser.add_argument("--environment-hash", required=True)
    parser.add_argument("--phase", required=True, choices=("tuning", "publication"))
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    configurations = json.loads(args.configurations.read_text(encoding="utf-8"))
    if not isinstance(configurations, list):
        raise ValueError("configurations must be a JSON list")
    cells = build_cells(
        prompt_ids=load_prompt_ids(args.prompts),
        platform=args.platform,
        environment_hash=args.environment_hash,
        configurations=configurations,
        repetitions=args.repetitions,
        phase=args.phase,
    )
    digest = write_cells(args.out, cells)
    print(canonical_json({"cells": len(cells), "path": str(args.out), "sha256": digest}))


if __name__ == "__main__":
    main()
