"""Pre-generate stable validation-v2 experiment cells."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from .records import canonical_json, sha256_json


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_prompt_hashes(path: Path) -> dict[str, str]:
    """Load unique prompt IDs and bind each ID to its complete source row."""

    prompts: dict[str, str] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            raw = json.loads(line)
            prompt_id = str(raw.get("id") or raw.get("case_id") or raw.get("name") or "")
            if not prompt_id:
                raise ValueError(f"prompt at {path}:{line_number} has no id/name")
            if prompt_id in prompts:
                raise ValueError("prompt IDs must be unique")
            declared = raw.get("prompt_sha256")
            if declared is not None:
                prompt = raw.get("prompt")
                if not isinstance(prompt, str):
                    raise ValueError(
                        f"prompt at {path}:{line_number} declares prompt_sha256 "
                        "without string prompt content"
                    )
                actual = hashlib.sha256(prompt.encode()).hexdigest()
                if declared != actual:
                    raise ValueError(f"prompt hash mismatch at {path}:{line_number}")
            prompts[prompt_id] = sha256_json(raw)
    if not prompts:
        raise ValueError("prompt corpus is empty")
    return prompts


def build_cells(
    *,
    prompt_hashes: Mapping[str, str],
    platform: str,
    environment_hash: str,
    protocol_hash: str,
    configurations: Iterable[Mapping[str, Any]],
    repetitions: int,
    phase: str,
) -> list[dict[str, Any]]:
    if repetitions < 1:
        raise ValueError("repetitions must be >= 1")
    if not prompt_hashes:
        raise ValueError("prompt corpus is empty")
    if platform not in {"dual-v100s", "rtx3090"}:
        raise ValueError(f"unsupported platform: {platform}")
    if phase not in {"tuning", "publication"}:
        raise ValueError(f"unsupported phase: {phase}")
    for label, digest in {
        "environment_hash": environment_hash,
        "protocol_hash": protocol_hash,
        **{f"prompt_hash[{key}]": value for key, value in prompt_hashes.items()},
    }.items():
        if len(digest) != 64 or any(ch not in "0123456789abcdef" for ch in digest):
            raise ValueError(f"{label} must be a lowercase SHA-256 digest")

    configs = [dict(raw) for raw in configurations]
    if not configs:
        raise ValueError("configuration list is empty")
    names = [str(config.get("name", "")).strip() for config in configs]
    if any(not name for name in names):
        raise ValueError("every configuration needs a non-empty name")
    if len(names) != len(set(names)):
        raise ValueError("configuration names must be unique")

    cells: list[dict[str, Any]] = []
    for config in configs:
        config_name = str(config["name"])
        config_hash = sha256_json(config)
        for prompt_id, prompt_hash in prompt_hashes.items():
            for repetition in range(1, repetitions + 1):
                config_id = f"{phase}:{platform}:{config_name}:{prompt_id}:r{repetition}"
                cell = {
                    "config_id": config_id,
                    "config": config,
                    "config_hash": config_hash,
                    "environment_hash": environment_hash,
                    "phase": phase,
                    "platform": platform,
                    "prompt_id": prompt_id,
                    "prompt_hash": prompt_hash,
                    "protocol_hash": protocol_hash,
                    "repetition": repetition,
                }
                cell["cell_hash"] = sha256_json(cell)
                cells.append(cell)
    return cells


def write_cells(path: Path, cells: Iterable[Mapping[str, Any]]) -> str:
    rows = list(cells)
    if not rows:
        raise ValueError("generated matrix is empty")
    config_ids = [str(row["config_id"]) for row in rows]
    if len(config_ids) != len(set(config_ids)):
        raise ValueError("generated matrix has duplicate config IDs")
    for row in rows:
        if row.get("config_hash") != sha256_json(row.get("config", {})):
            raise ValueError(f"matrix config hash mismatch: {row['config_id']}")
        content = dict(row)
        declared_cell_hash = content.pop("cell_hash", None)
        if declared_cell_hash != sha256_json(content):
            raise ValueError(f"matrix cell hash mismatch: {row['config_id']}")
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
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(__file__).with_name("protocol.json"),
    )
    parser.add_argument("--phase", required=True, choices=("tuning", "publication"))
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    configurations = json.loads(args.configurations.read_text(encoding="utf-8"))
    if not isinstance(configurations, list):
        raise ValueError("configurations must be a JSON list")
    cells = build_cells(
        prompt_hashes=load_prompt_hashes(args.prompts),
        platform=args.platform,
        environment_hash=args.environment_hash,
        protocol_hash=_sha256_file(args.protocol),
        configurations=configurations,
        repetitions=args.repetitions,
        phase=args.phase,
    )
    digest = write_cells(args.out, cells)
    print(canonical_json({"cells": len(cells), "path": str(args.out), "sha256": digest}))


if __name__ == "__main__":
    main()
