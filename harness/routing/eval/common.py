"""Shared paths, JSONL helpers and backend config for the routing eval pipeline."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any

ROUTING_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROUTING_DIR / "data"
RUNS_DIR = ROUTING_DIR / "runs"
CONFIG_DIR = ROUTING_DIR / "config"
PROMPTS_PATH = DATA_DIR / "prompts.jsonl"
BACKENDS_PATH = CONFIG_DIR / "backends.json"

# Used when config/backends.json is missing (e.g. in unit tests).
DEFAULT_BACKENDS: dict[str, dict[str, Any]] = {
    "qwen35-0.8b": {"base_url": "http://127.0.0.1:8401/v1", "rank": 0},
    "qwen35-2b": {"base_url": "http://127.0.0.1:8402/v1", "rank": 1},
    "qwen38-27b": {"base_url": "http://127.0.0.1:8216/v1", "kind": "luce_server", "rank": 2},
}

# The model whose thinking-on answer serves as the judge reference.
REFERENCE_MODEL = "qwen38-27b"


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        return []
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def iter_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    with Path(path).open() as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_jsonl(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


def run_key(row: dict[str, Any]) -> tuple[str, str, bool]:
    """Identity of one (prompt, model, thinking) run."""
    return (row["id"], row["model"], bool(row["thinking"]))


def latest_by_key(rows: Iterable[dict[str, Any]]) -> dict[tuple[str, str, bool], dict[str, Any]]:
    """Collapse append-only run files: the last row for each key wins."""
    out: dict[tuple[str, str, bool], dict[str, Any]] = {}
    for row in rows:
        out[run_key(row)] = row
    return out


def load_backends(path: str | Path | None = None) -> dict[str, dict[str, Any]]:
    """Return {model: spec} ordered by ascending rank (smallest model first)."""
    p = Path(path) if path else BACKENDS_PATH
    if p.exists():
        models = json.loads(p.read_text()).get("models", {})
    else:
        models = DEFAULT_BACKENDS
    return dict(sorted(models.items(), key=lambda kv: kv[1].get("rank", 0)))


def model_ranks(backends: dict[str, dict[str, Any]]) -> dict[str, int]:
    return {m: int(spec.get("rank", i)) for i, (m, spec) in enumerate(backends.items())}


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def strip_think(text: str | None) -> str:
    """Drop reasoning blocks so graders only see the final answer."""
    if not text:
        return ""
    text = _THINK_RE.sub("", text)
    # Unterminated or header-less reasoning: keep only what follows the last </think>.
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[1]
    return text.strip()


def prompt_text(messages: list[dict[str, Any]]) -> str:
    """Flatten chat messages to plain text (for kNN / judge display)."""
    parts = []
    for m in messages:
        content = m.get("content")
        if isinstance(content, list):
            content = "\n".join(c.get("text", "") for c in content if isinstance(c, dict))
        parts.append(f"{m.get('role', 'user')}: {content}" if len(messages) > 1 else str(content))
    return "\n\n".join(parts)
