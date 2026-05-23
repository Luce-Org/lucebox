"""Typed metrics parser for bandit run log lines.

Parses JSONL log lines emitted by the adaptive bandit / client harness.
All optional fields use None instead of sentinel strings like "N/A".
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional


@dataclass
class BanditRunMetrics:
    """Typed representation of one bandit run record."""

    session_id: Optional[str] = None
    accept_rate: Optional[float] = None
    wall_s: Optional[float] = None
    tokens: Optional[int] = None
    client: Optional[str] = None
    condition: Optional[str] = None


def parse_bandit_log_line(line: str) -> Optional[BanditRunMetrics]:
    """Parse a single log line. Returns None for non-JSON or non-record lines."""
    line = line.strip()
    if not line or not line.startswith("{"):
        return None
    try:
        obj = json.loads(line)
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None

    accept_raw = obj.get("accept_rate")
    wall_raw = obj.get("wall_s")
    tokens_raw = obj.get("tokens")

    return BanditRunMetrics(
        session_id=obj.get("session_id") or None,
        accept_rate=float(accept_raw) if accept_raw is not None else None,
        wall_s=float(wall_raw) if wall_raw is not None else None,
        tokens=int(tokens_raw) if tokens_raw is not None else None,
        client=obj.get("client") or None,
        condition=obj.get("condition") or None,
    )


def parse_bandit_log(text: str) -> list[BanditRunMetrics]:
    """Parse a multi-line log string. Skips non-record lines."""
    results = []
    for line in text.splitlines():
        m = parse_bandit_log_line(line)
        if m is not None:
            results.append(m)
    return results
