"""Typed metrics parser for bandit run log lines.

Parses JSONL log lines emitted by the adaptive bandit / client harness.
All optional fields use None instead of sentinel strings like "N/A".
Also parses [spec-decode] plain-text log lines for accept_rate fallback.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Optional

# Matches: [spec-decode] tokens=123 time=4.56 s speed=27.1 tok/s steps=10 accepted=8/10
_SPEC_DECODE_RE = re.compile(
    r"\[spec-decode\].*?steps=(\d+)\s+accepted=(\d+)/(\d+)"
)
_PFLASH_BANDIT_ACCEPT_RE = re.compile(r"\[pflash-bandit\].*?\baccept=([0-9]*\.?[0-9]+)")

# Matches: [pflash-bandit] session=X turn=N keep=A->B ema=C accept=D
_PFLASH_BANDIT_TURN_RE = re.compile(
    r"\[pflash-bandit\]\s+"
    r"session=(\S+)\s+"
    r"turn=(\d+)\s+"
    r"keep=([0-9]*\.?[0-9]+)->([0-9]*\.?[0-9]+)\s+"
    r"ema=([0-9]*\.?[0-9]+)\s+"
    r"accept=([0-9]*\.?[0-9]+)"
)


@dataclass
class BanditTurnRecord:
    """Per-turn record parsed from a plain-text [pflash-bandit] log line."""

    session_id: str
    turn: int
    keep_before: float
    keep_after: float
    ema: float
    accept_rate: float
    wall_s: Optional[float] = None


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


def parse_spec_decode_line(line: str) -> Optional[BanditRunMetrics]:
    """Parse a [spec-decode] plain-text log line.

    Example input:
        [spec-decode] tokens=312 time=18.50 s speed=16.9 tok/s steps=10 accepted=8/10

    Returns BanditRunMetrics with accept_rate=accepted/total, or None if no match.
    """
    m = _SPEC_DECODE_RE.search(line)
    if not m:
        return None
    accepted = int(m.group(2))
    total = int(m.group(3))
    if total == 0:
        return None
    return BanditRunMetrics(accept_rate=float(accepted) / float(total))


def parse_bandit_log(text: str) -> list[BanditRunMetrics]:
    """Parse a multi-line log string. Skips non-record lines."""
    results = []
    for line in text.splitlines():
        m = parse_bandit_log_line(line)
        if m is not None:
            results.append(m)
    return results


def extract_accept_rate_from_log(log_text: str) -> Optional[float]:
    """Extract the best accept_rate signal from a server log.

    Strategy:
    1. Scan for [pflash-bandit] JSONL lines — use the LAST one (converged state).
    2. Fall back to plain-text [pflash-bandit] accept=... lines — use the LAST one.
    3. Fall back to [spec-decode] lines — use the LAST one.
    4. Return None if neither is present.
    """
    last_bandit: Optional[BanditRunMetrics] = None
    last_plain_bandit: Optional[float] = None
    last_spec: Optional[BanditRunMetrics] = None

    for line in log_text.splitlines():
        stripped = line.strip()
        # [pflash-bandit] lines embed JSON after the prefix
        if "[pflash-bandit]" in stripped:
            json_start = stripped.find("{")
            if json_start != -1:
                m = parse_bandit_log_line(stripped[json_start:])
                if m is not None and m.accept_rate is not None:
                    last_bandit = m
        plain_match = _PFLASH_BANDIT_ACCEPT_RE.search(stripped)
        if plain_match:
            try:
                last_plain_bandit = float(plain_match.group(1))
            except ValueError:
                pass
        # [spec-decode] plain-text lines
        if "[spec-decode]" in stripped:
            m2 = parse_spec_decode_line(stripped)
            if m2 is not None:
                last_spec = m2

    if last_bandit is not None:
        return last_bandit.accept_rate
    if last_plain_bandit is not None:
        return last_plain_bandit
    if last_spec is not None:
        return last_spec.accept_rate
    return None


def parse_bandit_session_from_log(
    log_text: str,
    *,
    session_id: Optional[str] = None,
) -> list[BanditTurnRecord]:
    """Extract per-turn bandit records from a server log.

    Parses lines matching:
        [pflash-bandit] session=X turn=N keep=A->B ema=C accept=D

    If session_id is given, only records for that session are returned.
    Records are returned in log order (i.e. turn order).
    """
    records: list[BanditTurnRecord] = []
    for line in log_text.splitlines():
        m = _PFLASH_BANDIT_TURN_RE.search(line)
        if not m:
            continue
        sid = m.group(1)
        if session_id is not None and sid != session_id:
            continue
        records.append(BanditTurnRecord(
            session_id=sid,
            turn=int(m.group(2)),
            keep_before=float(m.group(3)),
            keep_after=float(m.group(4)),
            ema=float(m.group(5)),
            accept_rate=float(m.group(6)),
        ))
    return records
