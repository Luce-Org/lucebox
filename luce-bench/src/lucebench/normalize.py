"""Load any historical luce-bench result.json into the canonical schema.

Why this exists
---------------
luce-bench result files in the wild come in two incompatible shapes (see
``lucebench.schema`` for the long version). This module is the
load-time bridge: feed it a raw dict (or a file path), get back a
``CanonicalResult`` regardless of which shape it was written in.

``regrade.py`` and any future ``luce-bench report`` consumer should
load via ``normalize_result`` — never read ``result["pass_rate"]``
directly, because that field is a fraction in legacy files and a
percent in 0.2.7 files, and getting the unit wrong silently inflates
or deflates the headline number by 100x.

Two unit decisions worth flagging:

  * The detection heuristic: pass_rate values in ``[0.0, 1.0]`` are
    treated as fractions; values in ``(1.0, 100.0]`` are treated as
    percent and divided by 100. We tag the conversion in the canonical
    result's ``metrics["pass_rate_unit"]`` so a downstream consumer
    can see it happened.

  * ``semantic_pass_rate`` / ``semantic_passed`` are dropped on load
    unless a real semantic judge is plumbed (today: never). They were
    always 0.0 in shipped results and have been misread as "the
    semantic score crashed". If you wire a judge in the future, emit
    its score under ``metrics["semantic_judge"][<judge_id>]``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from lucebench._thinking import verify_thinking_control
from lucebench.schema import (
    SCHEMA_VERSION,
    CanonicalResult,
    CanonicalRow,
    HostInfo,
    from_dict,
    host_from_dict,
)


def _normalize_pass_rate(value: Any) -> tuple[float, str]:
    """Detect the unit of a legacy ``pass_rate`` field.

    Returns ``(fraction, tag)`` where ``tag`` describes how the value
    was interpreted so the caller can stash it in ``metrics`` for
    audit-ability.

    Heuristic:
      * None / non-numeric / negative → 0.0, ``"missing"``
      * 0.0 ≤ x ≤ 1.0  → already a fraction
      * 1.0 < x ≤ 100.0 → percent, divide by 100
      * x > 100.0 → clamp at 1.0 and tag as ``"clamped"`` (almost
        certainly a bug upstream; preserve so the rest of the file
        still loads)
    """
    if not isinstance(value, int | float):
        return 0.0, "missing"
    v = float(value)
    if v < 0:
        return 0.0, "missing"
    if 0.0 <= v <= 1.0:
        return v, "fraction"
    if v <= 100.0:
        return v / 100.0, "normalized_from_legacy_percent"
    return 1.0, "clamped"


def _looks_like_legacy(raw: dict[str, Any]) -> bool:
    """True iff the raw dict came from the pre-0.2.5 result.json layout.

    Tells legacy (``id, output, ok, graded_pass, strict_pass``) from
    current (``case_id, content, pass, graded{}``) by sniffing the
    first row's keys. Falls back to a top-level ``strict_pass_rate``
    field (which only the legacy writer emitted) when there are zero
    rows.
    """
    rows = raw.get("rows") or raw.get("results") or []
    if rows:
        keys = set(rows[0].keys())
        if "case_id" in keys or "graded" in keys:
            return False
        if "id" in keys or "output" in keys or "graded_pass" in keys:
            return True
    # Empty rows: rely on top-level fields the writers diverge on.
    return "strict_pass_rate" in raw and "passed" in raw


def _normalize_legacy_row(raw: dict[str, Any]) -> CanonicalRow:
    """Map a pre-0.2.5 row dict to a CanonicalRow."""
    case_id = str(raw.get("id") or raw.get("case_id") or "")
    timings = raw.get("timings") if isinstance(raw.get("timings"), dict) else {}

    graded: dict[str, Any] = {
        "pass": bool(raw.get("ok") or raw.get("graded_pass") or raw.get("strict_pass")),
        "strict_pass": bool(raw.get("strict_pass") or raw.get("graded_pass") or raw.get("ok")),
        "format_pass": bool(raw.get("format_pass")),
        "semantic_hint": bool(raw.get("semantic_hint")),
        "given": raw.get("given"),
        "correct": raw.get("correct"),
        "status": raw.get("status"),
    }
    # Drop None entries so equality-against-fresh-graded checks don't
    # trip on absent-vs-explicit-null differences.
    graded = {k: v for k, v in graded.items() if v is not None}

    return CanonicalRow(
        case_id=case_id,
        source=str(raw.get("source") or ""),
        kind=str(raw.get("kind") or ""),
        prompt_tokens=raw.get("prompt_tokens"),
        completion_tokens=raw.get("completion_tokens"),
        reasoning_tokens=raw.get("thinking_tokens"),
        ttft_seconds=raw.get("ttft_seconds"),
        wall_seconds=float(raw.get("wall_s") or raw.get("wall_seconds") or 0.0),
        prefill_ms=timings.get("prefill_ms") if isinstance(timings, dict) else None,
        decode_ms=timings.get("decode_ms") if isinstance(timings, dict) else None,
        decode_tokens_per_sec=(
            timings.get("decode_tokens_per_sec") if isinstance(timings, dict) else None
        ),
        content=str(raw.get("output") or raw.get("content") or ""),
        reasoning_content=raw.get("reasoning_content"),
        finish_reason=raw.get("finish_reason"),
        http_status=raw.get("http_status"),
        error=raw.get("error"),
        graded=graded,
    )


def _normalize_current_row(raw: dict[str, Any]) -> CanonicalRow:
    """Map a 0.2.5+ row dict to a CanonicalRow."""
    timings = raw.get("timings") if isinstance(raw.get("timings"), dict) else {}
    graded = dict(raw.get("graded") or {})

    return CanonicalRow(
        case_id=str(raw.get("case_id") or raw.get("id") or ""),
        source=str(raw.get("source") or ""),
        kind=str(raw.get("kind") or ""),
        prompt_tokens=raw.get("prompt_tokens"),
        completion_tokens=raw.get("completion_tokens"),
        reasoning_tokens=raw.get("reasoning_tokens"),
        ttft_seconds=raw.get("ttft_seconds"),
        wall_seconds=float(raw.get("wall_seconds") or raw.get("wall_s") or 0.0),
        prefill_ms=timings.get("prefill_ms") if isinstance(timings, dict) else None,
        decode_ms=timings.get("decode_ms") if isinstance(timings, dict) else None,
        decode_tokens_per_sec=(
            timings.get("decode_tokens_per_sec") if isinstance(timings, dict) else None
        ),
        content=str(raw.get("content") or raw.get("output") or ""),
        reasoning_content=raw.get("reasoning_content"),
        finish_reason=raw.get("finish_reason"),
        http_status=raw.get("http_status"),
        error=raw.get("error"),
        graded=graded,
    )


def _infer_area(raw: dict[str, Any], source_path: Path | None) -> str:
    """Best-effort area detection.

    Order: explicit top-level ``area`` field → file name (``ds4-eval.json``
    inside a sweep) → ``suite`` field (legacy ds4_eval writes
    ``"suite": "ds4-eval"``) → fallback ``"unknown"``.
    """
    area = raw.get("area")
    if isinstance(area, str) and area:
        return area
    if source_path is not None and source_path.suffix == ".json":
        stem = source_path.stem
        # Per-area sweep files: <out>/<name>/ds4-eval.json etc.
        if stem in {"ds4-eval", "code", "longctx", "agent", "agent_recorded",
                    "forge", "smoke", "gsm8k", "hellaswag", "truthfulqa-mc1"}:
            return stem
    suite = raw.get("suite")
    if isinstance(suite, str) and suite:
        return suite
    return "unknown"


def _infer_mode(raw: dict[str, Any]) -> str:
    """Map various 'were we thinking?' flags to think|nothink."""
    if "thinking_enabled" in raw:
        return "think" if raw.get("thinking_enabled") else "nothink"
    if "think" in raw:
        return "think" if raw.get("think") else "nothink"
    # No flag at all — default to nothink (the safer guess; most legacy
    # ds4_server runs were nothink).
    return "nothink"


def extract_host_from_props(props_json: dict[str, Any] | None) -> HostInfo | None:
    """Project ``/props.host`` JSON onto a HostInfo dataclass.

    Returns None when ``props_json`` is None, missing the host block, or
    the host block is JSON null. Tolerant of unexpected extra keys (the
    canonical field set lives in ``schema.HostInfo``).

    Always overwrites ``source`` to ``"props"`` — when the data crossed
    the wire via /props that IS the source from this consumer's view.
    The entrypoint's source label (typically ``"lucebox.sh"``) describes
    how the SERVER got the data and is preserved in ``collector``.
    """
    if not isinstance(props_json, dict):
        return None
    raw = props_json.get("host")
    if not isinstance(raw, dict):
        return None
    host = host_from_dict(raw)
    if host is not None:
        # Stamp source="props"; preserve the upstream's identification
        # in `collector` (older lucebox builds wrote only `source`,
        # real entrypoint.sh writes both — promote source→collector
        # when collector is missing, then overwrite source).
        as_dict = host.to_dict()
        if not as_dict.get("collector") and as_dict.get("source"):
            as_dict["collector"] = as_dict["source"]
        as_dict["source"] = "props"
        host = host_from_dict(as_dict)
    return host


def _has_real_semantic_judge(raw: dict[str, Any]) -> bool:
    """True only when ``metrics.semantic_judge`` has actual content.

    The legacy ``semantic_pass_rate`` top-level field is always 0.0 and
    must NOT be treated as a real semantic-judge result. A real judge
    would be plumbed under ``metrics["semantic_judge"][<judge_id>]``.
    """
    metrics = raw.get("metrics")
    if not isinstance(metrics, dict):
        return False
    judges = metrics.get("semantic_judge")
    return isinstance(judges, dict) and bool(judges)


def normalize_result(
    raw: dict[str, Any], *, source_path: Path | None = None
) -> CanonicalResult:
    """Load a luce-bench result dict into the canonical schema.

    Detects legacy vs. current shape and remaps. Pass-rate fields are
    always returned as fractions in [0.0, 1.0]; the unit interpretation
    is recorded in ``metrics["pass_rate_unit"]`` for audit.

    Parameters
    ----------
    raw:
        The deserialised result.json contents.
    source_path:
        Optional path the file came from. Used only to infer the area
        when the dict doesn't carry an explicit ``area`` field (e.g.
        sweep per-area files named ``ds4-eval.json``).
    """
    legacy = _looks_like_legacy(raw)
    rows_raw = raw.get("rows") or raw.get("results") or []

    if legacy:
        rows = [_normalize_legacy_row(r) for r in rows_raw]
    else:
        rows = [_normalize_current_row(r) for r in rows_raw]

    n = len(rows)

    # Recompute the rate fields straight from the rows so we don't trust
    # the writer's possibly-wrong-unit aggregates. (Legacy results often
    # ALSO ship explicit strict_pass_rate/format_pass_rate alongside
    # pass_rate — we still recompute from rows for consistency.)
    strict_passes = sum(
        1
        for r in rows
        if r.graded.get("strict_pass")
        or r.graded.get("pass")  # current shape uses just "pass"
    )
    format_passes = sum(1 for r in rows if r.graded.get("format_pass"))
    hints = sum(1 for r in rows if r.graded.get("semantic_hint"))

    strict_rate = strict_passes / n if n else 0.0
    format_rate = format_passes / n if n else 0.0
    hint_rate = hints / n if n else 0.0

    # Surface what the writer originally said so we can flag drift in
    # the regrade comparison. Detect-and-normalise the unit too. When
    # the writer didn't declare anything (synthetic test fixtures, sweep
    # per-area JSONs missing the aggregate field, etc.) fall back to the
    # row-derived strict_rate so the regrade Δ stays an apples-to-apples
    # "before vs after" rather than the misleading "we said 0.0".
    raw_pass_rate = raw.get("pass_rate")
    if raw_pass_rate is None:
        raw_pass_rate = raw.get("strict_pass_rate")
    if raw_pass_rate is None:
        declared_rate, rate_unit = strict_rate, "row_derived"
    else:
        declared_rate, rate_unit = _normalize_pass_rate(raw_pass_rate)

    # Mode + thinking-control verification. When the writer (current
    # lucebench post-run) already stamped `thinking_control_honored`
    # we trust it. For historical files that predate the verifier we
    # re-run the same shared check (``_thinking.verify_thinking_control``)
    # against the loaded rows so old OpenRouter Qwen3.6 baselines pick
    # up the same honored=False signal that fresh runs do — with the
    # same 5% slack threshold.
    mode = _infer_mode(raw)
    requested = mode
    if "thinking_control_honored" in raw:
        honored = bool(raw.get("thinking_control_honored", True))
        contradicting = int(raw.get("contradicting_rows") or 0)
    else:
        verify_rows = [
            {
                "reasoning_tokens": r.reasoning_tokens,
                "reasoning_content": r.reasoning_content,
            }
            for r in rows
        ]
        honored, contradicting = verify_thinking_control(verify_rows, requested)

    # Carry over `metrics` if the file already had one, otherwise build
    # a fresh dict. NOTE we drop the dead semantic_pass_rate fields
    # unless a real judge is plumbed (almost never on historical files).
    metrics: dict[str, Any] = dict(raw.get("metrics") or {})
    metrics["pass_rate_unit"] = rate_unit
    metrics["declared_strict_pass_rate"] = declared_rate
    if not _has_real_semantic_judge(raw):
        # The top-level semantic_pass_rate / semantic_passed were always
        # 0.0 in shipped legacy files (no judge plumbed). Drop them so
        # consumers don't accidentally treat 0.0 as "the semantic score
        # crashed". A future real judge would live under
        # metrics["semantic_judge"][<judge_id>].
        for dead in ("semantic_pass_rate", "semantic_passed"):
            metrics.pop(dead, None)

    # Host block. Three sources, in priority order:
    #   1. Explicit ``host`` field on the input dict (lucebench snapshot
    #      writes this into each per-area JSON so the file self-describes
    #      after being moved).
    #   2. Nested ``server_info.host`` / ``server_info.props.host`` for
    #      result.json shapes that carried the props snapshot inline.
    #   3. Historical files that pre-date schema v2 → stub
    #      ``HostInfo(source="unknown")`` so consumers can still tell
    #      "no host context" apart from "different host context".
    host: HostInfo | None = None
    raw_host = raw.get("host")
    if isinstance(raw_host, dict):
        host = host_from_dict(raw_host)
    elif isinstance(raw.get("server_info"), dict):
        si = raw["server_info"]
        if isinstance(si.get("host"), dict):
            host = host_from_dict(si["host"])
        elif isinstance(si.get("props"), dict):
            host = extract_host_from_props(si["props"])
    if host is None:
        host = HostInfo(source="unknown")

    return CanonicalResult(
        schema_version=SCHEMA_VERSION,
        lucebench_version=str(raw.get("lucebench_version") or ""),
        grader_version=str(raw.get("grader_version") or ""),
        area=_infer_area(raw, source_path),
        model=str(raw.get("model") or ""),
        quant=raw.get("quant"),
        serving_url=str(raw.get("url") or raw.get("serving_url") or ""),
        mode=mode,  # type: ignore[arg-type]
        seed=raw.get("seed"),
        n=n,
        started_at=str(raw.get("started_at") or raw.get("timestamp") or ""),
        thinking_control_requested=requested,
        thinking_control_honored=honored,
        contradicting_rows=contradicting,
        strict_pass_rate=strict_rate,
        format_pass_rate=format_rate,
        semantic_hint_rate=hint_rate,
        metrics=metrics,
        rows=rows,
        host=host,
    )


def load_result(path: Path) -> CanonicalResult:
    """Read a JSON file from disk and normalise it.

    Thin convenience wrapper around ``normalize_result`` for callers
    that have a path rather than the already-deserialised dict.
    """
    raw = json.loads(path.read_text())
    return normalize_result(raw, source_path=path)


__all__ = [
    "load_result",
    "normalize_result",
    "extract_host_from_props",
    "from_dict",  # re-export so callers don't need both modules
]
