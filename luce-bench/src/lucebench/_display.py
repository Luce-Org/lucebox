"""Terminal-output formatting for the lucebench CLI.

Pure string builders — no I/O, no network — split out of cli.py so the
per-column display conventions (the ``*``-suffix markers that distinguish
server-reported metrics from wall-clock fallbacks) can be unit-tested in
isolation. ``format_row`` is the public entry point; the ``_format_*``
helpers each own one column group.
"""

from __future__ import annotations

# Below this completion-token count the wall-clock throughput estimate is
# dominated by router/first-token latency rather than decode, so we drop the
# rate column entirely instead of printing a misleading "0tps*".
_FALLBACK_MIN_TOKENS = 8


def _fmt_tps(v: float) -> str:
    # Sub-10tps renders with one decimal so 0.3 doesn't round down to 0.
    if v < 10:
        return f"{v:.1f}"
    return f"{v:.0f}"


def _fmt_ms(ms: float) -> str:
    # Sub-second renders as e.g. "210ms"; >=1s as "3.5s" to keep the line tight.
    if ms < 1000:
        return f"{ms:.0f}ms"
    return f"{ms / 1000:.1f}s"


def _format_throughput(row: dict, timings: dict, wall: float) -> str:
    """Render the throughput token (``Ntps`` / ``Ntps*`` / "").

    Prefers the server-reported decode rate (lucebox / llama.cpp populate
    ``decode_tokens_per_sec``); falls back to a wall-clock estimate marked
    with a trailing ``*`` (it rolls prefill into the rate). Returns "" when
    there's no usable rate or too few tokens to be meaningful.
    """
    tps_val = timings.get("decode_tokens_per_sec")
    completion_tokens = row.get("completion_tokens")
    if tps_val:
        return f"{_fmt_tps(tps_val)}tps"
    if (
        completion_tokens
        and isinstance(completion_tokens, int)
        and completion_tokens >= _FALLBACK_MIN_TOKENS
        and wall
        and wall > 0
    ):
        return f"{_fmt_tps(completion_tokens / wall)}tps*"
    return ""


def _format_timing(row: dict, timings: dict, wall: float) -> str:
    """Render the timing token (prefill/decode/ttft/wall).

    Server-reported ``prefill_ms`` is preferred over the streaming
    wall-clock TTFT; the latter is marked with ``*`` when it's all we have.
    Falls back to plain wall time when no split is available.
    """
    prefill_ms = timings.get("prefill_ms")
    decode_ms = timings.get("decode_ms")
    ttft_seconds = row.get("ttft_seconds")
    ttft_ms: float | None = (
        ttft_seconds * 1000 if isinstance(ttft_seconds, int | float) else None
    )

    if prefill_ms is not None and decode_ms is not None:
        return f"prefill={_fmt_ms(prefill_ms)} decode={_fmt_ms(decode_ms)}"
    if prefill_ms is not None:
        return f"prefill={_fmt_ms(prefill_ms)} wall={wall:.2f}s"
    if ttft_ms is not None and decode_ms is not None:
        return f"ttft={_fmt_ms(ttft_ms)}* decode={_fmt_ms(decode_ms)}"
    if ttft_ms is not None:
        return f"ttft={_fmt_ms(ttft_ms)}* wall={wall:.2f}s"
    if decode_ms is not None:
        return f"decode={_fmt_ms(decode_ms)} wall={wall:.2f}s"
    return f"wall={wall:.2f}s"


def _format_tokens(row: dict) -> str:
    """Render the token breakdown (``in=`` / ``think=`` / ``out=``).

    No tokenizer dependency — we only echo server-supplied counts. When
    ``reasoning_tokens`` is present we split the completion into thinking vs
    non-thinking; otherwise ``out=`` carries the whole completion count.
    """
    prompt_tokens = row.get("prompt_tokens")
    completion_tokens = row.get("completion_tokens")
    reasoning_tokens = row.get("reasoning_tokens")
    tok_bits: list[str] = []
    if prompt_tokens is not None:
        tok_bits.append(f"in={prompt_tokens}")
    if isinstance(reasoning_tokens, int) and isinstance(completion_tokens, int):
        non_thinking = max(completion_tokens - reasoning_tokens, 0)
        tok_bits.append(f"think={reasoning_tokens}")
        tok_bits.append(f"out={non_thinking}")
    elif completion_tokens is not None:
        tok_bits.append(f"out={completion_tokens}")
    return " ".join(tok_bits)


def format_row(idx: int, row: dict, graded: dict) -> str:
    """Format one graded result row as a single console line."""
    src = row.get("source") or "?"
    cid = row.get("case_id") or "?"
    verdict = "PASS" if graded.get("pass") else "FAIL"
    given = graded.get("given") or "?"
    correct = graded.get("correct") or "?"
    wall = row.get("wall_seconds") or 0
    timings = row.get("timings") or {}
    if not isinstance(timings, dict):
        timings = {}

    tps_str = _format_throughput(row, timings, wall)
    time_str = _format_timing(row, timings, wall)
    tok_str = _format_tokens(row)

    tail_bits = [time_str]
    if tps_str:
        tail_bits.append(tps_str)
    if tok_str:
        tail_bits.append(tok_str)
    return (
        f"  {idx:3d} {verdict} {src:14s} {cid:24s} "
        f"given={given:20s} correct={correct:20s} " + " ".join(tail_bits)
    )


def _format_models_inline(ids: list[str], selected: str, budget: int = 62) -> str:
    """Render a comma-separated `/v1/models` listing for the preflight grid.

    Marks the chosen id with a `*` prefix. If the full list fits in
    `budget` characters, it's shown verbatim. Otherwise the layout is:
    first model, then the selected model (if different), then sequential
    fillers until the budget is hit, ending with `… (+N more)`.
    """
    if not ids:
        return "(none)"

    def render(picked_idx: list[int], remaining: int) -> str:
        parts = [(f"*{ids[i]}" if ids[i] == selected else ids[i]) for i in picked_idx]
        s = ", ".join(parts)
        if remaining:
            s += f", … (+{remaining} more)"
        return s

    full = render(list(range(len(ids))), 0)
    if len(full) <= budget:
        return full

    picked = [0]
    if selected in ids and ids[0] != selected:
        picked.append(ids.index(selected))
    for i in range(1, len(ids)):
        if i in picked:
            continue
        candidate = sorted(picked + [i])
        remaining = len(ids) - len(candidate)
        if len(render(candidate, remaining)) > budget:
            break
        picked = candidate
    remaining = len(ids) - len(picked)
    return render(sorted(picked), remaining)
