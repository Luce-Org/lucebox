"""GSM8K grade-school math word problems — `--areas gsm8k`.

100-case sample of the canonical ``openai/gsm8k`` test split (MIT
license), sampled with ``random.Random(42)`` and vendored as JSONL so
the runtime has no ``datasets`` dependency. Each upstream row has the
CoT walkthrough plus a ``#### <number>`` final-answer marker; the
fixture preserves the full ``upstream_answer`` for transparency but the
``expected`` field is just the numeric answer.

Prompting is intentionally **0-shot, no system prompt** — the canonical
GSM8K eval ships a 5-shot CoT prefix, but we want to measure raw model
behavior on the bench. Users who care about CoT pass ``--think`` and
the server (lucebox / dflash) will inject its own thinking trace.

Grading mirrors the upstream marker convention: first try
``r"####\\s*(-?[\\d,]+(?:\\.\\d+)?)"`` on the model's reply, fall back
to the last number in the response. Compare normalised (commas stripped)
as floats with a 1e-6 tolerance so "18", "18.0", and "18.00" all pass.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

# See lucebench.areas.ds4_eval.GRADER_VERSION for the bump policy.
GRADER_VERSION = 1

FIXTURE_PATH = (
    Path(__file__).resolve().parent.parent / "fixtures" / "gsm8k" / "cases.jsonl"
)

# 0-shot GSM8K against modern instruct models routinely walks through
# the CoT before emitting `#### N`. A 512 cap was truncating ~15-20% of
# the longest problems mid-reasoning on gemma-4-26b — those runs scored
# as FAIL despite the model being on the right track. Bench should
# measure model capability, not the cap, so we widen to 2048. Hard
# arithmetic problems with verbose CoT still fit; anything that
# genuinely exceeds 2048 tokens is a separate signal worth seeing
# (and `--max-tokens` is available for one-off overrides).
GSM8K_MAX_TOKENS = 2048

# Canonical GSM8K final-answer marker — `#### 18`, `#### -3,500`,
# `#### 0.5`, etc. Allows comma group separators and an optional
# decimal tail. Anchored to the literal `####` prefix so it doesn't
# false-match arbitrary "####" hashes inside the reasoning trace.
_MARKER_RE = re.compile(r"####\s*(-?[\d,]+(?:\.\d+)?)")

# Permissive fallback: any signed number, integer or decimal. Used when
# the model didn't emit the canonical marker — we then pick the LAST
# match in the response (final answer convention).
_NUMBER_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def load_gsm8k_cases(path: Path = FIXTURE_PATH) -> list[dict[str, Any]]:
    """Load the vendored GSM8K case set (JSONL).

    Each case carries the canonical fields the runner / grader rely on:
    ``area``, ``source``, ``id``, ``kind``, ``prompt``, ``answer``,
    plus ``expected`` (the numeric answer as a string) and
    ``upstream_answer`` (the full CoT-with-marker upstream string,
    kept for transparency).
    """
    out: list[dict[str, Any]] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            out.append(
                {
                    "area": "gsm8k",
                    "source": raw.get("source", "gsm8k"),
                    "id": raw["id"],
                    "kind": "math-word-problem",
                    "prompt": raw["prompt"],
                    "answer": raw["expected"],
                    "expected": raw["expected"],
                    "upstream_answer": raw.get("upstream_answer"),
                    "domain": "math",
                    "title": raw["id"],
                }
            )
    return out


def _normalize_number(s: str) -> str:
    return s.replace(",", "").strip()


def extract_gsm8k_answer(text: str) -> str | None:
    """Pull the model's final answer out of ``text``.

    Two-pass extractor:

    1. **Canonical marker** — ``r"####\\s*(-?[\\d,]+(?:\\.\\d+)?)"``.
       If the model followed the upstream format we trust the marker
       absolutely; later mentions of numbers (e.g. in a follow-on
       sentence) are ignored.
    2. **Last-number fallback** — many models drop the marker entirely
       and just emit "The answer is 18" / "18 dollars". We take the
       LAST signed number anywhere in the reply.

    Returns the normalised numeric string (commas stripped) or ``None``
    when no number is present.
    """
    if not text:
        return None
    m = _MARKER_RE.search(text)
    if m:
        return _normalize_number(m.group(1))
    matches = list(_NUMBER_RE.finditer(text))
    if matches:
        return _normalize_number(matches[-1].group(0))
    return None


def _to_float(s: str) -> float | None:
    try:
        return float(s)
    except (TypeError, ValueError):
        return None


def grade_gsm8k_case(case: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    """Grade a GSM8K row produced by ``lucebench.runner.run_case``.

    Pass = the extracted answer parses as a float within ``1e-6`` of
    the expected float. Empty / non-numeric replies are ``format_error``
    rather than ``failed`` so the sweep summary can tell "model produced
    no number" apart from "model produced a wrong number".
    """
    expected = str(case.get("expected") or case.get("answer") or "").strip()
    content = row.get("content") or ""
    reasoning = row.get("reasoning_content") or ""
    # GSM8K answers live in the visible reply, not the reasoning trace —
    # but if the model never closed `</think>` (e.g. ran out of tokens
    # mid-CoT), the answer may end up in reasoning_content instead.
    # Match either, content first.
    haystack = content if content.strip() else reasoning
    given = extract_gsm8k_answer(haystack)

    expected_f = _to_float(expected)
    given_f = _to_float(given) if given is not None else None

    format_pass = given is not None
    strict_pass = (
        format_pass
        and expected_f is not None
        and given_f is not None
        and abs(given_f - expected_f) < 1e-6
    )

    return {
        "pass": strict_pass,
        "given": given if given is not None else "?",
        "correct": expected,
        "status": "passed" if strict_pass else ("format_error" if not format_pass else "failed"),
        "format_pass": format_pass,
        "semantic_hint": strict_pass,
    }
