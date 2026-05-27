"""antirez/ds4 ``ds4_eval`` corpus + grading port.

This module is the colocation point for everything we lifted from
``antirez/ds4 ds4_eval.c`` so a future diff against upstream stays
narrow. Anything ds4-specific — case loader, answer extractors, scoring
helpers, the published evaluation budgets — lives here, not in the
generic capability bench.

Keep the structure close to ds4_eval.c when possible:

  * `DS4_EVAL_MAX_TOKENS` mirrors ds4_eval.c's `max_tokens` default
    (16000) — the combined cap covering reasoning + reply. The thinking
    budget split is server-side configuration (`--think-max-tokens`),
    not wire protocol, so cross-server comparisons stay clean.
  * `find_ds4_choice_answer` / `find_ds4_integer_answer` mirror ds4's
    permissive answer hunt — look for a literal "answer" marker first
    and accept the next valid letter/integer; fall back to the last
    valid one if the marker is missing.
  * `normalize_compsec_line_spec` matches ds4's COMPSEC line-range
    normalisation (range tokens collapsed, whitespace stripped).
  * `compsec_answer_matches` accepts any subset of the expected line
    set (ds4's grading semantics for partial COMPSEC answers).

The companion fixture at ``dflash/scripts/fixtures/ds4_eval_cases.json``
was generated from ds4_eval.c's embedded `eval_cases` table; re-export
that file from upstream when ds4 ships new cases.

Consumers (`bench_http_capability.py`, `lucebox_bench.py`,
`lucebox.profile`) import from this module rather than rolling their
own ds4 grading — keeping a single source of truth for what counts as
a ds4 pass.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures"
DS4_EVAL_CASES_PATH = FIXTURE_DIR / "ds4_eval_cases.json"
DS4_SOURCES = {"GPQA Diamond", "SuperGPQA", "AIME2025", "COMPSEC"}

# Combined cap from antirez/ds4 ds4_eval.c (`.max_tokens = 16000`). The
# bench sends this as the standard OpenAI `max_tokens`; each server applies
# its own configured thinking-budget split internally (`--think-max-tokens`
# on dflash, no split at all on stock ds4_server). Bump this if upstream
# bumps its default so cross-machine quality numbers stay comparable.
DS4_EVAL_MAX_TOKENS = 16000


def load_ds4_eval_cases(path: Path = DS4_EVAL_CASES_PATH) -> list[dict[str, Any]]:
    """Load the ported ds4_eval.c eval_cases JSON.

    Each row gets ``area = "ds4-eval"`` and ``ds4_eval = True`` so the
    capability dispatch can route it through the ds4 graders. The
    upstream ``index`` field is renamed to ``ds4_index`` so it doesn't
    collide with pytest's parametrize id convention or the row-numbering
    in trace output.
    """
    payload = json.loads(path.read_text())
    rows: list[dict[str, Any]] = []
    for raw in payload["cases"]:
        case = dict(raw)
        case["area"] = "ds4-eval"
        case["ds4_eval"] = True
        case["ds4_index"] = case.pop("index")
        rows.append(case)
    return rows


# Eager-loaded for cheap dispatch — the fixture is ~120 KB and parsing
# costs a few ms. Re-import the module to pick up fixture edits.
DS4_EVAL_CASES = load_ds4_eval_cases()


def visible_text(generated: str) -> str:
    """Strip the model's thinking prefix.

    Mirrors `visible_text` in bench_http_capability so this module can
    be used without round-tripping back through it. Anything after the
    first ``</think>`` is the visible reply; if the model never closed
    thinking we return the raw stream and let the extractors hunt for
    an answer marker anywhere.
    """
    close = generated.find("</think>")
    if close >= 0:
        return generated[close + len("</think>") :]
    return generated


def is_letter_boundary(before: str, after: str) -> bool:
    return not before.isalpha() and not after.isalpha()


def find_ds4_choice_answer(generated: str, nchoices: int) -> str:
    """Permissive choice-letter extractor matching ds4_eval.c.

    Strategy: find the literal "answer" keyword, then accept the next
    valid letter in a 96-character window. If no marker exists, fall
    back to the LAST valid letter in the whole text (ds4 plays loose
    with format so we accept anything that looks like a final letter).
    """
    if nchoices <= 0:
        return "?"
    text = visible_text(generated)
    max_answer = chr(ord("A") + nchoices - 1)
    answer = re.search(r"answer", text, flags=re.IGNORECASE)
    if answer:
        window = text[answer.start() : answer.start() + 96]
        for idx, char in enumerate(window):
            candidate = char.upper()
            if "A" <= candidate <= max_answer:
                before = window[idx - 1] if idx > 0 else " "
                after = window[idx + 1] if idx + 1 < len(window) else " "
                if is_letter_boundary(before, after):
                    return candidate

    for idx in range(len(text) - 1, -1, -1):
        candidate = text[idx].upper()
        if "A" <= candidate <= max_answer:
            before = text[idx - 1] if idx > 0 else " "
            after = text[idx + 1] if idx + 1 < len(text) else " "
            if is_letter_boundary(before, after):
                return candidate
    return "?"


def normalize_integer(value: str) -> str:
    sign = "-" if value.startswith("-") else ""
    digits = value[1:] if sign else value
    digits = digits.lstrip("0") or "0"
    return sign + digits


def find_ds4_integer_answer(generated: str) -> str:
    """Permissive integer extractor matching ds4_eval.c.

    Same shape as the choice extractor: prefer the first digit run after
    a literal "answer" marker; fall back to the last digit run anywhere.
    Leading zeros normalised so "0042" and "42" compare equal.
    """
    text = visible_text(generated)
    answer = re.search(r"answer", text, flags=re.IGNORECASE)
    if answer:
        window = text[answer.start() : answer.start() + 160]
        match = re.search(r"\d+", window)
        if match:
            return normalize_integer(match.group(0))

    matches = list(re.finditer(r"\d+", text))
    if matches:
        return normalize_integer(matches[-1].group(0))
    return "?"


def normalize_compsec_line_spec(text: str) -> str:
    """Collapse a COMPSEC line-spec to canonical form.

    "lines 17, 18 - 20 and 22" → "17,18-20,22". Range tokens preserved;
    whitespace inside ranges stripped. Returns "?" on no match.
    """
    parts: list[str] = []
    for match in re.finditer(r"\d+(?:\s*-\s*\d+)?", text):
        parts.append(re.sub(r"\s+", "", match.group(0)))
    return ",".join(parts) if parts else "?"


def find_compsec_answer(generated: str) -> str:
    """COMPSEC line-spec extractor: marker line first, then permissive."""
    text = visible_text(generated)
    answer = re.search(r"answer", text, flags=re.IGNORECASE)
    if answer:
        window = text[answer.start() : answer.start() + 160]
        window = window.splitlines()[0]
        got = normalize_compsec_line_spec(window)
        if got != "?":
            return got
    return find_ds4_integer_answer(generated)


def parse_line_spec(spec: str) -> set[int]:
    values: set[int] = set()
    for match in re.finditer(r"\d+(?:\s*-\s*\d+)?", spec):
        raw = match.group(0)
        if "-" in raw:
            left, right = raw.split("-", 1)
            start = int(left.strip())
            end = int(right.strip())
            if start > end:
                start, end = end, start
            values.update(range(start, end + 1))
        else:
            values.add(int(raw))
    return values


def compsec_answer_matches(expected_spec: str, got_spec: str) -> bool:
    """ds4-style COMPSEC partial-credit grader.

    Pass if the model's lines are a non-empty subset of the expected
    lines. Lets the model name e.g. line 18 out of expected 17-20 and
    still count as correct — matches ds4_eval.c's tolerance.
    """
    expected = parse_line_spec(expected_spec)
    got = parse_line_spec(got_spec)
    return bool(got) and got.issubset(expected)


def is_ds4_eval_case(case: dict[str, Any]) -> bool:
    """True when this case came from the ds4-eval corpus.

    Routes the capability bench's `find_answer` to the ds4 graders
    instead of the smoke graders. Set explicitly via the `ds4_eval`
    flag (set by load_ds4_eval_cases), with a source-name fallback in
    case a caller hand-builds a case dict.
    """
    return bool(case.get("ds4_eval")) or case.get("source") in DS4_SOURCES


# ──────────────────────────────────────────────────────────────────────
# Grading layer (extracted from luce-dflash bench_http_capability.py
# `grade_case`). Returns the shape the lucebench CLI expects:
#
#   {pass: bool, given: str, correct: str, status: str, format_pass: bool,
#    semantic_hint: bool}
#
# Pure ds4 semantics: a strict pass requires the model to write the
# canonical "Answer: <X>" line. Mid-stream mentions count as a semantic
# hint but not a pass.
# ──────────────────────────────────────────────────────────────────────


def expected_answers(case: dict[str, Any]) -> list[str]:
    raw = case["answer"]
    if isinstance(raw, list):
        return [str(item) for item in raw]
    return [str(raw)]


def find_answer(case: dict[str, Any], generated: str) -> str:
    """Permissive answer extractor matching antirez/ds4 ds4_eval.c."""
    if case.get("kind") == "choice":
        return find_ds4_choice_answer(generated, len(case.get("choices") or []))
    if case.get("kind") == "compsec":
        return find_compsec_answer(generated)
    return find_ds4_integer_answer(generated)


def _semantic_hint(case: dict[str, Any], content: str, reasoning_content: str | None) -> bool:
    """Did the model mention an expected answer anywhere (incl. reasoning)?

    Diagnostic only; not used for ds4-eval pass/fail (comparability
    requires the same final-answer grading semantics).
    """
    if case.get("kind") == "choice":
        return find_answer(case, content) in expected_answers(case)
    text = visible_text(content)
    if reasoning_content:
        text += "\n" + visible_text(reasoning_content)
    if case.get("kind") == "compsec":
        expected_lines = parse_line_spec(",".join(expected_answers(case)))
        found_lines = parse_line_spec(text)
        return bool(expected_lines & found_lines)
    expected = {str(int(answer)) for answer in expected_answers(case)}
    found = {
        str(int(m.group(0)))
        for m in re.finditer(r"-?\d+", text)
        if len(m.group(0).lstrip("-")) <= 20
    }
    return bool(expected & found)


def grade_case(case: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    """Grade a row from lucebench.runner.run_case.

    Returns a dict with at least {pass, given, correct, status, format_pass,
    semantic_hint}.
    """
    content = row.get("content") or ""
    reasoning = row.get("reasoning_content")
    got = find_answer(case, content)
    expected = expected_answers(case)
    format_pass = got != "?"
    if case.get("kind") == "compsec":
        strict_pass = any(compsec_answer_matches(ans, got) for ans in expected)
    else:
        strict_pass = got in expected
    hint = _semantic_hint(case, content, reasoning)
    return {
        "pass": strict_pass,
        "given": got,
        "correct": ",".join(expected),
        "status": "passed" if strict_pass else ("format_error" if not format_pass else "failed"),
        "format_pass": format_pass,
        "semantic_hint": hint,
    }
