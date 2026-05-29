"""Tiny smoke-test prompts for `--areas smoke` — the default "is the server
responding sensibly?" sanity check.

Three short prompts (arithmetic, literal echo, sequence continuation) that
together complete in a few seconds against any reasonable chat-completion
endpoint. Grading is intentionally permissive: extract the model's content,
check that the expected substring appears somewhere in it (case-insensitive
for the "OK" prompt). The goal is *binary "the server is alive and
producing text"*, not capability scoring — for that, use `--areas all`.

This area exists so the bare command::

    lucebench --url http://localhost:1236

does something useful in a few seconds instead of either erroring on a
missing `--area` or kicking off a 92-case sweep. The CLI defaults
`--areas` to `smoke` for that reason.
"""

from __future__ import annotations

from typing import Any

# See lucebench.areas.ds4_eval.GRADER_VERSION for the bump policy.
GRADER_VERSION = 1

# Three tiny, deterministic prompts. Each pairs the user-visible question
# with the substring the model's reply must contain. Two rules for picking
# expected substrings:
#
#   (1) The expected substring must NOT already appear in the prompt — a
#       thinking-mode model that echoes the prompt while reasoning would
#       otherwise pass vacuously. (Original "Reply with the word OK" was
#       this exact trap.)
#
#   (2) The grader checks both content and reasoning_content (some servers
#       route the trained-thinking trace to reasoning_content and leave
#       content empty if max_tokens trips mid-think).
_SMOKE_CASES: list[dict[str, Any]] = [
    {
        "id": "smoke-arithmetic",
        "prompt": "What is two plus two? Reply with just the digit.",
        "expected": "4",
    },
    {
        "id": "smoke-capital",
        "prompt": "Capital of France? Reply with just the city name.",
        "expected": "Paris",
    },
    {
        "id": "smoke-sequence",
        "prompt": "Continue this sequence with the next number only: 1, 2,",
        "expected": "3",
    },
]


def load_smoke_cases() -> list[dict[str, Any]]:
    """Return the three smoke cases shaped for the lucebench runner.

    Each case carries the canonical fields the runner / grader rely on:
    ``area``, ``source``, ``id``, ``kind``, ``prompt``, ``answer``, plus
    the local ``expected`` substring used by ``grade_smoke_case``.
    """
    out: list[dict[str, Any]] = []
    for raw in _SMOKE_CASES:
        out.append(
            {
                "area": "smoke",
                "source": "smoke",
                "id": raw["id"],
                "kind": "smoke",
                "prompt": raw["prompt"],
                "answer": raw["expected"],
                "expected": raw["expected"],
                "domain": "smoke",
                "title": raw["id"],
            }
        )
    return out


def grade_smoke_case(case: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    """Substring-match against ``case["expected"]`` — case-insensitive,
    checks both ``content`` and ``reasoning_content``.

    Many servers route a model's trained-thinking trace to
    ``reasoning_content`` and only emit visible text in ``content`` after
    the model self-closes ``</think>``. When ``max_tokens`` is tight (or
    the server's chat template forces thinking on), the answer can end
    up in either field. Smoke is a *binary "is the server alive"* gate,
    so a match anywhere counts.

    Returns the standard grader shape:
    ``{pass, given, correct, status, format_pass, semantic_hint}``.
    """
    expected = str(case.get("expected") or case.get("answer") or "").strip()
    content = (row.get("content") or "").strip()
    reasoning = (row.get("reasoning_content") or "").strip()
    haystack = (content + "\n" + reasoning).lower()

    ok = expected.lower() in haystack
    format_pass = bool(content) or bool(reasoning)

    # First 80 chars of whichever field has text. Keeps the printed
    # row readable; surfaces "?" when the model went totally silent.
    given_source = content or reasoning
    given = (given_source[:80] if given_source else "?") or "?"
    return {
        "pass": ok,
        "given": given,
        "correct": expected,
        "status": "passed" if ok else ("format_error" if not format_pass else "failed"),
        "format_pass": format_pass,
        "semantic_hint": ok,
    }
