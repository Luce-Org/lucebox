"""Shared multiple-choice helpers — used by ``truthfulqa_mc1`` and
``hellaswag``.

Both areas ship the same prompt shape ("question / context + N letter-
labelled choices, reply with the answer letter") and the same grader
contract (extract a single letter, compare to expected). The omlx
survey called the extractor ``_extract_mc_answer``; this is the ported
version, adapted to:

  * Accept a dynamic letter range (TruthfulQA cases have 2–13 choices).
  * Look for the canonical phrasings first ("answer is X", "Answer: X",
    "(X)"), then fall back to the LAST standalone letter in a 1..N
    range — matches the convention dlt models actually follow.
  * Strip ``</think>`` reasoning blocks so a model that thinks-then-
    answers doesn't get graded on a stray letter in the trace.
"""

from __future__ import annotations

import re

# Bump in lockstep across the consumers (truthfulqa_mc1, hellaswag) so a
# regrade can detect drift. See lucebench.areas.ds4_eval.GRADER_VERSION.
GRADER_VERSION = 1

# Phrases that mark "this is the final answer" — checked in order of
# decreasing specificity. Each is a (regex, capture-group-index) pair.
# The patterns intentionally match a SINGLE letter in [A-Z] and let the
# caller validate that the letter falls within the case's actual range
# (some cases have only 2 choices, in which case "G" is junk even if
# the model emitted it).
_ANSWER_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\banswer\s*(?:is|:)\s*\(?([A-Z])\)?", re.IGNORECASE),
    re.compile(r"\bfinal\s+answer\s*:?\s*\(?([A-Z])\)?", re.IGNORECASE),
    re.compile(r"\bthe\s+correct\s+(?:answer|choice|option)\s*(?:is|:)\s*\(?([A-Z])\)?", re.IGNORECASE),
    re.compile(r"\(([A-Z])\)\s*$"),  # trailing "(X)"
    re.compile(r"^\s*\(?([A-Z])\)?\s*$", re.MULTILINE),  # standalone letter on its own line
)


def _strip_think(text: str) -> str:
    """Drop reasoning traces. Mirrors ``ds4_eval.visible_text`` but the
    helper there is private to that module and uses a different policy
    (returns the raw text when ``</think>`` is missing); we instead
    return the post-``</think>`` text or the original text if no close
    tag appears.
    """
    close = text.find("</think>")
    if close >= 0:
        return text[close + len("</think>") :]
    return text


def extract_mc_answer(text: str, nchoices: int) -> str | None:
    """Pull a single letter answer out of ``text``.

    ``nchoices`` is the case's actual choice count — letters returned
    are guaranteed to fall in ``A..chr(ord('A')+nchoices-1)``. Returns
    ``None`` if nothing matches.

    Strategy:

    1. Strip everything before ``</think>`` so a model that thinks
       out loud doesn't get scored on a stray "B" in the trace.
    2. Walk ``_ANSWER_PATTERNS`` in order; on a match, validate the
       letter is in range; if not, keep trying.
    3. Fallback: take the LAST standalone uppercase letter in the
       visible text. Real-world models often just reply with the
       letter and nothing else.
    """
    if not text or nchoices <= 0:
        return None
    if nchoices > 26:
        nchoices = 26  # cap at A..Z
    max_letter = chr(ord("A") + nchoices - 1)
    visible = _strip_think(text)

    def _in_range(letter: str) -> bool:
        return "A" <= letter <= max_letter

    # Pattern-based extraction — last match wins so we get the model's
    # final answer rather than a mention earlier in the reply.
    for pat in _ANSWER_PATTERNS:
        matches = list(pat.finditer(visible))
        for m in reversed(matches):
            letter = m.group(1).upper()
            if _in_range(letter):
                return letter

    # Last-letter fallback. Letter must be on a word boundary so we
    # don't grab the "I" out of "I think the answer is unclear".
    last_letter: str | None = None
    for m in re.finditer(r"\b([A-Z])\b", visible):
        letter = m.group(1)
        if _in_range(letter):
            last_letter = letter
    return last_letter


def build_mc_prompt(question: str, choices: list[str], *, prefix: str | None = None) -> str:
    """Render the user-message text for a multiple-choice case.

    ``prefix`` lets the caller swap "Question:" for "Context:" (used by
    HellaSwag — its prompts are sentence completions, not questions).
    The footer asks for the answer letter only; downstream grader is
    permissive but the prompt nudges the model toward terse output.
    """
    if prefix is None:
        prefix = "Question:"
    lines: list[str] = [f"{prefix} {question}", "", "Choices:"]
    for idx, choice in enumerate(choices):
        lines.append(f"{chr(ord('A') + idx)}. {choice}")
    lines.append("")
    lines.append(
        "Reply with ONLY the letter of the correct answer (e.g. 'A'). "
        "Do not include any other text."
    )
    return "\n".join(lines)


def grade_mc_case(case: dict, row: dict) -> dict:
    """Grade an MC row produced by ``lucebench.runner.run_case``.

    Shared between truthfulqa-mc1 and hellaswag. Both areas vendor
    ``case["choices"]`` (list[str]) and ``case["expected"]`` (single
    letter). Returns the standard grader shape.
    """
    expected = str(case.get("expected") or "").strip().upper()
    choices = case.get("choices") or []
    nchoices = len(choices)
    content = row.get("content") or ""
    reasoning = row.get("reasoning_content") or ""
    # Visible content takes precedence — models that emit "</think>X"
    # render X to content. Only fall back to reasoning when content is
    # empty (truncated mid-think).
    haystack = content if content.strip() else reasoning
    given = extract_mc_answer(haystack, nchoices)

    format_pass = given is not None
    strict_pass = format_pass and given == expected
    return {
        "pass": strict_pass,
        "given": given if given is not None else "?",
        "correct": expected,
        "status": "passed" if strict_pass else ("format_error" if not format_pass else "failed"),
        "format_pass": format_pass,
        "semantic_hint": strict_pass,
    }
