"""HellaSwag commonsense ending-selection — `--areas hellaswag`.

100-case sample of the canonical ``Rowan/hellaswag`` validation split
(MIT), sampled with ``random.Random(42)`` and vendored as JSONL. Every
upstream row has exactly 4 candidate endings and a 0..3 label pointing
at the correct one; the loader resolves that into a single ``expected``
letter (A/B/C/D) so the grader is a trivial letter-compare.

Prompt: ``ctx`` followed by A. / B. / C. / D. labelled endings, asking
for the answer letter only. We re-use the multiple-choice scaffold +
extractor from :mod:`lucebench.areas._mc` — shared with TruthfulQA so
both areas grade identically.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ._mc import GRADER_VERSION as _MC_GRADER_VERSION
from ._mc import build_mc_prompt, grade_mc_case

# Re-exported from the shared MC grader so the regrade CLI can read a
# single ``GRADER_VERSION`` attribute off any area module.
GRADER_VERSION = _MC_GRADER_VERSION

FIXTURE_PATH = (
    Path(__file__).resolve().parent.parent / "fixtures" / "hellaswag" / "cases.jsonl"
)

# Tight cap: each case only needs the answer letter. HellaSwag answers
# rarely benefit from thinking — the task is "pick the plausible
# continuation", not a reasoning chain — so the budget is intentionally
# small to keep the sweep fast.
HELLASWAG_MAX_TOKENS = 128


def load_hellaswag_cases(path: Path = FIXTURE_PATH) -> list[dict[str, Any]]:
    """Load the vendored HellaSwag case set (JSONL).

    The ``prompt`` is pre-rendered with the shared MC scaffold; the
    ``ctx`` field is the prefix sentence (we render it under
    "Context:" rather than "Question:" because HellaSwag rows are
    sentence completions, not questions).
    """
    out: list[dict[str, Any]] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            endings = list(raw["endings"])
            expected_idx = int(raw["expected_index"])
            expected_letter = chr(ord("A") + expected_idx)
            prompt = build_mc_prompt(raw["ctx"], endings, prefix="Context:")
            out.append(
                {
                    "area": "hellaswag",
                    "source": raw.get("source", "hellaswag"),
                    "id": raw["id"],
                    "kind": "multiple-choice",
                    "prompt": prompt,
                    "ctx": raw["ctx"],
                    "activity_label": raw.get("activity_label"),
                    "choices": endings,
                    "answer": expected_letter,
                    "expected": expected_letter,
                    "expected_index": expected_idx,
                    "domain": "commonsense",
                    "title": raw["id"],
                }
            )
    return out


def grade_hellaswag_case(case: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    """Thin wrapper around the shared MC grader. See
    :func:`lucebench.areas._mc.grade_mc_case` for the contract.
    """
    return grade_mc_case(case, row)
