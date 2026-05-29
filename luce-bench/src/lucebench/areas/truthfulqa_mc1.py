"""TruthfulQA MC1 — `--areas truthfulqa-mc1`.

100-case sample of the canonical ``truthful_qa`` validation split
(Apache-2.0), ``multiple_choice`` config, sampled with
``random.Random(42)`` and vendored as JSONL. Each upstream row has
between 2 and 13 candidate answers in ``mc1_targets.choices`` with
exactly one labelled 1 (the truthful answer); the loader resolves that
into a single ``expected`` letter so the grader is a trivial
letter-compare.

Prompting: question + numbered choices ("A. …\\nB. …"), asking for the
answer letter only. The letter range is dynamic — cases with only 2
choices show "A" and "B"; cases with 13 show "A".."M".

Grader: ``lucebench.areas._mc.extract_mc_answer`` — looks for
``answer is X`` / ``final answer: X`` first, falls back to the last
standalone in-range letter. Shared with the HellaSwag area.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ._mc import GRADER_VERSION as _MC_GRADER_VERSION
from ._mc import build_mc_prompt, grade_mc_case

# Re-exported from the shared MC grader; see lucebench.areas._mc.
GRADER_VERSION = _MC_GRADER_VERSION

FIXTURE_PATH = (
    Path(__file__).resolve().parent.parent / "fixtures" / "truthfulqa_mc1" / "cases.jsonl"
)

# MC questions don't need much room for a real answer — the model just
# has to emit a letter. We allow some slack so reasoning-mode models
# can think briefly before answering without tripping the budget cap.
TRUTHFULQA_MC1_MAX_TOKENS = 256


def load_truthfulqa_mc1_cases(path: Path = FIXTURE_PATH) -> list[dict[str, Any]]:
    """Load the vendored TruthfulQA MC1 case set (JSONL).

    Each case carries the canonical fields the runner / grader rely on
    plus area-specific ``choices`` (list[str]) and ``expected`` (single
    uppercase letter). The ``prompt`` field is pre-rendered with the
    MC scaffold so the runner can pass it through verbatim
    (kind=multiple-choice).
    """
    out: list[dict[str, Any]] = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            choices = list(raw["choices"])
            expected_idx = int(raw["expected_index"])
            expected_letter = chr(ord("A") + expected_idx)
            prompt = build_mc_prompt(raw["question"], choices)
            out.append(
                {
                    "area": "truthfulqa-mc1",
                    "source": raw.get("source", "truthfulqa-mc1"),
                    "id": raw["id"],
                    "kind": "multiple-choice",
                    "prompt": prompt,
                    "question": raw["question"],
                    "choices": choices,
                    "answer": expected_letter,
                    "expected": expected_letter,
                    "expected_index": expected_idx,
                    "domain": "truthfulness",
                    "title": raw["id"],
                }
            )
    return out


def grade_truthfulqa_mc1_case(case: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    """Delegate to the shared MC grader. Exists as a thin wrapper so
    ``AREAS["truthfulqa-mc1"]["grade"]`` carries a stable reference even
    if the shared helper changes signature.
    """
    return grade_mc_case(case, row)
