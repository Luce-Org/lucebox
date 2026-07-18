"""Strict final-content scoring without benchmark-specific recovery."""

from __future__ import annotations

import dataclasses
import re
import unicodedata
from collections.abc import Iterable, Mapping
from typing import Any

from .records import sha256_json


@dataclasses.dataclass(frozen=True)
class ExactScore:
    passed: bool
    expected: str
    observed: str | None
    reason: str


def normalize_final_content(value: str) -> str:
    return unicodedata.normalize("NFC", value).strip()


def extract_final_content(response: str | Mapping[str, Any]) -> str | None:
    """Extract only a protocol-native final content field.

    Plain strings are treated as the complete final field. Structured inputs
    may use ``content`` directly or an OpenAI-compatible first choice. This
    intentionally does not search reasoning text, prompts, or arbitrary nested
    fields for the expected answer.
    """

    if isinstance(response, str):
        return response
    direct = response.get("content")
    if isinstance(direct, str):
        return direct
    choices = response.get("choices")
    if not isinstance(choices, list) or len(choices) != 1:
        return None
    choice = choices[0]
    if not isinstance(choice, Mapping):
        return None
    message = choice.get("message")
    if isinstance(message, Mapping) and isinstance(message.get("content"), str):
        return str(message["content"])
    if isinstance(choice.get("text"), str):
        return str(choice["text"])
    return None


def score_exact_final_content(expected: str, response: str | Mapping[str, Any]) -> ExactScore:
    wanted = normalize_final_content(expected)
    raw = extract_final_content(response)
    if raw is None:
        return ExactScore(False, wanted, None, "missing-or-ambiguous-final-content")
    observed = normalize_final_content(raw)
    if not observed:
        return ExactScore(False, wanted, observed, "empty-final-content")
    if observed == wanted:
        return ExactScore(True, wanted, observed, "exact-match")
    return ExactScore(False, wanted, observed, "not-exact")


def contains_expected_string(expected: str, response: str | Mapping[str, Any]) -> bool:
    """Diagnostic only; this must never be used as the publication scorer."""

    raw = extract_final_content(response)
    if raw is None:
        return False
    return normalize_final_content(expected) in normalize_final_content(raw)


def looks_like_prompt_echo(prompt: str, response: str | Mapping[str, Any]) -> bool:
    raw = extract_final_content(response)
    if raw is None:
        return False
    compact_prompt = re.sub(r"\s+", " ", normalize_final_content(prompt))
    compact_response = re.sub(r"\s+", " ", normalize_final_content(raw))
    return bool(compact_prompt and compact_prompt in compact_response)


def score_niah_records(
    cases: Iterable[Mapping[str, Any]], responses: Iterable[Mapping[str, Any]]
) -> dict[str, Any]:
    """Strictly join one hash-bound final response to every generated NIAH case."""

    case_by_id: dict[str, Mapping[str, Any]] = {}
    for case in cases:
        case_id = str(case.get("case_id") or "")
        if not case_id:
            raise ValueError("NIAH case lacks case_id")
        if case_id in case_by_id:
            raise ValueError(f"duplicate NIAH case: {case_id}")
        if not isinstance(case.get("answer"), str):
            raise ValueError(f"NIAH case lacks string answer: {case_id}")
        case_by_id[case_id] = case
    if not case_by_id:
        raise ValueError("NIAH case set is empty")

    response_by_id: dict[str, Mapping[str, Any]] = {}
    for response in responses:
        case_id = str(response.get("case_id") or response.get("prompt_id") or "")
        if not case_id:
            raise ValueError("NIAH response lacks case_id/prompt_id")
        if case_id not in case_by_id:
            raise ValueError(f"unplanned NIAH response: {case_id}")
        if case_id in response_by_id:
            raise ValueError(f"duplicate NIAH response: {case_id}")
        expected_hash = sha256_json(case_by_id[case_id])
        if response.get("prompt_hash") != expected_hash:
            raise ValueError(f"NIAH response prompt hash mismatch: {case_id}")
        if not isinstance(response.get("final_content"), str):
            raise ValueError(f"NIAH response lacks final_content: {case_id}")
        response_by_id[case_id] = response
    missing = sorted(set(case_by_id) - set(response_by_id))
    if missing:
        raise ValueError(f"NIAH responses are incomplete: {missing}")

    scores: list[dict[str, Any]] = []
    for case_id, case in case_by_id.items():
        exact = score_exact_final_content(str(case["answer"]), response_by_id[case_id]["final_content"])
        scores.append(
            {
                "case_id": case_id,
                "context_tokens": case.get("context_tokens"),
                "depth_fraction": case.get("depth_fraction"),
                "control": case.get("control"),
                "passed": exact.passed,
                "expected": exact.expected,
                "observed": exact.observed,
                "reason": exact.reason,
            }
        )
    primary = [score for score in scores if score["control"] is None]
    controls = [score for score in scores if score["control"] is not None]

    def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
        count = len(rows)
        passed_count = sum(bool(row["passed"]) for row in rows)
        return {
            "cases": count,
            "passed": passed_count,
            "failed": count - passed_count,
            "exact_match_rate": passed_count / count if count else None,
        }

    return {
        "schema": "lucebox.validation-v2.niah-scores/1",
        "summary": aggregate(scores),
        "primary": aggregate(primary),
        "controls": aggregate(controls),
        "scores": scores,
    }
