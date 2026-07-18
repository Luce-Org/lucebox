"""Strict final-content scoring without benchmark-specific recovery."""

from __future__ import annotations

import dataclasses
import re
import unicodedata
from collections.abc import Mapping
from typing import Any


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
