"""LLM-as-judge grader: a strict 0/1 rubric returned as JSON.

Defaults to the 27B (luce_server on :8216), thinking off, temperature 0. The judge
sees the user request, an optional rubric, an optional reference answer (for open
prompts the 27B thinking-on answer when one was collected) and the candidate answer.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .common import strip_think
from .graders import GradeResult

JUDGE_SYSTEM = """You are a strict grader for an AI assistant's answers.
Decide whether the CANDIDATE answer is acceptable to a demanding user who sent the REQUEST.

Score 1 only if ALL of these hold:
- It does what the request asks (task, format, language, length and other explicit constraints).
- It is factually and technically correct; code would work; no invented facts.
- It covers the essential points (use the RUBRIC and the REFERENCE, if given, as the bar;
  the candidate may be shorter or worded differently but must not miss or contradict key content).
- It is not truncated, empty, repetitive, or padded with irrelevant content.
Otherwise score 0. Do not reward length. When unsure, score 0.

Reply with ONLY a JSON object on one line: {"score": 0 or 1, "reason": "<one short sentence>"}"""


def build_judge_messages(
    request_text: str, candidate: str, reference: str | None = None, rubric: str | None = None
) -> list[dict[str, str]]:
    parts = [f"REQUEST:\n<<<\n{request_text}\n>>>"]
    if rubric:
        parts.append(f"RUBRIC:\n<<<\n{rubric}\n>>>")
    if reference:
        parts.append(f"REFERENCE (a known-good answer):\n<<<\n{reference}\n>>>")
    parts.append(f"CANDIDATE:\n<<<\n{candidate if candidate.strip() else '(empty)'}\n>>>")
    parts.append('Return the JSON object now.')
    return [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": "\n\n".join(parts)},
    ]


def parse_verdict(text: str) -> tuple[int, str]:
    text = strip_think(text)
    for m in reversed(list(re.finditer(r"\{[^{}]*\}", text, flags=re.DOTALL))):
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            continue
        if "score" in obj:
            score = 1 if str(obj["score"]).strip() in ("1", "1.0", "true", "True") else 0
            return score, str(obj.get("reason", ""))[:400]
    m = re.search(r'"?score"?\s*[:=]\s*([01])', text)
    if m:
        return int(m.group(1)), "unparsed_reason"
    raise ValueError(f"unparseable judge verdict: {text[:200]!r}")


@dataclass
class JudgeConfig:
    base_url: str = "http://127.0.0.1:8216/v1"
    model: str = "qwen38-27b"
    max_tokens: int = 256
    timeout_s: float = 300.0
    retries: int = 2


class Judge:
    def __init__(self, config: JudgeConfig, client: Any = None):
        import httpx

        self.config = config
        self._client = client or httpx.AsyncClient(timeout=config.timeout_s)

    async def aclose(self) -> None:
        await self._client.aclose()

    async def grade(
        self,
        request_text: str,
        raw_answer: str,
        reference: str | None = None,
        rubric: str | None = None,
    ) -> GradeResult:
        candidate = strip_think(raw_answer)
        body = {
            "model": self.config.model,
            "messages": build_judge_messages(request_text, candidate, reference, rubric),
            "temperature": 0,
            "max_tokens": self.config.max_tokens,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        last_err = ""
        for _ in range(self.config.retries + 1):
            try:
                r = await self._client.post(f"{self.config.base_url}/chat/completions", json=body)
                r.raise_for_status()
                content = r.json()["choices"][0]["message"].get("content") or ""
                score, reason = parse_verdict(content)
                return GradeResult(
                    bool(score), float(score), "judge",
                    {"reason": reason, "reference": bool(reference), "rubric": bool(rubric)},
                )
            except Exception as e:  # noqa: BLE001 - retried, then reported in detail
                last_err = f"{type(e).__name__}: {e}"[:300]
        return GradeResult(False, 0.0, "judge_error", {"error": last_err})
