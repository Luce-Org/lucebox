"""Run one case against an OpenAI-shape /v1/chat/completions endpoint.

Deliberately stdlib-only — urllib.request, no httpx/requests. Keeps
the install lean and the wire path obvious for debugging.
"""

from __future__ import annotations

import json
import time
import urllib.request
from typing import Any

# Visible-output cap for non-thinking cases (smoke MC, short recall, etc.).
# Per-case overrides win via `case["max_tokens"]`.
DEFAULT_MAX_TOKENS = 512

DEFAULT_SYSTEM_PROMPT = (
    "You are solving a hard benchmark question. Reason carefully. "
    "The final answer must follow the requested format exactly."
)


def build_prompt(case: dict[str, Any]) -> str:
    """Render the user-message text for a case.

    Each area module is free to put its own preferred shape under
    ``case["question"]`` / ``case["prompt"]`` / ``case["user_message"]``
    and we pick the right one here.
    """
    if case.get("kind") == "code-completion":
        return (
            "Continue the following Python code. Output ONLY the function "
            "body — no markdown, no explanation, no extra prose:\n\n" + case["prompt"]
        )
    if case.get("kind") == "longctx-frontier":
        return case["prompt"]
    if case.get("kind") == "agent-prompt":
        return case["user_message"]
    if case.get("kind") == "smoke":
        # Smoke prompts ship as plain strings — no answer-format scaffold.
        # The whole point of the smoke area is "does the server reply?",
        # so anything that wraps the prompt would just dilute the signal.
        return case["prompt"]
    parts = [case["question"]]
    choices = case.get("choices") or []
    if choices:
        parts.append("\nChoices:")
        for idx, choice in enumerate(choices):
            parts.append(f"{chr(ord('A') + idx)}. {choice}")
        parts.append(
            "\nSolve the question. At the end, write exactly one final line in "
            "this format and do not write anything after it:\nAnswer: <letter>"
        )
    elif case.get("kind") in {"line", "compsec"}:
        parts.append(
            "\nAt the end, write exactly one final line in this format and do "
            "not write anything after it:\n"
            "Answer: <line number or comma-separated line numbers>"
        )
    else:
        parts.append(
            "\nSolve the problem. At the end, write exactly one final line in "
            "this format and do not write anything after it:\nAnswer: <integer>"
        )
    return "\n".join(parts)


def run_case(
    url: str,
    case: dict[str, Any],
    *,
    timeout_s: int = 300,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    think: bool = False,
    model: str = "default",
    auth_header: str = "",
    temperature: float | None = None,
    top_p: float | None = None,
    top_k: int | None = None,
    extra_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Send one case to the server, return a normalized row dict.

    Sampling fields (``temperature``, ``top_p``, ``top_k``) are sent
    only when explicitly set. Omitted fields let the server apply its
    own defaults — for luce-dflash this is the loaded model card's
    ``sampling`` section. Forcing values here would defeat that
    fallback and on Gemma 4 cause degenerate-decode collapse.

    ``extra_body`` is merged into the request body verbatim — use for
    server-specific knobs (e.g. ``chat_template_kwargs``,
    ``reasoning_effort``, provider routing hints).

    The returned row is the bench schema used by lucebench.cli's
    summary table:
      pass, wall_seconds, prompt_tokens, completion_tokens,
      content, reasoning_content, finish_reason, finish_details,
      timings, http_status, error.
    """
    prompt = build_prompt(case)
    request_max_tokens = int(case.get("max_tokens", max_tokens))
    body: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": case.get("system_prompt", DEFAULT_SYSTEM_PROMPT)},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": request_max_tokens,
        "stream": False,
        # Thinking control: both shapes shipped — chat_template_kwargs is
        # the vLLM/SGLang convention; reasoning_effort is the OpenAI/OR
        # convention. Servers ignore what they don't understand.
        "chat_template_kwargs": {"enable_thinking": think},
        "thinking": {"type": "enabled" if think else "disabled"},
        "reasoning_effort": "high" if think else "none",
    }
    if temperature is not None:
        body["temperature"] = float(temperature)
    if top_p is not None:
        body["top_p"] = float(top_p)
    if top_k is not None and top_k > 0:
        body["top_k"] = int(top_k)
    if extra_body:
        body.update(extra_body)

    headers = {"Content-Type": "application/json"}
    if auth_header:
        headers["Authorization"] = auth_header
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers=headers,
    )

    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            data = json.loads(resp.read())
            http_status = resp.status
    except Exception as e:
        return {
            "case_id": case.get("id"),
            "source": case.get("source"),
            "pass": False,
            "error": f"{type(e).__name__}: {e}",
            "wall_seconds": round(time.perf_counter() - t0, 3),
            "http_status": None,
        }
    wall = round(time.perf_counter() - t0, 3)

    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message", {}) if isinstance(choice, dict) else {}
    usage = data.get("usage", {}) or {}
    finish_details = choice.get("finish_details") or msg.get("finish_details") or {}

    # Reasoning-token count is delivered in several shapes depending on the
    # server. OpenAI / OpenRouter nest it under completion_tokens_details;
    # some older builds put it at the top level of usage as `reasoning_tokens`.
    # We surface both so format_row can render a `think=N` breakdown without
    # needing a tokenizer.
    ctd = usage.get("completion_tokens_details") or {}
    if not isinstance(ctd, dict):
        ctd = {}
    reasoning_tokens = ctd.get("reasoning_tokens")
    if reasoning_tokens is None:
        reasoning_tokens = usage.get("reasoning_tokens")

    return {
        "case_id": case.get("id"),
        "source": case.get("source"),
        "kind": case.get("kind"),
        "wall_seconds": wall,
        "http_status": http_status,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "reasoning_tokens": reasoning_tokens,
        "content": msg.get("content"),
        "reasoning_content": msg.get("reasoning_content") or msg.get("reasoning"),
        "finish_reason": choice.get("finish_reason"),
        "finish_details": finish_details,
        "timings": usage.get("timings"),
        # Caller grades; we just normalize the wire shape.
        "_response": data,
    }
