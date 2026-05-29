"""Run one case against an OpenAI-shape /v1/chat/completions endpoint.

Deliberately stdlib-only — urllib.request, no httpx/requests. Keeps
the install lean and the wire path obvious for debugging.
"""

from __future__ import annotations

import json
import time
import urllib.request
from typing import Any

from lucebench._thinking import maybe_inject_thinking_token

# SSE framing: OpenAI-shape streaming responses send "data: {…}\n\n" frames
# terminated by "data: [DONE]\n\n". The parser below reads bytes off the
# socket as they arrive (no .read() waterfall) and yields per-frame JSON.
_SSE_DONE = "[DONE]"

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
    if case.get("kind") in {"math-word-problem", "multiple-choice"}:
        # Areas (gsm8k, truthfulqa, hellaswag) that vendor a prompt with
        # their own answer-format scaffold baked in by the loader. The
        # runner just passes them through verbatim — adding the generic
        # "Answer: <integer>" scaffold below would dilute the prompt the
        # area carefully constructed (e.g. asking for a letter when the
        # scaffold demanded an integer).
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


def _iter_sse_frames(resp: Any):
    """Yield decoded JSON payloads from a streamed SSE response.

    Reads the response body line-by-line so each ``data: …\\n\\n`` frame
    can be surfaced as it arrives — that's what lets the caller stamp
    TTFT on the first delta chunk carrying real content. Frames that
    fail to JSON-parse are skipped silently (some servers send keep-alive
    comments or non-JSON probe lines).
    """
    buf: list[str] = []
    for raw_line in resp:
        # urllib's response iterator hands back bytes per line, including
        # the trailing newline. SSE frame boundary = a blank line.
        line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
        if line == "":
            if not buf:
                continue
            frame = "\n".join(buf)
            buf = []
            # OpenAI prefixes each data line with "data: ". Strip and
            # bail on the [DONE] sentinel.
            if frame.startswith("data:"):
                payload = frame[5:].lstrip()
            else:
                payload = frame
            if payload == _SSE_DONE:
                return
            try:
                yield json.loads(payload)
            except ValueError:
                continue
        else:
            buf.append(line)
    # Final flush if the server closes without a trailing blank line.
    if buf:
        frame = "\n".join(buf)
        if frame.startswith("data:"):
            payload = frame[5:].lstrip()
            if payload != _SSE_DONE:
                try:
                    yield json.loads(payload)
                except ValueError:
                    pass


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
    stream: bool = True,
    thinking_control_flag: str = "off",
    model_card: dict[str, Any] | None = None,
    server_honors_api_flags: bool = True,
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

    ``thinking_control_flag`` toggles the client-side prompt-level
    injection fallback (see :mod:`lucebench._thinking`). The default
    ``"off"`` keeps the wire body unchanged for back-compat. ``"on"``
    appends the family/card thinking token to the last user turn
    regardless of server; ``"auto"`` skips injection only when
    ``server_honors_api_flags`` is True (the preflight confirmed a
    lucebox stack via /props). Per-row injection metadata is returned
    under ``row["_thinking_injection"]`` so the CLI can stamp the
    top-level ``thinking_control_injection`` block on result.json.

    ``stream`` defaults to True so every row carries a ``ttft_seconds``
    measurement (time from request POST to the first delta chunk carrying
    real content or reasoning_content). Pass ``stream=False`` to fall back
    to the legacy single-shot POST — same wire body, no SSE parsing, no
    ttft. The streaming path also sets ``stream_options.include_usage`` so
    the final chunk still carries the usage block (prompt/completion
    tokens, server-side timings).

    The returned row is the bench schema used by lucebench.cli's
    summary table:
      pass, wall_seconds, ttft_seconds, streaming, prompt_tokens,
      completion_tokens, content, reasoning_content, finish_reason,
      finish_details, timings, http_status, error.
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
        "stream": bool(stream),
        # Thinking control: both shapes shipped — chat_template_kwargs is
        # the vLLM/SGLang convention; reasoning_effort is the OpenAI/OR
        # convention. Servers ignore what they don't understand.
        "chat_template_kwargs": {"enable_thinking": think},
        "thinking": {"type": "enabled" if think else "disabled"},
        "reasoning_effort": "high" if think else "none",
    }
    if stream:
        # Without include_usage the final SSE chunk omits the usage block
        # for OpenAI / OpenRouter (lucebox sends it either way). Without
        # this we'd lose prompt/completion token counts on streaming runs.
        body["stream_options"] = {"include_usage": True}
    if temperature is not None:
        body["temperature"] = float(temperature)
    if top_p is not None:
        body["top_p"] = float(top_p)
    if top_k is not None and top_k > 0:
        body["top_k"] = int(top_k)
    if extra_body:
        body.update(extra_body)

    # ── Client-side thinking-control fallback. API flags above are sent
    # on every request; the prompt-level injection adds the family's
    # in-band token (``/think`` / ``/no_think`` on Qwen3.x) to the last
    # user turn when the operator opts in. Default ``"off"`` leaves the
    # body unchanged so callers that don't pass the flag get the same
    # wire shape as before this feature shipped. See _thinking.py for
    # the resolution order.
    mode = "think" if think else "nothink"
    new_messages, injection_info = maybe_inject_thinking_token(
        body["messages"],
        mode=mode,
        model_id=model,
        card=model_card,
        control_flag=thinking_control_flag,
        server_honors_api_flags=server_honors_api_flags,
    )
    body["messages"] = new_messages

    headers = {"Content-Type": "application/json"}
    if stream:
        headers["Accept"] = "text/event-stream"
    if auth_header:
        headers["Authorization"] = auth_header
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers=headers,
    )

    t0 = time.perf_counter()
    if not stream:
        # Legacy single-shot POST. Kept for callers that explicitly opt
        # out — tests with mocked urlopen, and any future profile run
        # where comparing streaming vs non-streaming wire cost matters.
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
                "streaming": False,
                "_thinking_injection": injection_info,
            }
        wall = round(time.perf_counter() - t0, 3)

        choice = (data.get("choices") or [{}])[0]
        msg = choice.get("message", {}) if isinstance(choice, dict) else {}
        usage = data.get("usage", {}) or {}
        finish_details = choice.get("finish_details") or msg.get("finish_details") or {}

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
            "ttft_seconds": None,
            "streaming": False,
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
            "_thinking_injection": injection_info,
        }

    # ── Streaming path ──────────────────────────────────────────────
    # Accumulate content/reasoning_content across delta chunks; stamp
    # ttft on the first delta carrying non-empty text. Usage + timings
    # land on the final chunk (with stream_options.include_usage set).
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    ttft: float | None = None
    finish_reason: str | None = None
    finish_details: dict[str, Any] = {}
    usage: dict[str, Any] = {}
    model_id: str | None = None
    last_chunk: dict[str, Any] = {}
    http_status: int | None = None
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            http_status = resp.status
            for chunk in _iter_sse_frames(resp):
                last_chunk = chunk
                if not isinstance(chunk, dict):
                    continue
                if model_id is None:
                    model_id = chunk.get("model")
                # Usage block can ride on the final chunk; some servers
                # also send a separate usage-only frame with choices=[].
                if isinstance(chunk.get("usage"), dict):
                    usage = chunk["usage"]
                choices = chunk.get("choices") or []
                if not choices:
                    continue
                choice = choices[0] if isinstance(choices[0], dict) else {}
                delta = choice.get("delta") or {}
                if not isinstance(delta, dict):
                    delta = {}
                piece_content = delta.get("content") or ""
                piece_reason = (
                    delta.get("reasoning_content") or delta.get("reasoning") or ""
                )
                if (piece_content or piece_reason) and ttft is None:
                    ttft = time.perf_counter() - t0
                if piece_content:
                    content_parts.append(piece_content)
                if piece_reason:
                    reasoning_parts.append(piece_reason)
                fr = choice.get("finish_reason")
                if fr:
                    finish_reason = fr
                fd = choice.get("finish_details") or {}
                if fd:
                    finish_details = fd
    except Exception as e:
        return {
            "case_id": case.get("id"),
            "source": case.get("source"),
            "pass": False,
            "error": f"{type(e).__name__}: {e}",
            "wall_seconds": round(time.perf_counter() - t0, 3),
            "ttft_seconds": round(ttft, 4) if ttft is not None else None,
            "streaming": True,
            "http_status": http_status,
            "_thinking_injection": injection_info,
        }
    wall = round(time.perf_counter() - t0, 3)

    ctd = usage.get("completion_tokens_details") or {}
    if not isinstance(ctd, dict):
        ctd = {}
    reasoning_tokens = ctd.get("reasoning_tokens")
    if reasoning_tokens is None:
        reasoning_tokens = usage.get("reasoning_tokens")

    content = "".join(content_parts) if content_parts else None
    reasoning = "".join(reasoning_parts) if reasoning_parts else None

    return {
        "case_id": case.get("id"),
        "source": case.get("source"),
        "kind": case.get("kind"),
        "wall_seconds": wall,
        "ttft_seconds": round(ttft, 4) if ttft is not None else None,
        "streaming": True,
        "http_status": http_status,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "reasoning_tokens": reasoning_tokens,
        "content": content,
        "reasoning_content": reasoning,
        "finish_reason": finish_reason,
        "finish_details": finish_details,
        "timings": usage.get("timings"),
        # Caller grades; we just normalize the wire shape.
        "_response": last_chunk,
        "_thinking_injection": injection_info,
    }
