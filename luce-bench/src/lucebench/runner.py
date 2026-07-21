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

# Tier-2 client-thinking-budget defaults (see docs/client-thinking-budget.md).
# Reply reserve for the forced-`</think>` continuation when the card carries no
# `hard_limit_reply_budget`.
_CLIENT_THINKING_DEFAULT_REPLY_RESERVE = 4096
# Fallback terminator when the card has no `thinking_terminator_hint`.
_CLIENT_THINKING_DEFAULT_TERMINATOR = "</think>\n\n"


def _estimate_reasoning_tokens(reasoning_text: str) -> int:
    """Approximate reasoning-token count from accumulated text (char/4).

    No tokenizer dependency — this is a soft gate, not an exact cutoff. It
    miscounts math/code/CJK/whitespace-heavy text and, because it's only
    checked once per streamed chunk, OVERSHOOTS the budget by up to one
    chunk's worth of reasoning before the abort fires.
    """
    return len(reasoning_text) // 4


def _resolve_terminator(card: dict[str, Any] | None) -> str:
    """Pick the force-close phrase: card hint if present, else `</think>`."""
    if isinstance(card, dict):
        hint = card.get("thinking_terminator_hint")
        if isinstance(hint, str) and hint:
            return hint
    return _CLIENT_THINKING_DEFAULT_TERMINATOR


def _resolve_reply_reserve(card: dict[str, Any] | None) -> int:
    """Reply `max_tokens` for the continuation: card budget else 4096."""
    if isinstance(card, dict):
        reserve = card.get("hard_limit_reply_budget")
        if isinstance(reserve, int) and reserve > 0:
            return reserve
    return _CLIENT_THINKING_DEFAULT_REPLY_RESERVE

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
    min_p: float | None = None,
    presence_penalty: float | None = None,
    repetition_penalty: float | None = None,
    sampling_source: str | None = None,
    extra_body: dict[str, Any] | None = None,
    stream: bool = True,
    thinking_control_flag: str = "off",
    model_card: dict[str, Any] | None = None,
    server_honors_api_flags: bool = True,
    reasoning_effort: str = "high",
    thinking_budget_tokens: int | None = None,
    client_thinking_budget: int | None = None,
    card_source: str | None = None,
    card_stem: str | None = None,
) -> dict[str, Any]:
    """Send one case to the server, return a normalized row dict.

    Sampling fields (``temperature``, ``top_p``, ``top_k``, ``min_p``,
    ``presence_penalty``, ``repetition_penalty``) are sent only when the
    corresponding argument is not None; a None field is omitted so the
    server applies its own default. The CLI applies the resolved model
    card's ``sampling`` block by default (forcing the model's recommended
    decode params even against card-less servers like OpenRouter / MLX) and
    layers explicit ``--temperature`` / ``--top-p`` / ``--top-k`` over it
    per field; ``--no-card-sampling`` reverts to "CLI flags or omit". The
    runner just sends whatever effective values the caller threads in.
    ``sampling_source`` (``"card"`` | ``"cli"`` | ``"mixed"`` | ``"none"``)
    is provenance only — stamped on the row alongside the actual ``sampling``
    dict sent; it does not change the wire body.

    ``extra_body`` is merged into the request body verbatim — use for
    server-specific knobs (e.g. ``chat_template_kwargs``,
    ``reasoning_effort``, provider routing hints).

    ``reasoning_effort`` sets the OpenAI/OpenRouter ``reasoning_effort``
    request field in think mode (``"low"`` / ``"medium"`` / ``"high"``);
    nothink always sends ``"none"`` regardless. Default ``"high"`` keeps
    the wire body byte-identical to pre-feature behavior.

    ``thinking_budget_tokens`` (Tier-1 native budget hint): when set AND in
    think mode, adds ``thinking.budget_tokens=N`` to the Anthropic-shape
    ``thinking`` block (mirrors lucebench.probe's think-low / think-medium
    modes). Unset → the field is omitted and the body is unchanged.

    ``client_thinking_budget`` (Tier-2 client abort + re-prompt): when set
    AND in think mode AND streaming, the runner counts reasoning tokens as
    the stream arrives (char/4 estimate) and, once the estimate exceeds N,
    stops reading the stream and issues a SECOND, non-streamed request that
    re-conditions on the captured partial reasoning plus the card's
    ``thinking_terminator_hint`` (else ``</think>\\n\\n``) as an
    assistant-prefill turn, think disabled. That continuation's answer is
    graded. This is a FRESH conditioned sample, not a resumption of the
    original decode. Per-row metadata lands under ``row["client_thinking"]``.
    Unset → byte-identical to the pre-feature path (no second request, same
    streaming). Reasoning that is neither marked by ``reasoning_content``
    deltas nor ``<think>`` tags is ``marking="unmarked"``: no boundary
    exists, so the run finishes normally and is never aborted.

    ``card_source`` / ``card_stem`` are provenance only — stamped verbatim
    onto the returned row for the result schema (see
    :mod:`lucebench.model_cards`). They do not affect the wire body.

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
        "reasoning_effort": reasoning_effort if think else "none",
    }
    # Tier-1 native budget hint (Anthropic shape). Only meaningful in think
    # mode; nothink leaves the disabled block untouched. Mirrors the
    # budget_tokens field probe.py sends for its think-low / think-medium
    # modes — servers that don't understand it ignore it.
    if think and thinking_budget_tokens is not None:
        body["thinking"]["budget_tokens"] = int(thinking_budget_tokens)
    if stream:
        # Without include_usage the final SSE chunk omits the usage block
        # for OpenAI / OpenRouter (lucebox sends it either way). Without
        # this we'd lose prompt/completion token counts on streaming runs.
        body["stream_options"] = {"include_usage": True}
    # ── Effective sampling. Each field rides only when not None; a None
    # field is omitted so the server applies its own default. The dict is
    # echoed onto the row (with `sampling_source`) for provenance.
    sampling_sent: dict[str, Any] = {}
    if temperature is not None:
        sampling_sent["temperature"] = float(temperature)
    if top_p is not None:
        sampling_sent["top_p"] = float(top_p)
    if top_k is not None and top_k > 0:
        sampling_sent["top_k"] = int(top_k)
    if min_p is not None:
        sampling_sent["min_p"] = float(min_p)
    if presence_penalty is not None:
        sampling_sent["presence_penalty"] = float(presence_penalty)
    if repetition_penalty is not None:
        # Send both wire names: `repetition_penalty` (vLLM/HF/OpenRouter) and
        # `repeat_penalty` (llama.cpp / lucebox). Same value; servers ignore the
        # name they don't recognize. Without this, a card's repetition_penalty
        # silently fails to reach lucebox (which only reads repeat_penalty).
        sampling_sent["repetition_penalty"] = float(repetition_penalty)
        sampling_sent["repeat_penalty"] = float(repetition_penalty)
    body.update(sampling_sent)
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

    # Tier-2 client-thinking-budget is only meaningful with --think on a
    # streamed request. Off otherwise — the block below records mode="off"
    # and the streaming loop never aborts, so the wire path is byte-identical
    # to the pre-feature behavior.
    client_thinking_active = (
        client_thinking_budget is not None and client_thinking_budget > 0 and think and stream
    )
    # Canonical "feature off / not engaged" block, stamped on every non-aborted
    # return so the row schema is uniform whether or not the budget was set.
    off_block: dict[str, Any] = {
        "mode": "client_abort" if client_thinking_active else "off",
        "requested_budget": client_thinking_budget if client_thinking_active else None,
        "engaged": False,
        "marking": "unmarked",
        "reasoning_tokens_at_abort": 0,
        "continuation": "skipped",
        "answer_started_before_abort": False,
    }

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
                "card_source": card_source,
                "card_stem": card_stem,
                "sampling_source": sampling_source,
                "sampling": sampling_sent,
                "client_thinking": off_block,
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
            "card_source": card_source,
            "card_stem": card_stem,
            "sampling_source": sampling_source,
            "sampling": sampling_sent,
            "client_thinking": off_block,
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
    # Tier-2 abort bookkeeping. `marking` identifies the reasoning channel:
    # "reasoning_content" (best), "think_tags" (<think>…</think> in content),
    # or "unmarked" (no boundary — never abort, finish normally). `aborted`
    # flips once the char/4 estimate trips the budget and we close the stream.
    marking = "unmarked"
    aborted = False
    answer_started_before_abort = False
    reasoning_tokens_at_abort = 0
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

                # ── Tier-2 client-thinking-budget gate. Identify the
                # reasoning channel and estimate reasoning tokens (char/4)
                # as deltas arrive; abort once over budget so a backend
                # that ignores native hints can't reason to the ceiling.
                if not client_thinking_active:
                    continue
                if piece_reason:
                    # reasoning_content / reasoning deltas → authoritative.
                    marking = "reasoning_content"
                if marking == "reasoning_content":
                    reasoning_text = "".join(reasoning_parts)
                    # Any visible content delta alongside reasoning means the
                    # answer has begun — a re-prompt could duplicate it.
                    if piece_content:
                        answer_started_before_abort = True
                elif "<think>" in "".join(content_parts):
                    # Inline <think>…</think> in the content stream. Count
                    # chars from after the open tag to either the close tag
                    # or, if still open, the current end of content. Anything
                    # after </think> is the visible answer.
                    marking = "think_tags"
                    joined = "".join(content_parts)
                    start = joined.find("<think>") + len("<think>")
                    end = joined.find("</think>", start)
                    if end == -1:
                        reasoning_text = joined[start:]
                    else:
                        reasoning_text = joined[start:end]
                        after = joined[end + len("</think>") :]
                        if after.strip():
                            answer_started_before_abort = True
                else:
                    # No reasoning_content, no <think> tag yet → unmarked.
                    # Cannot identify thinking; do NOT abort.
                    continue

                est = _estimate_reasoning_tokens(reasoning_text)
                # client_thinking_active guarantees the budget is a positive int.
                if est > (client_thinking_budget or 0):
                    # Over budget. Stop reading and close the stream (the
                    # `with` block exits on break). char/4 overshoots by up
                    # to one chunk, so `est` is the partial-at-abort estimate.
                    reasoning_tokens_at_abort = est
                    aborted = True
                    break
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
            "card_source": card_source,
            "card_stem": card_stem,
            "sampling_source": sampling_source,
            "sampling": sampling_sent,
            "client_thinking": off_block,
            "_thinking_injection": injection_info,
        }

    # ── Tier-2 forced-`</think>` continuation. When the budget tripped, the
    # captured partial reasoning is re-conditioned in a SECOND, independent
    # request: original messages + an assistant-prefill turn carrying the
    # partial reasoning plus the card's terminator, think disabled. This is a
    # FRESH conditioned sample, not a resumption of the aborted decode — the
    # original stream's server-side generation may keep running; we only
    # stopped reading. The continuation's answer is what the caller grades.
    if aborted:
        partial_reasoning = (
            "".join(reasoning_parts) if marking == "reasoning_content" else None
        )
        if partial_reasoning is None:
            # think_tags: reconstruct the reasoning slice from content.
            joined = "".join(content_parts)
            start = joined.find("<think>") + len("<think>")
            end = joined.find("</think>", start)
            partial_reasoning = joined[start:] if end == -1 else joined[start:end]
        cont_content, cont_ok = _client_thinking_continuation(
            url=url,
            base_messages=new_messages,
            partial_reasoning=partial_reasoning,
            terminator=_resolve_terminator(model_card),
            reply_reserve=_resolve_reply_reserve(model_card),
            model=model,
            auth_header=auth_header,
            timeout_s=timeout_s,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            min_p=min_p,
            presence_penalty=presence_penalty,
            repetition_penalty=repetition_penalty,
            extra_body=extra_body,
        )
        wall = round(time.perf_counter() - t0, 3)
        client_thinking_block: dict[str, Any] = {
            "mode": "client_abort",
            "requested_budget": client_thinking_budget,
            "engaged": True,
            "marking": marking,
            "reasoning_tokens_at_abort": reasoning_tokens_at_abort,
            "continuation": "ok" if cont_ok else "unsupported",
            "answer_started_before_abort": answer_started_before_abort,
        }
        return {
            "case_id": case.get("id"),
            "source": case.get("source"),
            "kind": case.get("kind"),
            "wall_seconds": wall,
            "ttft_seconds": round(ttft, 4) if ttft is not None else None,
            "streaming": True,
            "http_status": http_status,
            "prompt_tokens": None,
            "completion_tokens": None,
            # Estimated reasoning-at-abort; the aborted stream lacks final usage.
            "reasoning_tokens": reasoning_tokens_at_abort,
            # The continuation answer is graded; falls back to the (truncated)
            # original content when the continuation is unsupported.
            "content": cont_content if cont_ok else ("".join(content_parts) or None),
            "reasoning_content": partial_reasoning or None,
            "finish_reason": finish_reason,
            "finish_details": finish_details,
            "timings": None,
            "card_source": card_source,
            "card_stem": card_stem,
            "sampling_source": sampling_source,
            "sampling": sampling_sent,
            "client_thinking": client_thinking_block,
            "_response": last_chunk,
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

    # Budget set but never tripped (thinking finished under N, or output was
    # unmarked so no boundary existed): no second request, engaged=False. The
    # `marking` reflects what we actually saw so an unmarked backend surfaces.
    if client_thinking_active:
        client_thinking_final = dict(off_block)
        client_thinking_final["marking"] = marking
    else:
        client_thinking_final = off_block

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
        "card_source": card_source,
        "card_stem": card_stem,
        "sampling_source": sampling_source,
        "sampling": sampling_sent,
        "client_thinking": client_thinking_final,
        # Caller grades; we just normalize the wire shape.
        "_response": last_chunk,
        "_thinking_injection": injection_info,
    }


def _client_thinking_continuation(
    *,
    url: str,
    base_messages: list[dict[str, Any]],
    partial_reasoning: str,
    terminator: str,
    reply_reserve: int,
    model: str,
    auth_header: str,
    timeout_s: int,
    temperature: float | None,
    top_p: float | None,
    top_k: int | None,
    min_p: float | None = None,
    presence_penalty: float | None = None,
    repetition_penalty: float | None = None,
    extra_body: dict[str, Any] | None,
) -> tuple[str | None, bool]:
    """Tier-2 forced-`</think>` re-prompt — a fresh, non-streamed request.

    Builds ``base_messages + [{role: assistant, content: <partial reasoning>
    + terminator}]`` and POSTs it with thinking DISABLED (``enable_thinking``
    false, ``reasoning_effort`` none, no Anthropic ``thinking`` budget) and its
    own ``max_tokens`` reply reserve. The model is conditioned on the captured
    partial reasoning followed by the terminator, so it should emit the visible
    answer directly rather than keep reasoning.

    This is a **fresh conditioned sample, not a decoder continuation** of the
    aborted stream — the server has no memory of the original decode; we simply
    feed the partial reasoning back as assistant context.

    Returns ``(content, ok)``. ``ok`` is False — and the row is flagged
    ``continuation="unsupported"`` by the caller — when the request errors, the
    provider rejects the assistant-prefill, or the answer comes back empty.
    NEVER raises: a provider-capability failure must not abort the benchmark.
    """
    messages = list(base_messages) + [
        {"role": "assistant", "content": partial_reasoning + terminator}
    ]
    body: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": int(reply_reserve),
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": False},
        "thinking": {"type": "disabled"},
        "reasoning_effort": "none",
    }
    if temperature is not None:
        body["temperature"] = float(temperature)
    if top_p is not None:
        body["top_p"] = float(top_p)
    if top_k is not None and top_k > 0:
        body["top_k"] = int(top_k)
    if min_p is not None:
        body["min_p"] = float(min_p)
    if presence_penalty is not None:
        body["presence_penalty"] = float(presence_penalty)
    if repetition_penalty is not None:
        body["repetition_penalty"] = float(repetition_penalty)
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
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            data = json.loads(resp.read())
    except Exception:
        return None, False
    if not isinstance(data, dict):
        return None, False
    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message", {}) if isinstance(choice, dict) else {}
    content = msg.get("content") if isinstance(msg, dict) else None
    if not isinstance(content, str) or not content.strip():
        # Empty answer or provider rejected the prefill → unsupported.
        return None, False
    return content, True
