"""Small graded HTTP capability benchmark for lucebox.

The default area is `smoke`: short deterministic checks that should pass 100%
and are suitable for optimizer gating. `long` measures long-context
reliability. The `ds4-eval` area dispatches into the ported antirez/ds4
corpus living in `bench_ds4_eval.py`; ds4-specific data, graders, and
budget defaults stay colocated there so the upstream diff stays narrow.

The `forge` area runs antoinezambelli/forge's tool-calling eval scenarios
against our Anthropic-Messages-compatible ``/v1/messages`` endpoint. Both
the runtime (``AnthropicClient`` + ``WorkflowRunner``) and the eval
driver are vendored under ``scripts/fixtures/forge_eval/`` — the runtime
in ``_forge/`` and the scenarios + ``run_eval`` driver alongside it. The
``forge-guardrails`` PyPI dep is no longer required; only the
``anthropic`` SDK is installed via the ``eval`` extra.

Standalone `ds4-eval`, `long`, `forge`, and `all` runs are score-only
unless `--min-pass-rate` is set explicitly.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
# ds4-specific data, graders, and budget knobs live in bench_ds4_eval.py
# so the colocation with antirez/ds4 stays obvious. Re-export the names
# this module's callers (lucebox_bench, profile, tests) already import.
from bench_ds4_eval import (  # noqa: E402
    DS4_EVAL_CASES,
    DS4_EVAL_CASES_PATH,
    DS4_EVAL_MAX_TOKENS,
    DS4_SOURCES,
    compsec_answer_matches,
    find_compsec_answer,
    find_ds4_choice_answer,
    find_ds4_integer_answer,
    is_ds4_eval_case,
    load_ds4_eval_cases,
    parse_line_spec,
    visible_text,
)

__all__ = [
    "DS4_EVAL_CASES",
    "DS4_EVAL_CASES_PATH",
    "DS4_EVAL_MAX_TOKENS",
    "DS4_SOURCES",
    "compsec_answer_matches",
    "find_compsec_answer",
    "find_ds4_choice_answer",
    "find_ds4_integer_answer",
    "is_ds4_eval_case",
    "load_ds4_eval_cases",
    "parse_line_spec",
    "visible_text",
]

SYSTEM_PROMPT = (
    "You are solving a hard benchmark question. Reason carefully. "
    "The final answer must follow the requested format exactly."
)
SMOKE_SYSTEM_PROMPT = (
    "Return only the requested final answer line. Do not show work."
)

LOCAL_CASES = [
    {
        "area": "smoke",
        "source": "smoke-mc",
        "id": "arithmetic-choice",
        "kind": "choice",
        "question": "What is 19 + 23?",
        "choices": ["41", "42", "43", "44"],
        "answer": "B",
        "system_prompt": SMOKE_SYSTEM_PROMPT,
        "max_tokens": 64,
    },
    {
        "area": "smoke",
        "source": "smoke-recall",
        "id": "short-recall",
        "kind": "integer",
        "question": (
            "Remember this value for the final answer: project_code = 7319.\n"
            "What is project_code?"
        ),
        "answer": "7319",
        "system_prompt": SMOKE_SYSTEM_PROMPT,
        "max_tokens": 64,
    },
    {
        "area": "long",
        "source": "smoke-context",
        "id": "needle-recall",
        "kind": "integer",
        "question": (
            "Remember this value for the final answer: project_code = 7319.\n"
            "Background: Lucebox benchmark traces should preserve prompt text, "
            "server usage, model output, and grading decisions so regressions "
            "can be inspected after optimizer sweeps. " * 30
            + "\nWhat is project_code?"
        ),
        "answer": "7319",
        "system_prompt": SMOKE_SYSTEM_PROMPT,
        "max_tokens": 64,
    },
]

# Visible-output cap for non-thinking cases (smoke MC, short recall, etc.).
# The smoke cases override per-case to 64.
DEFAULT_MAX_TOKENS = 512


def build_prompt(case: dict[str, Any]) -> str:
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
    elif case["kind"] in {"line", "compsec"}:
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


def find_choice_answer(generated: str, nchoices: int) -> str:
    text = visible_text(generated)
    max_answer = chr(ord("A") + nchoices - 1)
    answer = re.search(r"answer\s*[:\-]?\s*([A-Z])\b", text, flags=re.IGNORECASE)
    if answer:
        got = answer.group(1).upper()
        return got if "A" <= got <= max_answer else "?"
    return "?"


def find_integer_answer(generated: str) -> str:
    """Strict-format integer extractor for smoke / non-ds4 cases.

    Smoke prompts instruct the model to write "Answer: N" verbatim;
    accepting it loosely (the way ds4 cases do) would let an unrelated
    digit run trigger a false pass. ds4-eval cases route through
    `bench_ds4_eval.find_ds4_integer_answer` instead.
    """
    text = visible_text(generated)
    answer = re.search(r"answer\s*[:\-]?\s*(-?\d+)\b", text, flags=re.IGNORECASE)
    if answer:
        return str(int(answer.group(1)))
    return "?"


def find_answer(case: dict[str, Any], generated: str) -> str:
    if is_ds4_eval_case(case):
        if case["kind"] == "choice":
            return find_ds4_choice_answer(generated, len(case.get("choices") or []))
        if case["kind"] == "compsec":
            return find_compsec_answer(generated)
        return find_ds4_integer_answer(generated)
    if case["kind"] == "choice":
        return find_choice_answer(generated, len(case.get("choices") or []))
    return find_integer_answer(generated)


def find_answer_with_fallback(
    case: dict[str, Any], content: str, reasoning_content: str | None,
) -> str:
    """Look in ``content`` first, then ``reasoning_content`` as a fallback.

    Qwen3.6 routinely keeps thinking past ``</think>`` on math/code prompts
    and exhausts the completion budget before writing the strict-format
    ``Answer: N`` line. The full answer is usually present in
    ``reasoning_content`` though — checking it lets us grade on "did the
    model arrive at the answer" rather than just "did the model also nail
    the output format." Choice questions ("Answer: B") stay content-only
    because letter-grade extraction in mid-stream reasoning is noisy.
    """
    got = find_answer(case, content)
    if is_ds4_eval_case(case):
        return got
    if got != "?" or not reasoning_content or case["kind"] == "choice":
        return got
    return find_answer(case, reasoning_content)


def semantic_hint_present(
    case: dict[str, Any], content: str, reasoning_content: str | None,
) -> bool:
    """Best-effort diagnostic: did the model mention an expected answer?

    The hint is reported separately. It is not used for ds4-eval pass/fail,
    because comparability requires the same final-answer grading semantics.
    """
    if case["kind"] == "choice":
        return (
            find_answer(case, content)
            in expected_answers(case)
        )

    text = visible_text(content)
    if reasoning_content:
        text += "\n" + visible_text(reasoning_content)
    if case["kind"] == "compsec":
        expected_lines = parse_line_spec(",".join(expected_answers(case)))
        found_lines = parse_line_spec(text)
        return bool(expected_lines & found_lines)
    expected = {str(int(answer)) for answer in expected_answers(case)}
    # Cap digit length: a degenerate model output can emit a 16000-digit run
    # which trips Python 3.11+'s 4300-digit `int()` limit. Real answers fit
    # in 20 digits; longer matches can never equal an expected answer.
    found = {
        str(int(match.group(0)))
        for match in re.finditer(r"-?\d+", text)
        if len(match.group(0).lstrip("-")) <= 20
    }
    return bool(expected & found)


def grade_case(
    case: dict[str, Any], got: str, content: str, reasoning_content: str | None,
) -> dict[str, Any]:
    expected = expected_answers(case)
    format_pass = got != "?"
    if case["kind"] == "compsec":
        strict_pass = any(compsec_answer_matches(answer, got) for answer in expected)
    else:
        strict_pass = got in expected
    hint = semantic_hint_present(case, content, reasoning_content)
    semantic_pass = bool(case.get("score_mode") == "semantic" and hint)
    graded_pass = strict_pass or semantic_pass

    if strict_pass:
        status = "passed"
    elif semantic_pass:
        status = "semantic_pass"
    elif format_pass:
        status = "failed"
    else:
        status = "format_error"

    return {
        "status": status,
        "ok": graded_pass,
        "graded_pass": graded_pass,
        "strict_pass": strict_pass,
        "format_pass": format_pass,
        "semantic_hint": hint,
        "semantic_pass": semantic_pass,
    }


def expected_answers(case: dict[str, Any]) -> list[str]:
    raw = case["answer"]
    if isinstance(raw, list):
        return [str(item) for item in raw]
    return [str(raw)]


def select_cases(
    area: str,
    questions: int | None = None,
    case_index: int | None = None,
    case_id: str | None = None,
) -> list[dict[str, Any]]:
    if area == "ds4-eval":
        selected = list(DS4_EVAL_CASES)
    elif area == "all":
        selected = list(LOCAL_CASES) + list(DS4_EVAL_CASES)
    else:
        selected = [case for case in LOCAL_CASES if case["area"] == area]
    if case_index is not None:
        selected = [case for case in selected if case.get("ds4_index") == case_index]
    if case_id:
        selected = [case for case in selected if case.get("id") == case_id]
    if questions is not None:
        selected = selected[:max(0, min(questions, len(selected)))]
    return selected


def default_min_pass_rate(area: str) -> float:
    return 1.0 if area == "smoke" else 0.0


def default_max_tokens(area: str) -> int:
    # Combined cap (reasoning + reply). Thinking areas get the ds4_eval.c
    # default so the model has headroom; non-thinking areas (smoke, recall)
    # keep the short cap since a one-word answer doesn't need a buffer.
    # ``forge`` falls back to the long cap too — its scenarios are
    # multi-step tool-calling chains and benefit from headroom.
    return (
        DS4_EVAL_MAX_TOKENS
        if area in {"ds4-eval", "forge", "all"}
        else DEFAULT_MAX_TOKENS
    )


def default_thinking_enabled(area: str) -> bool:
    return area in {"ds4-eval", "all"}


def format_timings_suffix(timings: dict[str, Any] | None) -> str:
    """Render the ``prefill=… decode=… <tps>tok/s`` suffix for a bench row.

    Returns an empty string when ``timings`` is missing/malformed so the
    bench printout degrades gracefully against backends that don't
    surface ``usage.timings`` (OpenRouter, older sindri binaries, the
    forge area whose client doesn't propagate per-call timings).
    """
    if not isinstance(timings, dict):
        return ""
    if not any(k in timings for k in
               ("prefill_ms", "decode_ms", "decode_tokens_per_sec",
                "prefill_tokens_per_sec")):
        return ""
    try:
        prefill_ms = float(timings.get("prefill_ms", 0.0))
        decode_ms = float(timings.get("decode_ms", 0.0))
        dec_tps = float(timings.get("decode_tokens_per_sec", 0.0))
        pre_tps = float(timings.get("prefill_tokens_per_sec", 0.0))
    except (TypeError, ValueError):
        return ""
    return (
        f" prefill={prefill_ms / 1000.0:.1f}s/{pre_tps:.0f}tps "
        f"decode={decode_ms / 1000.0:.1f}s/{dec_tps:.0f}tps"
    )


def run_case(
    url: str,
    case: dict[str, Any],
    timeout_s: int,
    max_tokens: int,
    think: bool,
    model: str = "luce-dflash",
    auth_header: str = "",
) -> dict[str, Any]:
    """Send one case to the server.

    The wire payload is standard OpenAI ``/v1/chat/completions``: a
    ``thinking: {type: "enabled"}`` toggle (Anthropic-shaped opt-in for
    servers that implement a thinking-budget policy) plus a combined
    ``max_tokens`` cap. The actual think-vs-reply split is server config
    (`--think-max-tokens` on dflash; servers that don't speak split
    budgets — ds4_server, OpenRouter, vLLM — just apply ``max_tokens``
    as a single cap). Per-case ``max_tokens`` overrides still win.

    ``model`` sets the request body's model field (defaults to
    ``luce-dflash`` for our server; pass e.g. ``qwen/qwen3.6-27b`` for a
    gateway like OpenRouter). ``auth_header`` is the full header value
    (``Bearer sk-or-...``); empty means no auth.
    """
    prompt = build_prompt(case)
    request_max_tokens = int(case.get("max_tokens", max_tokens))
    body_payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {"role": "system", "content": case.get("system_prompt", SYSTEM_PROMPT)},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "top_p": 1.0,
        "max_tokens": request_max_tokens,
        "stream": False,
        "chat_template_kwargs": {"enable_thinking": think},
    }
    # Send the thinking control field for both modes. chat_template_kwargs
    # is only a Qwen-template hint and servers like antirez/ds4_server.c
    # don't read it, so --no-think previously left ds4-server defaulting to
    # DS4_THINK_HIGH and silently re-enabling thinking. Explicit type makes
    # the opt-out reach servers that follow the Anthropic-shaped contract.
    body_payload["thinking"] = {"type": "enabled" if think else "disabled"}
    body = json.dumps(body_payload).encode()
    headers = {"Content-Type": "application/json"}
    if auth_header:
        headers["Authorization"] = auth_header
    req = urllib.request.Request(
        url + "/v1/chat/completions",
        data=body,
        headers=headers,
    )

    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            data = json.loads(resp.read())
            http_status = resp.status
    except Exception as e:
        # Distinguish timeout from other errors so the JSON row carries a
        # clean flag rather than forcing readers to grep the error string.
        # urllib raises socket.timeout (wrapped by urllib.error.URLError
        # in older releases) for read-timeouts; the message also contains
        # "timed out" / "timeout".
        import socket
        err_str = str(e)
        timed_out = (
            isinstance(e, socket.timeout)
            or "timed out" in err_str.lower()
            or "timeout" in err_str.lower()
        )
        return {
            "source": case["source"],
            "id": case["id"],
            "name": f"{case['source']}/{case['id']}",
            "status": "error",
            "ok": False,
            "error": err_str,
            "timed_out": timed_out,
            "wall_s": round(time.perf_counter() - t0, 3),
            "prompt": prompt,
            "output": "",
        }

    wall = time.perf_counter() - t0
    choices = data.get("choices") or []
    choice = (choices[0] if choices else {}) or {}
    msg = choice.get("message") or {}
    output = msg.get("content") or ""
    # Multi-dialect reasoning extraction — see
    # docs/specs/thinking-budget.md "Bench fallback chain".
    #   reasoning_content : DeepSeek R1 / dflash primary
    #   reasoning         : OpenRouter / Anthropic-gateway flat
    #   reasoning_details : typed-block list (Anthropic / OR-structured)
    reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
    if not reasoning:
        details = msg.get("reasoning_details") or []
        if isinstance(details, list):
            reasoning = "\n".join(
                d.get("text", "")
                for d in details
                if isinstance(d, dict) and d.get("type") == "reasoning.text"
            )
    got = find_answer_with_fallback(case, output, reasoning)
    expected = expected_answers(case)
    grade = grade_case(case, got, output, reasoning)
    usage = data.get("usage") or {}
    # finish_details is the server's ds4-style close-info record. Only
    # populated when the client opts into the thinking budget. Surfacing it
    # in the per-case row lets the trace explain why a turn ended (model
    # closed </think> naturally vs server force-closed at the budget).
    finish_details = choice.get("finish_details") or {}
    # OpenAI/Anthropic/OR convention: usage.completion_tokens_details has
    # a reasoning_tokens field for thinking-mode models. Fall back to it
    # when the server didn't emit finish_details (most non-dflash gateways).
    or_ct_details = (usage.get("completion_tokens_details") or {})
    or_reasoning_tokens = or_ct_details.get("reasoning_tokens")
    # Derive content_tokens when usage gives us total + reasoning split.
    or_content_tokens = None
    if or_reasoning_tokens is not None and usage.get("completion_tokens") is not None:
        or_content_tokens = int(usage["completion_tokens"]) - int(or_reasoning_tokens)
    thinking_tokens_final = (
        finish_details.get("thinking_tokens")
        if finish_details.get("thinking_tokens") is not None
        else or_reasoning_tokens
    )
    content_tokens_final = (
        finish_details.get("content_tokens")
        if finish_details.get("content_tokens") is not None
        else or_content_tokens
    )
    # usage.timings — per-request prefill/decode wall-clock breakdown
    # emitted by dflash_server (spec §6.3). Absent for backends that
    # don't surface timings (OpenRouter, older sindri binaries); the
    # bench printout and JSON degrade gracefully when None.
    raw_timings = usage.get("timings")
    timings = raw_timings if isinstance(raw_timings, dict) else None
    # Derive prefill_tokens_per_sec from prompt_tokens / prefill_ms when both
    # are present. Server's usage.timings emits decode_tokens_per_sec but not
    # the prefill side (spec §6.3); deriving it here keeps the row uniform.
    if isinstance(timings, dict):
        pt = usage.get("prompt_tokens")
        pms = timings.get("prefill_ms")
        if pt and pms and float(pms) > 0:
            timings = {**timings,
                       "prefill_tokens_per_sec": round(pt * 1000.0 / float(pms), 1)}
    # OpenRouter attribution: which provider actually served this request,
    # what versioned model id was used, and the inference cost. None of
    # these are emitted by dflash_server (we know our own provider/model)
    # but the OR JSON exposes them and we were silently discarding them.
    # Keeping them per-row lets cross-provider quant analysis attribute
    # accuracy / latency to the actual fp8 vs bf16 routing decision.
    or_provider = data.get("provider")
    or_model_version = data.get("model")
    or_cost = usage.get("cost")
    return {
        "area": case["area"],
        "source": case["source"],
        "id": case["id"],
        "name": f"{case['source']}/{case['id']}",
        "domain": case.get("domain"),
        "title": case.get("title"),
        "ds4_index": case.get("ds4_index"),
        "kind": case["kind"],
        **grade,
        "http_status": http_status,
        "finish_reason": choice.get("finish_reason"),
        "close_kind": finish_details.get("close_kind"),
        "thinking_tokens": thinking_tokens_final,
        "content_tokens": content_tokens_final,
        "provider": or_provider,
        "model_version": or_model_version,
        "cost_usd": or_cost,
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "timings": timings,
        "timed_out": False,
        "given": got,
        "correct": expected,
        "wall_s": round(wall, 3),
        "prompt": prompt,
        "output": output,
        "reasoning_content": reasoning,
    }


def write_trace(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        f.write("# lucebox HTTP capability trace\n")
        for idx, row in enumerate(rows, start=1):
            f.write(
                f"\n===== CASE {idx} {row['source']}/{row['id']} =====\n"
                f"ds4_index: {row.get('ds4_index')}\n"
                f"status: {row['status']}\n"
                f"http_status: {row.get('http_status')}\n"
                f"finish_reason: {row.get('finish_reason')}\n"
                f"given: {row.get('given', '?')}\n"
                f"correct: {row.get('correct', '?')}\n"
                f"format_pass: {row.get('format_pass')}\n"
                f"semantic_hint: {row.get('semantic_hint')}\n"
                f"semantic_pass: {row.get('semantic_pass')}\n"
                f"strict_pass: {row.get('strict_pass')}\n"
                f"graded_pass: {row.get('graded_pass')}\n"
                f"prompt_tokens: {row.get('prompt_tokens')}\n"
                f"completion_tokens: {row.get('completion_tokens')}\n"
                "PROMPT_BEGIN\n"
                f"{row.get('prompt', '')}\n"
                "PROMPT_END\n"
                "MODEL_OUTPUT_BEGIN\n"
                f"{row.get('output', '')}\n"
                "MODEL_OUTPUT_END\n"
            )
            reasoning = row.get("reasoning_content") or ""
            if reasoning:
                f.write(
                    "MODEL_REASONING_BEGIN\n"
                    f"{reasoning}\n"
                    "MODEL_REASONING_END\n"
                )


def _forge_anthropic_finish_reason(stop_reason: str | None) -> str | None:
    """Map Anthropic ``stop_reason`` → ds4-eval-style ``finish_reason`` token.

    Anthropic emits ``end_turn`` / ``tool_use`` / ``max_tokens`` / ``stop_sequence``.
    Translate to the OpenAI lexicon used by ds4-eval rows so downstream
    consumers (trace dump, dashboards) don't have to special-case the area.
    """
    if stop_reason is None:
        return None
    mapping = {
        "end_turn": "stop",
        "stop_sequence": "stop",
        "max_tokens": "length",
        "tool_use": "tool_calls",
    }
    return mapping.get(stop_reason, stop_reason)


def _forge_extract_timings(raw_usage: dict[str, Any] | None) -> dict[str, Any] | None:
    """Pull dflash's ``usage.timings`` sub-block from a raw anthropic Message dump.

    Our cpp-server (post-#37) emits ``usage.timings`` with ``prefill_ms``,
    ``decode_ms``, ``decode_tokens_per_sec`` etc. OR/Anthropic backends
    don't surface this — return None so consumers degrade gracefully.
    """
    if not isinstance(raw_usage, dict):
        return None
    t = raw_usage.get("timings")
    if not isinstance(t, dict):
        return None
    # Carry the keys verbatim plus a derived prefill_tokens_per_sec when
    # we have prompt_tokens and prefill_ms — mirrors the ds4-eval path.
    out = dict(t)
    pt = raw_usage.get("input_tokens")
    pms = out.get("prefill_ms")
    try:
        if pt and pms and float(pms) > 0:
            out["prefill_tokens_per_sec"] = round(pt * 1000.0 / float(pms), 1)
    except (TypeError, ValueError):
        pass
    return out


def _forge_aggregate_timings(per_call: list[dict[str, Any] | None]) -> dict[str, Any] | None:
    """Sum prefill_ms / decode_ms across iterations; weighted-mean tok/s.

    Returns None when no iteration reports timings. Backends that only
    emit timings on a subset of calls still produce a usable aggregate
    over the calls that did report.
    """
    have_any = False
    pf_ms = 0.0
    de_ms = 0.0
    dec_tok = 0.0  # estimated from decode_tokens_per_sec * decode_ms / 1000
    pf_tok = 0.0
    for t in per_call:
        if not isinstance(t, dict):
            continue
        have_any = True
        try:
            pf_ms += float(t.get("prefill_ms") or 0.0)
            de_ms += float(t.get("decode_ms") or 0.0)
            dtps = float(t.get("decode_tokens_per_sec") or 0.0)
            ptps = float(t.get("prefill_tokens_per_sec") or 0.0)
            dec_tok += dtps * float(t.get("decode_ms") or 0.0) / 1000.0
            pf_tok += ptps * float(t.get("prefill_ms") or 0.0) / 1000.0
        except (TypeError, ValueError):
            continue
    if not have_any:
        return None
    agg: dict[str, Any] = {
        "prefill_ms": round(pf_ms, 3),
        "decode_ms": round(de_ms, 3),
    }
    if de_ms > 0 and dec_tok > 0:
        agg["decode_tokens_per_sec"] = round(dec_tok * 1000.0 / de_ms, 1)
    if pf_ms > 0 and pf_tok > 0:
        agg["prefill_tokens_per_sec"] = round(pf_tok * 1000.0 / pf_ms, 1)
    return agg


def run_forge_area(
    url: str,
    *,
    model: str,
    max_tokens: int,
    timeout_s: int,
    auth_header: str,
    tags: list[str] | None,
    names: list[str] | None,
    questions: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run antoinezambelli/forge tool-calling scenarios via /v1/messages.

    Lazy-imports the vendored ``forge_eval`` package (which carries its
    own ``_forge`` runtime) so that other areas (smoke, ds4-eval, long)
    keep working when the ``anthropic`` SDK is not installed. Builds a
    *recording* subclass of ``AnthropicClient`` pointed at ``url``
    (the Anthropic SDK appends ``/v1/messages``) so we can intercept the
    raw per-call API response (stop_reason, usage, usage.timings, raw
    content blocks) BEFORE forge collapses it into the parsed
    ``LLMResponse``. Each scenario is then driven through
    ``forge_eval.eval_runner.run_scenario`` so its iteration log can be
    isolated and aggregated into the enriched row.

    Returns ``(rows, forge_block)`` where ``rows`` carries the same
    ds4-eval-shaped fields (http_status, finish_reason, prompt_tokens,
    completion_tokens, timings, prompt, output, …) PLUS a per-call
    ``iterations[]`` breakdown. ``forge_block`` keeps the older
    forge-specific summary for backward compatibility with existing
    ``--json-out`` consumers.
    """
    import asyncio
    import json as _json
    import time as _time

    # Vendored fixtures live next to this script. The path append is
    # local to this function so other areas don't see the forge_eval
    # namespace even if the import fails.
    fixtures_dir = SCRIPT_DIR / "fixtures"
    if str(fixtures_dir) not in sys.path:
        sys.path.insert(0, str(fixtures_dir))

    try:
        # AnthropicClient lives inside the vendored ``_forge`` tree —
        # see dflash/scripts/fixtures/forge_eval/_forge/. The pypi
        # ``forge-guardrails`` package is no longer required; only the
        # ``anthropic`` SDK needs to be installed.
        from forge_eval._forge.clients.anthropic import (  # type: ignore[import-not-found]
            AnthropicClient,
        )
        from forge_eval._forge.core.workflow import (  # type: ignore[import-not-found]
            TextResponse,
        )
    except ImportError as exc:
        raise SystemExit(
            "[capability] --area forge requires the ``anthropic`` SDK. "
            "Install via: pip install -e 'dflash[eval]' "
            f"(import failed: {exc})"
        ) from exc

    try:
        from forge_eval.eval_runner import (  # type: ignore[import-not-found]
            ALL_SCENARIOS,
            EvalConfig,
            RunResult,
            run_scenario,
        )
    except ImportError as exc:
        raise SystemExit(
            "[capability] --area forge requires the vendored "
            "scripts/fixtures/forge_eval/ tree to be present "
            f"(import failed: {exc})"
        ) from exc

    api_key = "dummy"
    if auth_header:
        # If the caller supplied --auth-env, forward the underlying token
        # without the ``Bearer `` prefix — the Anthropic SDK sets the
        # ``x-api-key`` header itself, so we just need a non-empty value.
        api_key = auth_header.removeprefix("Bearer ").strip() or "dummy"

    # ── Recording client ──────────────────────────────────────────────
    # Subclass AnthropicClient and override ``send`` so we can record the
    # raw SDK Message (stop_reason, usage with timings, content blocks)
    # BEFORE _parse_response collapses it to a forge LLMResponse.
    # The bench's ds4-eval path uses urllib + direct JSON access; here
    # we ride the Anthropic SDK so we must reach for ``model_dump()`` /
    # attribute access on the typed Message.
    class _RecordingAnthropicClient(AnthropicClient):  # type: ignore[misc, valid-type]
        """AnthropicClient that records every send() into ``iteration_log``.

        Each entry is a dict with: wall_s, http_status (200 on success,
        BackendError.code on failure), finish_reason (OpenAI lexicon),
        stop_reason (raw Anthropic), prompt_tokens, completion_tokens,
        tool_calls (list of {name, arguments}), prompt (the messages
        we sent, serialized), output (text content concat), reasoning
        (text emitted alongside tool_use blocks — Anthropic puts the
        "thinking-out-loud" text in plain text blocks ahead of the
        tool_use), timings (dflash usage.timings or None), error (str
        when send raised), raw_usage (the full usage dict for forensic
        re-grading).
        """

        def __init__(self, *a: Any, **kw: Any) -> None:
            super().__init__(*a, **kw)
            self.iteration_log: list[dict[str, Any]] = []

        def reset_log(self) -> None:
            """Clear iteration log before driving the next scenario."""
            self.iteration_log.clear()

        async def send(  # type: ignore[override]
            self,
            messages: list[dict[str, Any]],
            tools: Any = None,
            sampling: dict[str, Any] | None = None,
            passthrough: dict[str, Any] | None = None,
            inbound_anthropic_body: dict[str, Any] | None = None,
        ) -> Any:
            # Snapshot what forge is about to send so we can attribute
            # the request-shape (incl. tool-call history) to this
            # iteration. Best-effort serialize — anthropic-shape blocks
            # are JSON-safe dicts of strings.
            try:
                prompt_blob = _json.dumps(messages, ensure_ascii=False, default=str)
            except Exception:
                prompt_blob = str(messages)

            # Reach past the parent's send() so we can grab the raw SDK
            # Message before _parse_response throws away usage / stop_reason.
            from forge_eval._forge.errors import BackendError  # type: ignore[import-not-found]
            import anthropic as _anthropic  # type: ignore[import-not-found]

            kwargs = self._build_kwargs(
                messages, tools, passthrough, inbound_anthropic_body,
            )
            t0 = _time.perf_counter()
            record: dict[str, Any] = {
                "wall_s": 0.0,
                "http_status": None,
                "finish_reason": None,
                "stop_reason": None,
                "prompt_tokens": None,
                "completion_tokens": None,
                "tool_calls": [],
                "prompt": prompt_blob,
                "output": "",
                "reasoning_content": "",
                "timings": None,
                "raw_usage": None,
                "error": None,
            }
            try:
                response = await self._client.messages.create(**kwargs)
            except _anthropic.APIError as exc:
                record["wall_s"] = round(_time.perf_counter() - t0, 4)
                record["http_status"] = getattr(exc, "status_code", 0) or 0
                record["error"] = f"{type(exc).__name__}: {exc}"
                self.iteration_log.append(record)
                # Preserve forge's error contract — the runner expects
                # BackendError to surface as a ForgeError so it lands in
                # RunResult.error_type. Don't bury a 5xx as a silent FAIL.
                raise BackendError(
                    getattr(exc, "status_code", 0), str(exc)
                ) from exc

            record["wall_s"] = round(_time.perf_counter() - t0, 4)
            record["http_status"] = 200

            # Pull the typed fields off the SDK Message. usage.input_tokens
            # / output_tokens are stable; ``stop_reason`` is the Anthropic
            # close lexicon. ``model_dump()`` is the safe path for the
            # nested usage.timings the cpp-server emits (the SDK won't
            # know about that custom field so attribute access fails).
            try:
                record["prompt_tokens"] = int(response.usage.input_tokens)
                record["completion_tokens"] = int(response.usage.output_tokens)
            except (AttributeError, TypeError, ValueError):
                pass
            stop_reason = getattr(response, "stop_reason", None)
            record["stop_reason"] = stop_reason
            record["finish_reason"] = _forge_anthropic_finish_reason(stop_reason)
            raw_usage: dict[str, Any] | None = None
            try:
                dumped = response.model_dump()
                raw_usage = dumped.get("usage") if isinstance(dumped, dict) else None
            except Exception:
                raw_usage = None
            record["raw_usage"] = raw_usage
            record["timings"] = _forge_extract_timings(raw_usage)

            # Walk the content blocks to split text (reasoning when
            # paired with tool_use) and tool_calls.
            text_parts: list[str] = []
            tool_calls_out: list[dict[str, Any]] = []
            tool_uses_present = False
            for block in getattr(response, "content", []) or []:
                btype = getattr(block, "type", None)
                if btype == "tool_use":
                    tool_uses_present = True
                    args = getattr(block, "input", None)
                    if not isinstance(args, dict):
                        try:
                            args = dict(args or {})
                        except Exception:
                            args = {}
                    tool_calls_out.append({
                        "id": getattr(block, "id", None),
                        "name": getattr(block, "name", None),
                        "arguments": args,
                    })
                elif btype == "text":
                    text_parts.append(getattr(block, "text", "") or "")
            text_joined = "\n".join(text_parts)
            record["tool_calls"] = tool_calls_out
            if tool_uses_present:
                # Anthropic shape: text alongside tool_use is the model's
                # narration / chain-of-thought for the call. Mirror the
                # ds4-eval convention and surface it as reasoning_content
                # so consumers can grep both areas the same way.
                record["reasoning_content"] = text_joined
                record["output"] = ""
            else:
                record["output"] = text_joined

            self.iteration_log.append(record)

            # Update last_usage so forge's context manager keeps working
            # — see AnthropicClient.send for the source-of-truth shape.
            from forge_eval._forge.clients.base import TokenUsage  # type: ignore[import-not-found]
            try:
                self.last_usage = {
                    0: TokenUsage(
                        prompt_tokens=int(response.usage.input_tokens),
                        completion_tokens=int(response.usage.output_tokens),
                        total_tokens=int(response.usage.input_tokens)
                        + int(response.usage.output_tokens),
                    )
                }
            except (AttributeError, TypeError, ValueError):
                pass
            return self._parse_response(response)

    client = _RecordingAnthropicClient(
        model=model,
        api_key=api_key,
        base_url=url.rstrip("/"),
        max_tokens=max_tokens,
        timeout=float(timeout_s),
        max_retries=0,  # let bench surface errors directly; no retry storms
    )

    all_scenarios = list(ALL_SCENARIOS)
    # Apply tag and name filters here (forge's run_eval did this internally
    # but we drive run_scenario directly to isolate the iteration log per
    # scenario; reuse the same selection rules).
    scenarios = all_scenarios
    if tags:
        scenarios = [s for s in scenarios if any(t in s.tags for t in tags)]
    if names:
        scenarios = [s for s in scenarios if s.name in names]
    if questions is not None and questions >= 0:
        scenarios = scenarios[:questions]

    config = EvalConfig(
        runs_per_scenario=1,  # one shot per scenario, like ds4-eval
        stream=False,
        keep_message_history=False,  # we don't surface raw forge messages
        verbose=False,
        # No global budget override — let scenarios use their own defaults.
        # Anthropic backends don't need a ServerManager-resolved budget.
    )

    print(
        f"[capability] forge scenarios={len(scenarios)} "
        f"tags={tags or 'all'} names={names or 'all'}",
        flush=True,
    )

    # Drive scenarios one at a time so each gets its own iteration log
    # snapshot. forge's run_scenario already runs a single scenario once
    # — we just have to reset the recorder between calls.
    async def _drive_all() -> list[tuple[Any, "RunResult", list[dict[str, Any]]]]:
        out: list[tuple[Any, RunResult, list[dict[str, Any]]]] = []
        for scenario in scenarios:
            client.reset_log()
            result = await run_scenario(client, scenario, config, ablation=None)
            # Snapshot the log NOW — the next scenario will mutate it.
            iter_log = [dict(r) for r in client.iteration_log]
            status = (
                "OK" if result.completeness and result.accuracy is not False
                else ("OK (incorrect)" if result.completeness else f"FAIL ({result.error_type})")
            )
            print(
                f"  {scenario.name}: {status} — "
                f"{result.iterations_used} iterations, "
                f"{result.elapsed_seconds:.1f}s",
                flush=True,
            )
            out.append((scenario, result, iter_log))
        return out

    driven = asyncio.run(_drive_all())

    rows: list[dict[str, Any]] = []
    forge_scenarios: list[dict[str, Any]] = []
    for idx, (scenario, result, iter_log) in enumerate(driven, start=1):
        scenario_name = scenario.name
        if result is None:
            graded_pass = False
            error_type = "no_runs"
            row_status = "error"
            elapsed = 0.0
            iterations = 0
            accuracy = None
            completeness = False
            error_message = None
        else:
            completeness = bool(result.completeness)
            # forge grades pass = completed AND validator did not return
            # False. ``accuracy=None`` means the scenario has no validator
            # (treat as passed if completeness is True).
            accuracy = result.accuracy
            graded_pass = completeness and (accuracy is not False)
            error_type = result.error_type
            error_message = result.error_message
            elapsed = result.elapsed_seconds
            iterations = result.iterations_used
            if not completeness:
                row_status = "error"
            elif accuracy is False:
                row_status = "failed"
            else:
                row_status = "passed"

        # Aggregate iteration-level metrics into the scenario row.
        prompt_tokens_sum = sum(
            int(r.get("prompt_tokens") or 0) for r in iter_log
        )
        completion_tokens_sum = sum(
            int(r.get("completion_tokens") or 0) for r in iter_log
        )
        # Forge runs without thinking (--no-think); the Anthropic shape
        # doesn't split reasoning tokens out of completion_tokens. Until
        # we wire a thinking-aware count, the conservative report is
        # thinking_tokens=0 and content_tokens=completion_tokens.
        thinking_tokens_sum = 0
        content_tokens_sum = completion_tokens_sum

        agg_timings = _forge_aggregate_timings([r.get("timings") for r in iter_log])

        # Final-iteration top-level fields. Empty when iter_log is empty
        # (e.g. import-time failure before the first call landed).
        final = iter_log[-1] if iter_log else {}

        # Build per-iteration rows. Strip raw_usage from the public
        # iteration shape — it's redundant once we've extracted timings
        # and (prompt|completion)_tokens, and bloats the JSON. Keep
        # everything else.
        iterations_out: list[dict[str, Any]] = []
        for i, r in enumerate(iter_log, start=1):
            iterations_out.append({
                "i": i,
                "wall_s": r.get("wall_s"),
                "http_status": r.get("http_status"),
                "finish_reason": r.get("finish_reason"),
                "stop_reason": r.get("stop_reason"),
                "prompt_tokens": r.get("prompt_tokens"),
                "completion_tokens": r.get("completion_tokens"),
                "tool_calls": r.get("tool_calls") or [],
                "timings": r.get("timings"),
                "reasoning_content": r.get("reasoning_content") or "",
                "prompt": r.get("prompt") or "",
                "output": r.get("output") or "",
                "error": r.get("error"),
            })

        rows.append({
            # Identity (unchanged)
            "area": "forge",
            "source": "forge-guardrails@0.7.1-vendored",
            "id": scenario_name,
            "name": f"forge/{scenario_name}",
            "kind": "tool-calling",
            # Aggregate verdict (unchanged)
            "status": row_status,
            "ok": graded_pass,
            "graded_pass": graded_pass,
            "strict_pass": graded_pass,
            "format_pass": completeness,
            "semantic_hint": False,
            "semantic_pass": False,
            "given": "PASS" if graded_pass else (error_type or "FAIL"),
            "correct": ["PASS"],
            # Aggregate timing / tokens — now actually populated.
            "wall_s": round(elapsed, 3),
            "prompt_tokens": prompt_tokens_sum or None,
            "completion_tokens": completion_tokens_sum or None,
            "thinking_tokens": thinking_tokens_sum,
            "content_tokens": content_tokens_sum or None,
            "iterations_used": iterations,
            "timings": agg_timings,
            "timed_out": (error_type in {"TimeoutError", "ReadTimeoutError"}),
            "finish_reason": final.get("finish_reason"),
            # forge doesn't use the dflash thinking-budget close path,
            # so close_kind is always null. Kept for ds4-eval schema parity.
            "close_kind": None,
            "http_status": final.get("http_status"),
            "reasoning_content": "\n".join(
                r.get("reasoning_content") or "" for r in iter_log
            ).strip(),
            # Final-iteration content (existing fields, now populated).
            "prompt": final.get("prompt") or "",
            "output": final.get("output") or "",
            # NEW: per-call detail.
            "iterations": iterations_out,
        })
        forge_scenarios.append({
            "name": scenario_name,
            "completeness": completeness,
            "accuracy": accuracy,
            "graded_pass": graded_pass,
            "iterations_used": iterations,
            "error_type": error_type,
            "error_message": error_message,
            "elapsed_seconds": round(elapsed, 3),
            "input_tokens": getattr(result, "input_tokens", 0) if result else 0,
            "output_tokens": getattr(result, "output_tokens", 0) if result else 0,
            "stream_retries": getattr(result, "stream_retries", 0) if result else 0,
        })

    forge_block = {
        "scenarios": forge_scenarios,
        "client": "forge_eval._forge.clients.anthropic.AnthropicClient",
        "endpoint": f"{url.rstrip('/')}/v1/messages",
        "runs_per_scenario": 1,
        "tags_filter": tags,
        "names_filter": names,
    }
    return rows, forge_block


def main() -> int:
    ap = argparse.ArgumentParser(description="Run graded HTTP capability prompts.")
    ap.add_argument("--url", default="http://127.0.0.1:8000", help="Server base URL.")
    ap.add_argument(
        "--area",
        choices=("smoke", "ds4-eval", "long", "forge", "all"),
        default="smoke",
        help=(
            "Case area to run. ``forge`` drives forge-guardrails' "
            "tool-calling scenarios (vendored at "
            "scripts/fixtures/forge_eval/) via /v1/messages. The "
            "``eval`` extra of dflash/pyproject.toml (anthropic SDK) "
            "must be installed."
        ),
    )
    ap.add_argument("--questions", type=int, default=None,
                    help="Run only the first N selected questions.")
    ap.add_argument("--case-index", type=int, default=None,
                    help="Run one ds4-eval case by its 1-based upstream index.")
    ap.add_argument("--case-id", default="",
                    help="Run one case by its upstream id.")
    ap.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Maximum visible reply tokens. Defaults to 4096 for ds4-eval/all, else 512.",
    )
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument(
        "--min-pass-rate",
        type=float,
        default=None,
        help=(
            "Minimum pass rate for exit 0. Defaults to 1.0 for smoke and 0.0 "
            "for score-only ds4-eval/long areas."
        ),
    )
    ap.add_argument(
        "--think",
        dest="think",
        action="store_true",
        default=None,
        help="Enable Qwen thinking mode. Defaults on for ds4-eval/all.",
    )
    ap.add_argument(
        "--no-think",
        dest="think",
        action="store_false",
        help="Disable Qwen thinking mode even for ds4-eval/all.",
    )
    ap.add_argument(
        "--model",
        default="luce-dflash",
        help=(
            "Value for the `model` field in the request body. Defaults to "
            "lucebox's `luce-dflash`. Override when pointing at a gateway "
            "(e.g. `qwen/qwen3-72b-instruct` for OpenRouter)."
        ),
    )
    ap.add_argument(
        "--auth-env",
        default="",
        help=(
            "Env var whose value to send as a Bearer token in the "
            "Authorization header (e.g. OPENROUTER_API_KEY). Empty disables "
            "auth. Reading from env keeps the secret out of process listings "
            "and trace files."
        ),
    )
    ap.add_argument(
        "--forge-tags",
        default="",
        help=(
            "Comma-separated forge scenario tag filter (e.g. "
            "``plumbing,model_quality``). Maps to forge's run_eval(tags=).  "
            "Only meaningful with --area forge."
        ),
    )
    ap.add_argument(
        "--forge-scenario",
        default="",
        help=(
            "Comma-separated forge scenario name filter (e.g. "
            "``basic_2step,sequential_3step``). Maps to run_eval(names=). "
            "Only meaningful with --area forge."
        ),
    )
    ap.add_argument("--json-out", type=Path)
    ap.add_argument("--trace", type=Path)
    ap.add_argument(
        "--parallel",
        type=int,
        default=1,
        help=(
            "Run up to N cases concurrently. Default 1 (sequential). Safe to "
            "raise for stateless HTTP gateways (OpenRouter); leave at 1 for "
            "single-GPU local servers since concurrent requests just queue."
        ),
    )
    args = ap.parse_args()
    min_pass_rate = (
        default_min_pass_rate(args.area)
        if args.min_pass_rate is None
        else float(args.min_pass_rate)
    )
    max_tokens = (
        default_max_tokens(args.area)
        if args.max_tokens is None
        else int(args.max_tokens)
    )
    think = (
        default_thinking_enabled(args.area)
        if args.think is None
        else bool(args.think)
    )

    auth_header = ""
    if args.auth_env:
        import os
        token = os.environ.get(args.auth_env, "")
        if not token:
            ap.error(f"--auth-env {args.auth_env}: env var is empty or unset")
        auth_header = f"Bearer {token}"

    # ``forge`` is dispatched separately — it doesn't share the
    # build_prompt/grade_case path. ``all`` runs the case-based areas
    # first, then appends the forge results.
    case_areas = {"smoke", "ds4-eval", "long", "all"}
    forge_areas = {"forge", "all"}

    if args.area in case_areas:
        selected = select_cases(
            args.area,
            args.questions,
            case_index=args.case_index,
            case_id=args.case_id or None,
        )
        if (args.case_index is not None or args.case_id) and not selected:
            ap.error("selected case was not found")
    else:
        selected = []
        if args.case_index is not None or args.case_id:
            ap.error(
                "--case-index / --case-id are not meaningful for "
                f"--area {args.area} (use --forge-scenario instead)"
            )

    rows: list[dict[str, Any]] = []
    forge_block: dict[str, Any] | None = None
    total_for_log = (
        len(selected) if args.area in case_areas else "forge"
    )
    print(
        f"[capability] url={args.url} area={args.area} questions={total_for_log}"
        + (f" parallel={args.parallel}"
           if args.parallel > 1 and args.area in case_areas else ""),
        flush=True,
    )

    def format_status(idx: int, case: dict[str, Any], row: dict[str, Any]) -> str:
        status = "PASS" if row["ok"] else "FAIL"
        correct = "|".join(expected_answers(case))
        timings_suffix = format_timings_suffix(row.get("timings"))
        # Thinking-token count from finish_details (qwen thinking opt-in);
        # None when not provided.
        tt = row.get("thinking_tokens")
        tt_suffix = f" thk={tt}" if tt is not None else ""
        # Timeout marker — flag explicitly so it stands out in the log.
        timeout_suffix = " TIMEOUT" if row.get("timed_out") else ""
        return (
            f"  {idx:2d} {status:4s} {case['source']:14s} {case['id']:20s} "
            f"given={row.get('given', '?')} correct={correct} "
            f"format={row.get('format_pass')} hint={row.get('semantic_hint')} "
            f"wall={row['wall_s']:.2f}s{timings_suffix}{tt_suffix}{timeout_suffix}"
        )

    # Case-area runner (smoke / ds4-eval / long / all). Parallel is opt-in
    # via --parallel and only applies here; the forge runner below has its
    # own scenario flow and stays sequential.
    if args.area in case_areas:
        if args.parallel > 1:
            # Parallel runner: stateless HTTP gateways (OpenRouter) can serve
            # many concurrent requests. Local single-GPU servers (dflash,
            # ds4-server) just queue them, so this stays opt-in via --parallel.
            # Output order is "as completed" so faster cases stream first; the
            # JSON rows are sorted back to selection order before write so the
            # snapshot matches the upstream eval_cases sequence.
            from concurrent.futures import ThreadPoolExecutor, as_completed
            with ThreadPoolExecutor(max_workers=args.parallel) as pool:
                future_to_meta = {
                    pool.submit(run_case, args.url, case, args.timeout,
                                max_tokens, think, model=args.model,
                                auth_header=auth_header):
                        (idx, case)
                    for idx, case in enumerate(selected, start=1)
                }
                for fut in as_completed(future_to_meta):
                    idx, case = future_to_meta[fut]
                    row = fut.result()
                    row["_idx"] = idx
                    rows.append(row)
                    print(format_status(idx, case, row), flush=True)
            rows.sort(key=lambda r: r["_idx"])
            for r in rows:
                r.pop("_idx", None)
        else:
            for idx, case in enumerate(selected, start=1):
                row = run_case(args.url, case, args.timeout, max_tokens,
                               think, model=args.model,
                               auth_header=auth_header)
                rows.append(row)
                print(format_status(idx, case, row), flush=True)

    if args.area in forge_areas:
        forge_tags = (
            [t.strip() for t in args.forge_tags.split(",") if t.strip()]
            or None
        )
        forge_names = (
            [n.strip() for n in args.forge_scenario.split(",") if n.strip()]
            or None
        )
        forge_rows, forge_block = run_forge_area(
            args.url,
            model=args.model,
            max_tokens=max_tokens,
            timeout_s=args.timeout,
            auth_header=auth_header,
            tags=forge_tags,
            names=forge_names,
            questions=args.questions if args.area == "forge" else None,
        )
        # Continue numbering across smoke/ds4 rows so ``--area all`` reads
        # like a single sequence.
        base = len(rows)
        rows.extend(forge_rows)
        for offset, row in enumerate(forge_rows, start=1):
            status = "PASS" if row["ok"] else "FAIL"
            # Forge's RunResult doesn't expose per-iteration server
            # timings (the AnthropicClient abstraction strips them);
            # format_timings_suffix() returns "" for None.
            timings_suffix = format_timings_suffix(row.get("timings"))
            print(
                f"  {base + offset:2d} {status:4s} {row['source']:14s} "
                f"{row['id']:30s} given={row.get('given', '?')} "
                f"wall={row['wall_s']:.2f}s{timings_suffix}",
                flush=True,
            )

    passed = sum(1 for row in rows if row["ok"])
    strict_passed = sum(1 for row in rows if row.get("strict_pass"))
    format_passed = sum(1 for row in rows if row.get("format_pass"))
    semantic_hints = sum(1 for row in rows if row.get("semantic_hint"))
    semantic_passed = sum(1 for row in rows if row.get("semantic_pass"))
    pass_rate = passed / len(rows) if rows else 0.0
    strict_pass_rate = strict_passed / len(rows) if rows else 0.0
    format_pass_rate = format_passed / len(rows) if rows else 0.0
    semantic_hint_rate = semantic_hints / len(rows) if rows else 0.0
    semantic_pass_rate = semantic_passed / len(rows) if rows else 0.0
    source = "lucebox-http-capability"
    if args.area == "ds4-eval":
        source = "antirez/ds4 ds4-eval HTTP port"
    elif args.area == "forge":
        source = "antoinezambelli/forge tool-calling eval"

    payload = {
        "suite": "capability",
        "area": args.area,
        "source": source,
        "max_tokens": max_tokens,
        "passed": passed,
        "graded_passed": passed,
        "strict_passed": strict_passed,
        "format_passed": format_passed,
        "semantic_hints": semantic_hints,
        "semantic_passed": semantic_passed,
        "total": len(rows),
        "pass_rate": round(pass_rate, 4),
        "graded_pass_rate": round(pass_rate, 4),
        "strict_pass_rate": round(strict_pass_rate, 4),
        "format_pass_rate": round(format_pass_rate, 4),
        "semantic_hint_rate": round(semantic_hint_rate, 4),
        "semantic_pass_rate": round(semantic_pass_rate, 4),
        "thinking_enabled": think,
        "min_pass_rate": min_pass_rate,
        "rows": rows,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    if forge_block is not None:
        payload["forge_results"] = forge_block
    print(f"[capability] pass_rate={pass_rate:.2%}", flush=True)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n")
    if args.trace:
        write_trace(args.trace, rows)
    return 0 if pass_rate >= min_pass_rate else 1


if __name__ == "__main__":
    raise SystemExit(main())
