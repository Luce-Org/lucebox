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
        "thinking_tokens": finish_details.get("thinking_tokens"),
        "content_tokens": finish_details.get("content_tokens"),
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
    keep working when the ``anthropic`` SDK is not installed. Builds an
    ``AnthropicClient`` pointed at ``url`` (the Anthropic SDK appends
    ``/v1/messages``), then runs
    each scenario once via ``forge_eval.eval_runner.run_eval``.

    Returns ``(rows, forge_block)`` where ``rows`` is the per-scenario
    summary in the shape ``bench_http_capability`` already uses for its
    table output and ``forge_block`` carries the raw forge-specific
    detail (token counts, iteration counts, error type, …) for
    ``--json-out`` consumers.
    """
    import asyncio
    import os

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
            run_eval,
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

    client = AnthropicClient(
        model=model,
        api_key=api_key,
        base_url=url.rstrip("/"),
        max_tokens=max_tokens,
        timeout=float(timeout_s),
        max_retries=0,  # let bench surface errors directly; no retry storms
    )

    scenarios = list(ALL_SCENARIOS)
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
    raw_results: dict[str, list[RunResult]] = asyncio.run(
        run_eval(
            client,
            scenarios,
            config,
            resolved_budget=None,
            tags=tags,
            names=names,
            ablation=None,
        )
    )

    rows: list[dict[str, Any]] = []
    forge_scenarios: list[dict[str, Any]] = []
    for idx, (scenario_name, run_list) in enumerate(raw_results.items(), start=1):
        # runs_per_scenario=1, so pick the first (and only) RunResult.
        result = run_list[0] if run_list else None
        if result is None:
            graded_pass = False
            error_type = "no_runs"
            row_status = "error"
            elapsed = 0.0
            iterations = 0
            accuracy = None
            completeness = False
        else:
            completeness = bool(result.completeness)
            # forge grades pass = completed AND validator did not return
            # False. ``accuracy=None`` means the scenario has no validator
            # (treat as passed if completeness is True).
            accuracy = result.accuracy
            graded_pass = completeness and (accuracy is not False)
            error_type = result.error_type
            elapsed = result.elapsed_seconds
            iterations = result.iterations_used
            if not completeness:
                row_status = "error"
            elif accuracy is False:
                row_status = "failed"
            else:
                row_status = "passed"

        rows.append({
            "area": "forge",
            "source": "forge-guardrails@0.7.1-vendored",
            "id": scenario_name,
            "name": f"forge/{scenario_name}",
            "kind": "tool-calling",
            "status": row_status,
            "ok": graded_pass,
            "graded_pass": graded_pass,
            "strict_pass": graded_pass,
            "format_pass": completeness,
            "semantic_hint": False,
            "semantic_pass": False,
            "given": "PASS" if graded_pass else (error_type or "FAIL"),
            "correct": ["PASS"],
            "wall_s": round(elapsed, 3),
            # forge's AnthropicClient doesn't propagate the server's
            # usage.timings into RunResult — keep the key for schema
            # parity with the ds4/smoke rows; format_timings_suffix()
            # omits the printed suffix when this is None.
            "timings": None,
            "prompt": "",
            "output": "",
        })
        forge_scenarios.append({
            "name": scenario_name,
            "completeness": completeness,
            "accuracy": accuracy,
            "graded_pass": graded_pass,
            "iterations_used": iterations,
            "error_type": error_type,
            "error_message": getattr(result, "error_message", None) if result else None,
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
