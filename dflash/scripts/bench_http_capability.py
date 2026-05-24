"""Small graded HTTP capability benchmark for lucebox.

The default area is `smoke`: short deterministic checks that should pass 100%
and are suitable for optimizer gating. `long` measures long-context
reliability. The `ds4-eval` area dispatches into the ported antirez/ds4
corpus living in `bench_ds4_eval.py`; ds4-specific data, graders, and
budget defaults stay colocated there so the upstream diff stays narrow.

Standalone `ds4-eval`, `long`, and `all` runs are score-only unless
`--min-pass-rate` is set explicitly.
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
    return DS4_EVAL_MAX_TOKENS if area in {"ds4-eval", "all"} else DEFAULT_MAX_TOKENS


def default_thinking_enabled(area: str) -> bool:
    return area in {"ds4-eval", "all"}


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
        return {
            "source": case["source"],
            "id": case["id"],
            "name": f"{case['source']}/{case['id']}",
            "status": "error",
            "ok": False,
            "error": str(e),
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


def main() -> int:
    ap = argparse.ArgumentParser(description="Run graded HTTP capability prompts.")
    ap.add_argument("--url", default="http://127.0.0.1:8000", help="Server base URL.")
    ap.add_argument(
        "--area",
        choices=("smoke", "ds4-eval", "long", "all"),
        default="smoke",
        help="Case area to run.",
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
    ap.add_argument("--json-out", type=Path)
    ap.add_argument("--trace", type=Path)
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

    selected = select_cases(
        args.area,
        args.questions,
        case_index=args.case_index,
        case_id=args.case_id or None,
    )
    if (args.case_index is not None or args.case_id) and not selected:
        ap.error("selected case was not found")
    rows: list[dict[str, Any]] = []
    print(f"[capability] url={args.url} area={args.area} questions={len(selected)}", flush=True)
    for idx, case in enumerate(selected, start=1):
        row = run_case(args.url, case, args.timeout, max_tokens, think,
                        model=args.model, auth_header=auth_header)
        rows.append(row)
        status = "PASS" if row["ok"] else "FAIL"
        correct = "|".join(expected_answers(case))
        print(
            f"  {idx:2d} {status:4s} {case['source']:14s} {case['id']:20s} "
            f"given={row.get('given', '?')} correct={correct} "
            f"format={row.get('format_pass')} hint={row.get('semantic_hint')} "
            f"wall={row['wall_s']:.2f}s",
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
    payload = {
        "suite": "capability",
        "area": args.area,
        "source": (
            "antirez/ds4 ds4-eval HTTP port"
            if args.area == "ds4-eval"
            else "lucebox-http-capability"
        ),
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
    print(f"[capability] pass_rate={pass_rate:.2%}", flush=True)
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n")
    if args.trace:
        write_trace(args.trace, rows)
    return 0 if pass_rate >= min_pass_rate else 1


if __name__ == "__main__":
    raise SystemExit(main())
