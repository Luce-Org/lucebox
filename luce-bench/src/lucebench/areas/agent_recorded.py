r"""Recorded-session agent probes for ``--areas agent_recorded``.

This module ships **two** evaluation areas mined from real Claude Code
and Codex sessions (see ``scripts/extract-agentic-fixture.py``):

1. ``agent_recorded`` — **sequential multi-turn replay with cache
   metrics + LLM judge** (default). Replays recorded sessions turn by
   turn as growing conversation prefixes, measures prefill/decode/cache
   timings for each request, and grades the final model response with a
   Claude Sonnet judge that compares against the reference assistant
   response. Fixture:
   ``fixtures/agent_recorded/multi_turn_cases.json`` (schema v1).

2. ``agent_recorded_v1`` — **single-turn tool-schema coverage**
   (deprecated). Sends just the first user message and pattern-matches
   the model's reply against the tools Claude actually used. Kept for
   back-compat with archived result snapshots. Fixture:
   ``fixtures/agent_recorded/cases.json``.

------------------------------------------------------------------------
Why two areas? Substring matching against Claude's tool names biases
against models that solve the task with a different but valid approach
(e.g. ``write_file`` vs ``Edit``). The new area uses Claude itself as
the judge, with explicit license to mark "divergent but valid" as a
pass. See :mod:`lucebench.grading.llm_judge`.

Cost: one full pass over the 48-case fixture is roughly $0.15 -- $0.75
in Anthropic billing (Sonnet 4.6 list pricing, ~5K input + ~150 output
per judge call × 48 final-turn calls). The judge cache zeroes-out
re-runs with identical inputs (see ``~/.cache/luce-bench/judge/``).

Replay modes
============
By default each case runs in ``sequential`` mode:

* The service is optionally restarted once before a recorded session.
* The first request sends messages through the first user turn.
* Each later request sends the same conversation extended through the
  next user turn, using the recorded assistant/tool transcript between
  turns so the prefix is deterministic.

This matches long-running agent traffic where request N+1 shares most
of request N as a stable prefix, and therefore exercises prefix-cache
behavior. Rows include the per-turn timings under ``turns`` plus legacy
``cold`` / ``warm`` aliases for first/final request compatibility.

The ``exact-repeat`` mode is retained for full-prompt/prefill cache
checks: it sends the same complete ``messages`` list twice and reports
``cold.prefill_s / warm.prefill_s`` as ``cache_speedup_x``.

If ``systemctl`` isn't available (CI, non-systemd hosts), the runner
prints a warning and runs without a cold restart. The sequential cache
metrics still reflect growing-prompt reuse inside that run, but cache
state may already be warm before the first turn.

Constraints
===========
* Does NOT modify systemd state files or the installed lucebox binary.
  It only invokes ``systemctl --user restart`` and polls ``/health``
  on the configured URL. The user's running service is the one being
  benchmarked — that's the intended use case.
* ``ANTHROPIC_API_KEY`` must be set for the judge to run. Missing key
  → ``JudgeUnavailable`` raised on first case; the area aborts cleanly.

Schema migration
================
The new fixture schema is
``lucebox-bench-agent-recorded-multi-turn-v1``. v0 (no
``reference_response``) is loadable for back-compat: cases without a
reference get ``judge_pending`` and don't contribute to the pass rate.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

# See lucebench.areas.ds4_eval.GRADER_VERSION for the bump policy.
GRADER_VERSION = 2

SCRIPT_DIR = Path(__file__).resolve().parent.parent
FIXTURE_PATH = SCRIPT_DIR / "fixtures" / "agent_recorded" / "cases.json"
MULTI_TURN_FIXTURE_PATH = SCRIPT_DIR / "fixtures" / "agent_recorded" / "multi_turn_cases.json"
AGENT_RECORDED_REPLAY_MODES = ("sequential", "exact-repeat")
_PREFILL_SPEEDUP_MIN_WARM_S = 0.001

# ── v1 (legacy single-turn) constants — see module docstring. ────────
#
# These all support the ``agent_recorded_v1`` area now. The v0 grader
# matched the model's text against the tools Claude actually used; that
# biases against valid non-Claude approaches, which is why the new
# default area exists. We keep this code unchanged for back-compat with
# archived result snapshots that referenced it.

# Tool name → set of phrases that count as "the model meant this tool".
# Intentionally loose so a model that says "use a Bash command" passes
# even though it didn't literally say "Bash". Keys must match the
# canonical names emitted by the collector (Claude tool names + the
# normalized Codex shapes — see ``scripts/extract-agentic-fixture.py``).
_TOOL_SYNONYMS: dict[str, tuple[str, ...]] = {
    "Bash": (
        "bash", "shell command", "shell", "run a command",
        "execute a command", "command line",
        "exec_command", "exec-command", "shell-exec", "run_shell",
        "run-script", "exec_shell",
    ),
    "Read": (
        "read the file", "read file", "open the file", "view the file",
        "cat ", "look at",
        "read_file", "read-file", "readfile", "fs.read", "fs:read", "open_file",
    ),
    "Edit": (
        "edit the file", "modify the file", "edit ", "change the file",
        "patch the file",
        "edit_file", "edit-file", "modify_file", "modify-file", "fs.edit",
    ),
    "Write": (
        "write the file", "create the file", "create a file", "write a new file",
        "write_file", "write-file", "create_file", "create-file", "fs.write",
    ),
    "Grep": (
        "grep", "search for", "search the code", "ripgrep", "rg ",
        "grep_files", "grep-files", "search_code", "search-code",
    ),
    "Glob": (
        "glob", "find files", "list files matching",
        "list_files", "list-files", "ls_files", "ls-files", "ls ",
        "find_files", "find-files", "readdir",
    ),
    "MultiEdit": ("multiple edits", "multi-edit", "multi_edit"),
    "NotebookEdit": ("notebook edit", "edit the notebook", "notebook_edit"),
    "WebFetch": (
        "fetch the url", "fetch the page", "web request",
        "fetch_url", "fetch-url", "http_get", "http-get",
    ),
    "WebSearch": (
        "web search", "search the web", "search_web", "search-web",
    ),
    "Task": ("subagent", "spawn an agent", "task tool"),
    "apply_patch": (
        "apply_patch", "apply-patch", "apply patch",
        "*** begin patch", "patch envelope",
    ),
}

_CALL_VERB_RE = re.compile(r"\bcall:(?:[A-Za-z0-9_.-]+:)*([A-Za-z0-9_.-]+)\s*\{")

_REFUSAL_PATTERNS = (
    re.compile(r"\bi (?:can(?:not|'t)|am unable|won't|will not)\b", re.IGNORECASE),
    re.compile(r"\bsorry,?\s+(?:but\s+)?i\b", re.IGNORECASE),
    re.compile(r"\bi don't have (?:access|the ability|tools)\b", re.IGNORECASE),
)

_MIN_REPLY_CHARS = 80


# ── v1 (legacy single-turn) loader + grader. ─────────────────────────


def load_agent_recorded_v1_cases(path: Path = FIXTURE_PATH) -> list[dict[str, Any]]:
    """Return the legacy single-turn cases shaped for the lucebench runner.

    Each fixture entry maps to the canonical runner case shape: the
    ``user_message`` carries the original session prompt verbatim
    (post-PII-strip), and the runner-internal fields are filled in here.
    The verifier / reference_trace / initial_state blobs ride along
    under their own keys so the grader can read them without re-loading
    the JSON.
    """
    if not path.exists():  # pragma: no cover - missing fixture = packaging bug
        return []
    payload = json.loads(path.read_text())
    out: list[dict[str, Any]] = []
    for raw in payload["cases"]:
        out.append(
            {
                "area": "agent_recorded_v1",
                "source": "agent-recorded-" + raw["source"],
                "id": raw["id"],
                "kind": "agent-prompt",
                "prompt": raw["prompt"],
                "user_message": raw["prompt"],
                "answer": None,
                "domain": "agent_recorded_v1",
                "title": raw["id"],
                "initial_state": raw.get("initial_state", {}),
                "reference_trace": raw.get("reference_trace", {}),
                "verifier": raw.get("verifier", {}),
            }
        )
    return out


# Back-compat alias for snapshot consumers / regrade pipelines that
# stamp ``load_agent_recorded_cases``. The v1 area still gets the same
# legacy cases.
load_agent_recorded_cases = load_agent_recorded_v1_cases


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower())


def _refused(text: str) -> bool:
    head = (text or "")[:300]
    return any(p.search(head) for p in _REFUSAL_PATTERNS)


def _tool_mentioned(text: str, tool: str) -> bool:
    if not text:
        return False
    if re.search(rf"\b{re.escape(tool)}\b", text):
        return True
    haystack = _normalize(text)
    synonyms = _TOOL_SYNONYMS.get(tool, ())
    for syn in synonyms:
        if syn in haystack:
            return True
    if synonyms:
        for m in _CALL_VERB_RE.finditer(text):
            verb = m.group(1).lower()
            if verb in synonyms:
                return True
    return False


def _file_mentioned(text: str, file_path: str) -> bool:
    if not text or not file_path:
        return False
    haystack = _normalize(text)
    base = file_path.split("/")[-1]
    if base and base.lower() in haystack:
        return True
    if file_path.lower() in haystack:
        return True
    return False


def grade_agent_recorded_v1_case(case: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    """Three-bin tool-schema-coverage grader (legacy v1 area).

    See module docstring for why this exists and why the new
    ``agent_recorded`` area uses an LLM judge instead.
    """
    verifier = case.get("verifier") or {}
    completion = (row.get("content") or "")
    reasoning = (row.get("reasoning_content") or "")
    text = (completion + "\n" + reasoning).strip()

    nonempty = len(text) >= _MIN_REPLY_CHARS
    refused = _refused(text)

    expected_tools: list[str] = list(verifier.get("expected_tools") or [])
    expected_files: list[str] = list(verifier.get("expected_files_touched") or [])

    tools_hit = [t for t in expected_tools if _tool_mentioned(text, t)]
    files_hit = [f for f in expected_files if _file_mentioned(text, f)]

    tool_coverage = len(tools_hit) / len(expected_tools) if expected_tools else 0.0
    file_coverage = len(files_hit) / len(expected_files) if expected_files else None

    if not nonempty or refused:
        bin_ = "fail"
    elif expected_files:
        if tools_hit and files_hit:
            bin_ = "pass"
        elif tools_hit or files_hit:
            bin_ = "partial"
        else:
            bin_ = "fail"
    else:
        if tools_hit:
            bin_ = "pass"
        else:
            bin_ = "fail"

    passed = bin_ == "pass"
    given = "engaged" if nonempty and not refused else ("refused" if refused else "stub")
    correct_str = ",".join(expected_tools[:4]) + (
        " | " + ",".join(expected_files[:2]) if expected_files else ""
    )

    return {
        "pass": passed,
        "given": given,
        "correct": correct_str,
        "status": "passed" if passed else ("partial" if bin_ == "partial" else "failed"),
        "format_pass": nonempty,
        "semantic_hint": bool(tools_hit) or bool(files_hit),
        "bin": bin_,
        "tools_hit": tools_hit,
        "files_hit": files_hit,
        "tool_coverage": round(tool_coverage, 3),
        "file_coverage": None if file_coverage is None else round(file_coverage, 3),
    }


# Back-compat alias.
grade_agent_recorded_case = grade_agent_recorded_v1_case


# ── Multi-turn replay (new default agent_recorded area). ─────────────


def load_agent_recorded_multi_turn_cases(
    path: Path = MULTI_TURN_FIXTURE_PATH,
) -> list[dict[str, Any]]:
    """Return multi-turn replay cases (v1 schema: with reference_response).

    Each case carries an OpenAI-shape ``messages`` list ending on a user
    turn (sendable verbatim to ``/v1/chat/completions``) plus a
    ``reference_response`` string holding the assistant text Claude
    actually emitted at that point — that's the ground truth the LLM
    judge compares the candidate's response to.

    Back-compat: v0 fixtures (no ``reference_response``) load with
    ``reference_response = None``; consumers that need it must handle
    None (the area runner surfaces ``judge_pending`` for those cases).

    The fixture is OPTIONAL: returns ``[]`` when absent. The autotune
    sweep also calls this loader and expects ``[]`` to mean "no data,
    skip the cell" — see ``lucebox.sweep.run_multi_turn_probe``.
    """
    if not path.exists():
        return []
    payload = json.loads(path.read_text())
    cases: list[dict[str, Any]] = []
    for raw in payload.get("cases", []):
        cases.append(
            {
                "id": raw["id"],
                "source": raw["source"],
                "kind": raw.get("kind", "multi-turn-replay"),
                "messages": raw["messages"],
                "reference_response": raw.get("reference_response"),
                "context_tokens_approx": raw["context_tokens_approx"],
                "target_bucket_tokens": raw["target_bucket_tokens"],
                "n_messages": raw.get("n_messages", len(raw["messages"])),
                "initial_state": raw.get("initial_state", {}),
                "verifier": raw.get("verifier", {}),
            }
        )
    cases.sort(key=lambda c: c["target_bucket_tokens"])
    return cases


def pick_multi_turn_case_for_budget(
    cases: list[dict[str, Any]],
    prompt_budget_tokens: int,
    *,
    safety_factor: float = 0.7,
) -> dict[str, Any] | None:
    """Pick the largest multi-turn case that fits within ``prompt_budget_tokens``.

    Used by the autotune sweep to choose a case for a given (max_ctx,
    reply_budget) cell. ``safety_factor`` defaults to 0.7 so the
    extractor's ``chars / 4`` approximation has headroom against the
    real tokenizer + chat template expansion (see
    ``docs/experiments/gemma4-26b-coding-agent-loop-sweep-2026-05-30.md``).
    """
    effective_budget = int(prompt_budget_tokens * safety_factor)
    fit = [c for c in cases if c["context_tokens_approx"] <= effective_budget]
    if not fit:
        return None
    return max(fit, key=lambda c: c["context_tokens_approx"])


def _content_to_text(content: Any) -> str:
    """Normalize OpenAI message content to text for judge references."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return json.dumps(content, ensure_ascii=False, sort_keys=True, default=str)


def _estimate_messages_tokens_approx(messages: list[dict[str, Any]]) -> int:
    """Cheap stable approximation for per-turn context growth metadata."""
    blob = json.dumps(messages, ensure_ascii=False, sort_keys=True, default=str)
    return max(1, len(blob) // 4)


def _build_sequential_replay_turns(case: dict[str, Any]) -> list[dict[str, Any]]:
    """Slice a recorded transcript into growing prompts ending on user turns.

    The fixture stores one complete prompt ending on the user turn to be
    judged. Earlier assistant turns inside ``messages`` are the recorded
    transcript, so replaying prefixes ending at each user turn mirrors a
    long-running agent session without depending on the candidate model
    to regenerate byte-identical intermediate assistant messages.
    """
    messages = list(case.get("messages") or [])
    if not messages:
        return []

    user_indices = [idx for idx, msg in enumerate(messages) if msg.get("role") == "user"]
    if not user_indices:
        return [
            {
                "turn_index": 1,
                "message_index": len(messages) - 1,
                "messages": messages,
                "reference_response": case.get("reference_response"),
                "is_final": True,
            }
        ]

    last_user_index = user_indices[-1]
    turns: list[dict[str, Any]] = []
    for turn_index, message_index in enumerate(user_indices, start=1):
        reference = case.get("reference_response") if message_index == last_user_index else None
        if reference is None and message_index + 1 < len(messages):
            next_msg = messages[message_index + 1]
            if next_msg.get("role") == "assistant":
                reference = _content_to_text(next_msg.get("content"))
        prefix = messages[: message_index + 1]
        turns.append(
            {
                "turn_index": turn_index,
                "message_index": message_index,
                "messages": prefix,
                "reference_response": reference,
                "is_final": message_index == last_user_index,
            }
        )
    return turns


def _prefill_speedup(cold_row: dict[str, Any], warm_row: dict[str, Any]) -> float | None:
    """Return cold/warm prefill speedup when both rows provide signal."""
    cold_pf = cold_row.get("prefill_s")
    warm_pf = warm_row.get("prefill_s")
    if cold_pf is not None and warm_pf is not None:
        return round(float(cold_pf) / max(float(warm_pf), _PREFILL_SPEEDUP_MIN_WARM_S), 2)
    return None


# ── Legacy prefill-and-decode grader. ────────────────────────────────


def grade_prefill_and_decode(
    row: dict[str, Any], *, min_response_chars: int = 1, max_wall_seconds: float = 300.0
) -> dict[str, Any]:
    """Pass/fail grader for the legacy prefill-and-decode verifier.

    Pre-judge consumers (the autotune sweep) still call this. The new
    multi-turn replay area uses :func:`grade_cache_and_quality` instead,
    which additionally invokes the LLM judge.
    """
    err = row.get("error")
    if err:
        return {"pass": False, "reason": f"server error: {err}"}
    wall = float(row.get("wall_s") or 0.0)
    if wall > max_wall_seconds:
        return {"pass": False, "reason": f"wall {wall:.1f}s > {max_wall_seconds}s budget"}
    content = (row.get("content") or "") + (row.get("reasoning_content") or "")
    if len(content) < min_response_chars:
        return {"pass": False, "reason": f"response too short ({len(content)} < {min_response_chars})"}
    return {"pass": True, "reason": f"prefill+decode ok, {len(content)} chars in {wall:.1f}s"}


# ── HTTP + service-restart helpers. ──────────────────────────────────


def _http_post_chat_completions(
    *,
    url: str,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    auth_header: str = "",
    timeout_s: int = 600,
) -> tuple[dict[str, Any], float]:
    """POST ``messages`` to ``url/v1/chat/completions``, return (parsed, wall).

    Raises on transport/HTTP failures so the caller can stamp a row
    error and continue. Streaming is intentionally OFF — we want the
    server-reported ``usage.timings`` block (prefill_ms / decode_ms /
    prefix_len) which lucebox surfaces on the non-stream path.
    """
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }
    headers = {"Content-Type": "application/json"}
    if auth_header:
        headers["Authorization"] = auth_header
    req = urllib.request.Request(
        url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers=headers,
    )
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        raw = json.loads(resp.read())
    wall = time.perf_counter() - t0
    return raw, wall


def _extract_row_from_response(raw: dict[str, Any], wall: float) -> dict[str, Any]:
    """Pull the bench-shape row fields out of a chat completions response."""
    choice = (raw.get("choices") or [{}])[0]
    msg = choice.get("message", {}) if isinstance(choice, dict) else {}
    usage = raw.get("usage", {}) or {}
    timings = usage.get("timings") if isinstance(usage, dict) else None
    if not isinstance(timings, dict):
        timings = {}
    prefill_ms = timings.get("prefill_ms")
    decode_ms = timings.get("decode_ms")
    # Explicit ``is not None`` checks — ``prompt_n_cached=0`` (cold cache)
    # is a valid value that ``or`` would treat as falsy, incorrectly
    # falling through to ``prefix_len`` and inflating cache-hit metrics.
    prefix_len = timings.get("prompt_n_cached")
    if prefix_len is None:
        prefix_len = timings.get("prefix_len")
    if prefix_len is None:
        prefix_len = 0
    tps_decode = timings.get("decode_tokens_per_sec")
    out_tokens = usage.get("completion_tokens")
    if tps_decode is None and decode_ms and out_tokens:
        # Compute a fallback if the server didn't ship decode_tokens_per_sec.
        if decode_ms > 0:
            tps_decode = (out_tokens / decode_ms) * 1000.0
    return {
        "content": msg.get("content"),
        "reasoning_content": msg.get("reasoning_content") or msg.get("reasoning"),
        "finish_reason": choice.get("finish_reason"),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": out_tokens,
        "wall_s": round(wall, 3),
        "prefill_s": round(prefill_ms / 1000.0, 3) if prefill_ms is not None else None,
        "decode_s": round(decode_ms / 1000.0, 3) if decode_ms is not None else None,
        "tps_decode": round(tps_decode, 2) if tps_decode is not None else None,
        "prefix_len": int(prefix_len) if prefix_len else 0,
        "restore": bool(prefix_len) and int(prefix_len) > 0,
        "timings_raw": timings,
        "out_tokens": out_tokens,
    }


def _health_wait(url: str, *, timeout_s: float = 120.0) -> bool:
    """Poll ``url/health`` until 2xx or ``timeout_s`` elapses. Returns success."""
    deadline = time.monotonic() + timeout_s
    health_url = url.rstrip("/") + "/health"
    last_err: str | None = None
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=5) as resp:
                if 200 <= resp.status < 300:
                    return True
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as e:
            last_err = f"{type(e).__name__}: {e}"
        time.sleep(1.0)
    print(
        f"[agent_recorded] health wait failed: last error: {last_err}",
        file=sys.stderr,
        flush=True,
    )
    return False


def _restart_lucebox_service(*, log_prefix: str = "[agent_recorded]") -> tuple[bool, str]:
    """Issue ``systemctl --user restart lucebox.service``.

    Returns ``(ok, reason)``. ``ok=False`` triggers the warm-only
    fallback. We intentionally do NOT modify the unit file or the
    binary — just restart the running service. If the user opted into
    a different supervisor (no systemd, podman, etc.) we skip and warn.
    """
    if shutil.which("systemctl") is None:
        return (False, "systemctl not on PATH")
    timeout_s = float(os.environ.get("LUCEBENCH_AGENT_RECORDED_RESTART_TIMEOUT", "180"))
    try:
        result = subprocess.run(  # noqa: S603 - operator-approved invocation
            ["systemctl", "--user", "restart", "lucebox.service"],
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    except subprocess.TimeoutExpired:
        return (False, f"systemctl restart timed out (>{timeout_s:.0f}s)")
    except OSError as e:
        return (False, f"systemctl invocation failed: {e}")
    if result.returncode != 0:
        return (
            False,
            f"systemctl exited {result.returncode}: {result.stderr.strip() or result.stdout.strip()}",
        )
    return (True, "ok")


# ── Per-case cold/warm driver. ───────────────────────────────────────


def _run_one_exact_repeat_case(
    *,
    case: dict[str, Any],
    url: str,
    model: str,
    max_tokens: int,
    auth_header: str,
    timeout_s: int,
    restart_between_cases: bool,
    judge_fn: Any,
    log_prefix: str = "[agent_recorded]",
) -> dict[str, Any]:
    """Run one case through cold-then-warm exact-repeat passes.

    ``judge_fn`` is the callable used to grade the cold response. The
    production path passes :func:`lucebench.grading.llm_judge.judge_response`;
    tests pass a mock so the runner can be exercised without billing
    Anthropic.
    """
    case_id = case["id"]
    bucket = case["target_bucket_tokens"]
    messages = case["messages"]
    reference = case.get("reference_response")

    cold_row: dict[str, Any] = {}
    warm_row: dict[str, Any] = {}
    error: str | None = None
    restart_status = "skipped"
    restart_reason = "restart_between_cases=False"

    if restart_between_cases:
        ok, reason = _restart_lucebox_service(log_prefix=log_prefix)
        restart_status = "ok" if ok else "failed"
        restart_reason = reason
        if ok:
            if not _health_wait(url):
                error = "service did not become healthy after restart"
        else:
            print(
                f"{log_prefix} cold-restart not available ({reason}); "
                f"falling back to warm/warm for case {case_id}",
                file=sys.stderr,
                flush=True,
            )

    # Cold pass.
    if error is None:
        try:
            raw, wall = _http_post_chat_completions(
                url=url,
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                auth_header=auth_header,
                timeout_s=timeout_s,
            )
            cold_row = _extract_row_from_response(raw, wall)
        except urllib.error.HTTPError as e:
            try:
                body = e.read().decode("utf-8", errors="replace")[:300]
            except Exception:  # noqa: BLE001
                body = ""
            error = f"cold HTTP {e.code}: {body}"
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            error = f"cold transport: {type(e).__name__}: {e}"

    # Warm pass — sent immediately after cold with no restart.
    if error is None:
        try:
            raw, wall = _http_post_chat_completions(
                url=url,
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                auth_header=auth_header,
                timeout_s=timeout_s,
            )
            warm_row = _extract_row_from_response(raw, wall)
        except urllib.error.HTTPError as e:
            try:
                body = e.read().decode("utf-8", errors="replace")[:300]
            except Exception:  # noqa: BLE001
                body = ""
            error = f"warm HTTP {e.code}: {body}"
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            error = f"warm transport: {type(e).__name__}: {e}"

    cache_speedup_x = _prefill_speedup(cold_row, warm_row)

    # Judge the COLD response (not warm — warm is solely about cache
    # measurement; the model output should be identical bit-for-bit
    # except for sampler RNG, but we judge the first sample for
    # deterministic accounting).
    judge_result: dict[str, Any]
    cold_content = (cold_row.get("content") or "") + (cold_row.get("reasoning_content") or "")
    if error is not None:
        judge_result = {
            "pass": False,
            "verdict": "off_track",
            "rationale": f"transport error precluded judging: {error}",
            "cached": False,
            "model": None,
            "error": error,
        }
    elif not reference:
        judge_result = {
            "pass": False,
            "verdict": "judge_pending",
            "rationale": "case has no reference_response (v0 fixture); judge skipped",
            "cached": False,
            "model": None,
            "error": "no_reference",
        }
    else:
        # ``JudgeUnavailable`` (missing API key / SDK) is a configuration
        # failure rather than a per-call hiccup — let it propagate so the
        # area aborts on the first case instead of silently masking every
        # row as ``judge_pending``. See module docstring.
        from lucebench.grading.llm_judge import JudgeUnavailable

        try:
            verdict = judge_fn(
                case_id=case_id,
                messages=messages,
                reference_response=reference,
                candidate_response=cold_content,
            )
            judge_result = verdict.to_dict() if hasattr(verdict, "to_dict") else dict(verdict)
        except JudgeUnavailable:
            raise
        except Exception as e:  # noqa: BLE001 - one judge failure → pending, not abort
            judge_result = {
                "pass": False,
                "verdict": "judge_pending",
                "rationale": f"judge invocation raised: {type(e).__name__}: {e}",
                "cached": False,
                "model": None,
                "error": f"{type(e).__name__}: {e}",
            }

    return {
        "replay_mode": "exact-repeat",
        "case_id": case_id,
        "source": case.get("source"),
        "bucket_tokens": bucket,
        "n_messages": case.get("n_messages"),
        "context_tokens_approx": case.get("context_tokens_approx"),
        "restart": {"status": restart_status, "reason": restart_reason},
        "cold": cold_row,
        "warm": warm_row,
        "turns": [],
        "cache_eligible_turns": 1 if warm_row else 0,
        "cache_hit_turns": 1 if warm_row.get("restore") else 0,
        "cache_speedup_x": cache_speedup_x,
        "model_response_cold": cold_content,
        "model_response_final": cold_content,
        "judge": judge_result,
        "pass": bool(judge_result.get("pass")),
        "error": error,
    }


def _run_one_sequential_case(
    *,
    case: dict[str, Any],
    url: str,
    model: str,
    max_tokens: int,
    auth_header: str,
    timeout_s: int,
    restart_between_cases: bool,
    judge_fn: Any,
    log_prefix: str = "[agent_recorded]",
) -> dict[str, Any]:
    """Replay one recorded session as growing prompts, then judge final turn."""
    case_id = case["id"]
    bucket = case["target_bucket_tokens"]
    turns = _build_sequential_replay_turns(case)

    turn_rows: list[dict[str, Any]] = []
    error: str | None = None
    restart_status = "skipped"
    restart_reason = "restart_between_cases=False"

    if restart_between_cases:
        ok, reason = _restart_lucebox_service(log_prefix=log_prefix)
        restart_status = "ok" if ok else "failed"
        restart_reason = reason
        if ok:
            if not _health_wait(url):
                error = "service did not become healthy after restart"
        else:
            print(
                f"{log_prefix} cold-restart not available ({reason}); "
                f"running sequential replay against existing cache state for case {case_id}",
                file=sys.stderr,
                flush=True,
            )

    if error is None and not turns:
        error = "case has no replayable messages"

    if error is None:
        for turn in turns:
            turn_index = int(turn["turn_index"])
            turn_messages = turn["messages"]
            try:
                raw, wall = _http_post_chat_completions(
                    url=url,
                    model=model,
                    messages=turn_messages,
                    max_tokens=max_tokens,
                    auth_header=auth_header,
                    timeout_s=timeout_s,
                )
                turn_row = _extract_row_from_response(raw, wall)
                turn_row.update(
                    {
                        "turn_index": turn_index,
                        "message_index": turn["message_index"],
                        "n_messages": len(turn_messages),
                        "context_tokens_approx": _estimate_messages_tokens_approx(turn_messages),
                        "cache_eligible": turn_index > 1,
                        "is_final": bool(turn.get("is_final")),
                    }
                )
                turn_rows.append(turn_row)
            except urllib.error.HTTPError as e:
                try:
                    body = e.read().decode("utf-8", errors="replace")[:300]
                except Exception:  # noqa: BLE001
                    body = ""
                error = f"turn {turn_index} HTTP {e.code}: {body}"
                break
            except (urllib.error.URLError, OSError, TimeoutError) as e:
                error = f"turn {turn_index} transport: {type(e).__name__}: {e}"
                break

    cold_row = turn_rows[0] if turn_rows else {}
    final_row = next((row for row in reversed(turn_rows) if row.get("is_final")), None)
    if final_row is None and turn_rows:
        final_row = turn_rows[-1]
    warm_row = final_row if final_row is not None and len(turn_rows) > 1 else {}

    eligible_turns = [row for row in turn_rows if row.get("cache_eligible")]
    cache_hit_turns = sum(1 for row in eligible_turns if row.get("restore"))

    reference = case.get("reference_response")
    final_messages = turns[-1]["messages"] if turns else case.get("messages", [])
    final_content = ""
    if final_row is not None:
        final_content = (final_row.get("content") or "") + (final_row.get("reasoning_content") or "")

    judge_result: dict[str, Any]
    if error is not None:
        judge_result = {
            "pass": False,
            "verdict": "off_track",
            "rationale": f"transport error precluded judging: {error}",
            "cached": False,
            "model": None,
            "error": error,
        }
    elif not reference:
        judge_result = {
            "pass": False,
            "verdict": "judge_pending",
            "rationale": "case has no reference_response (v0 fixture); judge skipped",
            "cached": False,
            "model": None,
            "error": "no_reference",
        }
    else:
        from lucebench.grading.llm_judge import JudgeUnavailable

        try:
            verdict = judge_fn(
                case_id=case_id,
                messages=final_messages,
                reference_response=reference,
                candidate_response=final_content,
            )
            judge_result = verdict.to_dict() if hasattr(verdict, "to_dict") else dict(verdict)
        except JudgeUnavailable:
            raise
        except Exception as e:  # noqa: BLE001
            judge_result = {
                "pass": False,
                "verdict": "judge_pending",
                "rationale": f"judge invocation raised: {type(e).__name__}: {e}",
                "cached": False,
                "model": None,
                "error": f"{type(e).__name__}: {e}",
            }

    return {
        "replay_mode": "sequential",
        "case_id": case_id,
        "source": case.get("source"),
        "bucket_tokens": bucket,
        "n_messages": case.get("n_messages"),
        "context_tokens_approx": case.get("context_tokens_approx"),
        "restart": {"status": restart_status, "reason": restart_reason},
        "cold": cold_row,
        "warm": warm_row,
        "turns": turn_rows,
        "cache_eligible_turns": len(eligible_turns),
        "cache_hit_turns": cache_hit_turns,
        "cache_speedup_x": None,
        "model_response_cold": final_content,
        "model_response_final": final_content,
        "judge": judge_result,
        "pass": bool(judge_result.get("pass")),
        "error": error,
    }


def _run_one_multi_turn_case(
    *,
    case: dict[str, Any],
    url: str,
    model: str,
    max_tokens: int,
    auth_header: str,
    timeout_s: int,
    restart_between_cases: bool,
    judge_fn: Any,
    replay_mode: str,
    log_prefix: str = "[agent_recorded]",
) -> dict[str, Any]:
    if replay_mode == "exact-repeat":
        return _run_one_exact_repeat_case(
            case=case,
            url=url,
            model=model,
            max_tokens=max_tokens,
            auth_header=auth_header,
            timeout_s=timeout_s,
            restart_between_cases=restart_between_cases,
            judge_fn=judge_fn,
            log_prefix=log_prefix,
        )
    return _run_one_sequential_case(
        case=case,
        url=url,
        model=model,
        max_tokens=max_tokens,
        auth_header=auth_header,
        timeout_s=timeout_s,
        restart_between_cases=restart_between_cases,
        judge_fn=judge_fn,
        log_prefix=log_prefix,
    )


def _resolve_judge_fn(*, mock_judge: Any = None) -> Any:
    """Return the judge callable to use for the run.

    Production: :func:`lucebench.grading.llm_judge.judge_response`.
    Tests / dry-runs: pass a mock with the same signature.
    """
    if mock_judge is not None:
        return mock_judge
    from lucebench.grading.llm_judge import judge_response

    return judge_response


def run_agent_recorded_area(
    *,
    url: str,
    model: str,
    max_tokens: int = 512,
    auth_header: str = "",
    timeout_s: int = 600,
    restart_between_cases: bool = True,
    fixture_path: Path = MULTI_TURN_FIXTURE_PATH,
    questions: int | None = None,
    mock_judge: Any = None,
    replay_mode: str = "sequential",
    log_prefix: str = "[agent_recorded]",
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Run the full multi-turn replay sweep and return (rows, summary).

    In default ``sequential`` mode, each row carries per-turn sub-rows
    under ``turns`` plus first/final aliases under ``cold`` / ``warm``.
    In ``exact-repeat`` mode, each row keeps the historical cold/warm
    duplicate-prompt shape. The summary aggregates pass rate and cache
    metrics appropriate to the selected replay mode.
    """
    if replay_mode not in AGENT_RECORDED_REPLAY_MODES:
        raise ValueError(
            f"replay_mode must be one of {AGENT_RECORDED_REPLAY_MODES}, got {replay_mode!r}"
        )
    cases = load_agent_recorded_multi_turn_cases(fixture_path)
    if questions is not None:
        cases = cases[:questions]
    if not cases:
        return ([], {"area": "agent_recorded", "n": 0, "error": "no cases in fixture"})

    judge_fn = _resolve_judge_fn(mock_judge=mock_judge)
    rows: list[dict[str, Any]] = []
    for idx, case in enumerate(cases, start=1):
        print(
            f"{log_prefix} case {idx}/{len(cases)} id={case['id']} "
            f"bucket={case['target_bucket_tokens']} ctx≈{case['context_tokens_approx']}",
            flush=True,
        )
        row = _run_one_multi_turn_case(
            case=case,
            url=url,
            model=model,
            max_tokens=max_tokens,
            auth_header=auth_header,
            timeout_s=timeout_s,
            restart_between_cases=restart_between_cases,
            judge_fn=judge_fn,
            replay_mode=replay_mode,
            log_prefix=log_prefix,
        )
        rows.append(row)
        # Brief per-row console line so the operator can watch progress.
        judge = row["judge"]
        if row.get("replay_mode") == "sequential":
            final = row.get("warm") or row.get("cold") or {}
            eligible = row.get("cache_eligible_turns", 0) or 0
            hit_rate = (100.0 * row.get("cache_hit_turns", 0) / eligible) if eligible else 0.0
            print(
                f"{log_prefix}   turns={len(row.get('turns') or [])} "
                f"cache_hits={row.get('cache_hit_turns', 0)}/{eligible} ({hit_rate:.1f}%) "
                f"final_prefill={final.get('prefill_s')}s final_prefix={final.get('prefix_len')} "
                f"judge={judge.get('verdict')} pass={row['pass']}",
                flush=True,
            )
        else:
            cold = row["cold"]
            warm = row["warm"]
            print(
                f"{log_prefix}   cold prefill={cold.get('prefill_s')}s decode={cold.get('decode_s')}s "
                f"tps={cold.get('tps_decode')}  warm prefill={warm.get('prefill_s')}s "
                f"speedup={row['cache_speedup_x']}x  judge={judge.get('verdict')} "
                f"pass={row['pass']}",
                flush=True,
            )

    return rows, _summarize_rows(rows)


def _percentile(values: list[float], pct: float) -> float | None:
    """Naive linear-interp percentile, or None for an empty input.

    NumPy isn't a dep so we hand-roll. The bench rows are at most a few
    hundred per area; sort + linear interp is plenty fast.
    """
    if not values:
        return None
    if len(values) == 1:
        return values[0]
    s = sorted(values)
    pos = (len(s) - 1) * (pct / 100.0)
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return s[lo] + (s[hi] - s[lo]) * frac


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-row metrics into the per-area summary.

    Pass rate excludes ``judge_pending`` from the denominator — those
    cases couldn't be graded (no reference response, or judge
    errored) and shouldn't drag the rate down or inflate it.
    """
    n_total = len(rows)
    judged = [r for r in rows if r["judge"].get("verdict") != "judge_pending"]
    n_pass = sum(1 for r in judged if r["pass"])
    pass_rate = (100.0 * n_pass / len(judged)) if judged else 0.0

    speedups = [r["cache_speedup_x"] for r in rows if r["cache_speedup_x"] is not None]
    cold_prefills = [r["cold"].get("prefill_s") for r in rows if r["cold"].get("prefill_s") is not None]
    warm_prefills = [r["warm"].get("prefill_s") for r in rows if r["warm"].get("prefill_s") is not None]
    cold_decodes = [r["cold"].get("decode_s") for r in rows if r["cold"].get("decode_s") is not None]
    warm_decodes = [r["warm"].get("decode_s") for r in rows if r["warm"].get("decode_s") is not None]
    all_turns = [turn for r in rows for turn in (r.get("turns") or [])]
    eligible_turns = [turn for turn in all_turns if turn.get("cache_eligible")]
    hit_turns = [turn for turn in eligible_turns if turn.get("restore")]
    miss_turns = [turn for turn in eligible_turns if not turn.get("restore")]
    exact_repeat_rows = [r for r in rows if r.get("replay_mode") == "exact-repeat"]
    exact_eligible_rows = [
        r.get("warm") or {}
        for r in exact_repeat_rows
        if int(r.get("cache_eligible_turns") or 0) > 0
    ]
    exact_hit_rows = [
        r.get("warm") or {}
        for r in exact_repeat_rows
        if int(r.get("cache_hit_turns") or 0) > 0
    ]
    exact_miss_rows = [
        r.get("warm") or {}
        for r in exact_repeat_rows
        if int(r.get("cache_eligible_turns") or 0) > 0
        and int(r.get("cache_hit_turns") or 0) == 0
    ]
    final_turns = [
        (r.get("warm") or r.get("cold") or {})
        for r in rows
        if r.get("replay_mode") == "sequential"
    ]
    modes = sorted({r.get("replay_mode") or "exact-repeat" for r in rows})
    cache_eligible_count = len(eligible_turns) + len(exact_eligible_rows)
    cache_hit_count = len(hit_turns) + len(exact_hit_rows)
    cache_hit_rate = (
        100.0 * cache_hit_count / cache_eligible_count if cache_eligible_count else 0.0
    )

    # Per-verdict counts so the operator can see suitable_match vs
    # divergent_but_valid vs off_track without re-parsing rows.
    verdict_counts: dict[str, int] = {}
    for r in rows:
        v = r["judge"].get("verdict") or "unknown"
        verdict_counts[v] = verdict_counts.get(v, 0) + 1

    return {
        "area": "agent_recorded",
        "replay_mode": modes[0] if len(modes) == 1 else ",".join(modes),
        "n": n_total,
        "n_judged": len(judged),
        "n_pass": n_pass,
        "pass_rate": round(pass_rate, 2),
        "cache_speedup_p50": _percentile(speedups, 50.0),
        "cache_speedup_p90": _percentile(speedups, 90.0),
        "cold_prefill_p50": _percentile(cold_prefills, 50.0),
        "warm_prefill_p50": _percentile(warm_prefills, 50.0),
        "cold_decode_p50": _percentile(cold_decodes, 50.0),
        "warm_decode_p50": _percentile(warm_decodes, 50.0),
        "turns_total": len(all_turns),
        "cache_eligible_turns": cache_eligible_count,
        "cache_hit_turns": cache_hit_count,
        "cache_hit_rate": round(cache_hit_rate, 2),
        "turn_prefill_p50": _percentile(
            [t.get("prefill_s") for t in all_turns if t.get("prefill_s") is not None], 50.0,
        ),
        "cache_eligible_prefill_p50": _percentile(
            [
                t.get("prefill_s")
                for t in [*eligible_turns, *exact_eligible_rows]
                if t.get("prefill_s") is not None
            ],
            50.0,
        ),
        "cache_hit_prefill_p50": _percentile(
            [
                t.get("prefill_s")
                for t in [*hit_turns, *exact_hit_rows]
                if t.get("prefill_s") is not None
            ],
            50.0,
        ),
        "cache_miss_prefill_p50": _percentile(
            [
                t.get("prefill_s")
                for t in [*miss_turns, *exact_miss_rows]
                if t.get("prefill_s") is not None
            ],
            50.0,
        ),
        "prefix_len_p50": _percentile(
            [
                t.get("prefix_len")
                for t in [*eligible_turns, *exact_eligible_rows]
                if t.get("prefix_len") is not None
            ],
            50.0,
        ),
        "final_prefill_p50": _percentile(
            [t.get("prefill_s") for t in final_turns if t.get("prefill_s") is not None], 50.0,
        ),
        "turn_wall_p50": _percentile(
            [t.get("wall_s") for t in all_turns if t.get("wall_s") is not None], 50.0,
        ),
        "cache_eligible_wall_p50": _percentile(
            [t.get("wall_s") for t in eligible_turns if t.get("wall_s") is not None], 50.0,
        ),
        "verdict_counts": verdict_counts,
        "median_wall_cold": _percentile(
            [r["cold"].get("wall_s") for r in rows if r["cold"].get("wall_s") is not None], 50.0,
        ),
        "median_wall_warm": _percentile(
            [r["warm"].get("wall_s") for r in rows if r["warm"].get("wall_s") is not None], 50.0,
        ),
    }


# ── Stub grader for the standard runner dispatch. ────────────────────
#
# The CLI's _run_standard_area_to_dir path expects an ``AREAS[area]``
# entry with a ``grade`` callable that takes (case, row). The new area
# doesn't fit that 1:1 path (it needs cold+warm + restart + judge), so
# the CLI dispatches it through a separate ``_run_agent_recorded_to_dir``
# wrapper. The stub below is kept ONLY so legacy CLI code paths /
# regrade hooks that import ``grade_agent_recorded_case`` still load
# this module without an AttributeError. It returns a noop "pending"
# grade — the real grading happens in run_agent_recorded_area.


def grade_multi_turn_replay_stub(case: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    """Stub grader. The real per-row grading lives in
    :func:`run_agent_recorded_area` which builds and grades each row
    end-to-end. This stub exists so import-time hooks don't crash."""
    return {
        "pass": bool(row.get("pass")),
        "given": row.get("judge", {}).get("verdict") or "?",
        "correct": "judge-graded",
        "status": "passed" if row.get("pass") else "failed",
    }
