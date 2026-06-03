r"""Recorded-session agent probes for ``--areas agent_recorded``.

Replaces the format-detection grader of the synthetic ``agent`` area
with cases mined from *real* Claude Code and Codex sessions the user
actually drove (see ``scripts/extract-agentic-fixture.py``). The
fixture lives at ``fixtures/agent_recorded/cases.json`` and is the
output of running that collector with ``--scan`` against the local
``~/.claude/projects`` and ``~/.codex/sessions`` trees, with PII
strip + tool-result hashing applied at collection time.

Case shape (see also the collector's docstring):

.. code-block:: json

    {
      "id": "claude-2026-05-28-...-c22cc4fdad",
      "source": "claude-code" | "codex",
      "prompt": "<the first user message of the session>",
      "initial_state": {
        "cwd": "<HOME>/Projects/...",
        "git_ref": "abc1234" | null,
        "git_branch": "feat/...",
        "files_referenced": ["path/to/file", ...]
      },
      "reference_trace": {
        "tool_calls": [{"tool": "Edit", "args": {"file_path": "...",
                       "old_string_hash": "...", "new_string_hash": "..."}},
                      ...],
        "outcome": {
          "files_modified": [...],
          "commands_run_count": 7,
          "total_tool_calls": 21
        }
      },
      "verifier": {
        "type": "tool-schema-coverage",
        "expected_tools": ["Edit", "Bash", ...],
        "min_tool_calls": 2,
        "expected_files_touched": ["lucebox.sh", ...]
      }
    }

v0 verifier — ``tool-schema-coverage``
======================================

This module ships exactly one verifier today: ``tool-schema-coverage``.
The luce-bench runner sends the candidate model just the ``prompt``
(no tool definitions, no system prompt, single-turn) and we grade by
*pattern-matching the model's text reply* against the case's
``expected_tools`` + ``expected_files_touched``. Three bins:

* ``pass`` — model named at least one tool from ``expected_tools``
  AND named at least one file from ``expected_files_touched`` (or, if
  no files are expected, named >= 1 expected tool). Response is also
  required to be coherent (>= 80 chars, not a refusal).
* ``partial`` — some expected tool OR some expected file was named, but
  not both. Response is coherent.
* ``fail`` — model refused, produced a stub, or wandered off-topic
  (no expected tool AND no expected file mentioned).

Why pattern-match instead of replay? The user's stated mitigation
ordering: *grade on outcome, not trace; verifiable subgoals over
end-to-end; tool-schema validation when no verifier exists*. v0
implements the tool-schema-validation step — it's broadly applicable
(works for every recorded session without per-case grader code), cheap
(no shell, no git replay), and catches the obvious failure mode of
"model produced narrative prose instead of engaging as an agent". The
``verifier.type`` field is a versioned discriminator so future verifier
types (``outcome-equivalence``, ``subgoal``) can land additively
without forking the area.

Threshold rationale
-------------------

* ``len(text) >= 80`` for the coherence bar. A real agent reply to one
  of these prompts (most are 200-2000 char engineering tasks) that
  fits in <80 chars is either a refusal ("I cannot help with this.")
  or a stub. 80 is two short sentences — generous enough to not eat
  legitimate one-line answers.
* Tool-name matching is case-sensitive but tolerates the common
  near-synonyms ("Bash" matches "bash command", "shell", "run a
  command"). See ``_TOOL_SYNONYMS``.
* File matching uses basename to dodge the path-rewriting variance
  ("docs/foo.md" vs "./docs/foo.md" vs "the foo.md file"). When the
  expected file list is empty the file-name check is skipped, not
  failed.

Future work this module is shaped to absorb without rewrites
-----------------------------------------------------------

* ``verifier.type == "outcome-equivalence"`` would need a sandbox
  that replays the case's ``initial_state`` (git checkout to
  ``git_ref``) and compares the candidate's ``files_modified`` to the
  reference. The grader can branch on ``case["verifier"]["type"]``.
* ``verifier.type == "subgoal"`` adds a ``subgoals`` list of
  ``{description, check}`` pairs and grades the *intermediate* state
  of a multi-turn run. Same branch point.
* Multi-reference: ``reference_trace`` already nests under one key, so
  a future ``alternative_traces: [...]`` can sit alongside without
  schema churn.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

# See lucebench.areas.ds4_eval.GRADER_VERSION for the bump policy.
GRADER_VERSION = 1

SCRIPT_DIR = Path(__file__).resolve().parent.parent
FIXTURE_PATH = SCRIPT_DIR / "fixtures" / "agent_recorded" / "cases.json"
MULTI_TURN_FIXTURE_PATH = SCRIPT_DIR / "fixtures" / "agent_recorded" / "multi_turn_cases.json"

# Tool name → set of phrases that count as "the model meant this tool".
# Intentionally loose so a model that says "use a Bash command" passes
# even though it didn't literally say "Bash". Keys must match the
# canonical names emitted by the collector (Claude tool names + the
# normalized Codex shapes — see ``scripts/extract-agentic-fixture.py``).
#
# Also covers hyphen/underscore-named verbs that models commonly emit
# when they invent their own tool format (``call:execute-bead:read-file``,
# ``call:read_file``). Models running through this benchmark may not be
# Claude/Codex — they only know what the prompt taught them, which is
# often a DDX/bead-style verb namespace. We map those verbs back to the
# Claude tool the fixture expected.
_TOOL_SYNONYMS: dict[str, tuple[str, ...]] = {
    "Bash": (
        "bash", "shell command", "shell", "run a command",
        "execute a command", "command line",
        # Verb-style emissions. ``execute-bead`` alone is intentionally
        # NOT here: bead execution is a tool-namespace that wraps many
        # verbs (read_file, list_files, etc.) — the verb under the
        # namespace decides what Claude tool it maps to, not the
        # namespace itself.
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

# Compiled once: extract verbs from any ``call:<...:>:<verb>{...}`` or
# ``call:<verb>{...}`` pattern the model emits. The fallback synonym
# match handles plain-English mentions; this captures the structured
# tool-call-shaped emissions models invent when given a custom tool
# namespace in the prompt (DDX bead verbs, codex-mini-style commands,
# etc.). The verb is whatever follows the LAST colon before the brace —
# we strip the namespace prefix so ``call:execute-bead:read-file{}``
# yields the verb ``read-file`` (which the Read synonym list matches).
_CALL_VERB_RE = re.compile(r"\bcall:(?:[A-Za-z0-9_.-]+:)*([A-Za-z0-9_.-]+)\s*\{")

# Phrases that signal the model refused / punted. Any of these in the
# first ~200 chars of the reply forces a fail regardless of tool
# coverage — a "I can't do that" with a Bash-shaped citation in the
# follow-up still loses.
_REFUSAL_PATTERNS = (
    re.compile(r"\bi (?:can(?:not|'t)|am unable|won't|will not)\b", re.IGNORECASE),
    re.compile(r"\bsorry,?\s+(?:but\s+)?i\b", re.IGNORECASE),
    re.compile(r"\bi don't have (?:access|the ability|tools)\b", re.IGNORECASE),
)

# Minimum reply length to count as "the model engaged at all". Below
# this the result is fail-due-to-stub regardless of content.
_MIN_REPLY_CHARS = 80


def load_agent_recorded_cases(path: Path = FIXTURE_PATH) -> list[dict[str, Any]]:
    """Return the fixture cases shaped for the lucebench runner.

    Each fixture entry maps to the canonical runner case shape: the
    ``user_message`` carries the original session prompt verbatim
    (post-PII-strip), and the runner-internal fields
    (``area``, ``source``, ``id``, ``kind``, ``answer``, ``domain``,
    ``title``) are filled in here. The verifier / reference_trace /
    initial_state blobs ride along under their own keys so the grader
    can read them without re-loading the JSON.
    """
    if not path.exists():  # pragma: no cover - missing fixture = packaging bug
        return []
    payload = json.loads(path.read_text())
    out: list[dict[str, Any]] = []
    for raw in payload["cases"]:
        out.append(
            {
                "area": "agent_recorded",
                "source": "agent-recorded-" + raw["source"],
                "id": raw["id"],
                # Use the existing "agent-prompt" kind so the
                # lucebench runner's build_prompt() routes us to
                # ``case["user_message"]`` directly. That matches the
                # synthetic ``agent`` area's dispatch path so we
                # don't need a new runner branch.
                "kind": "agent-prompt",
                # Ship both fields so either dispatch path works.
                "prompt": raw["prompt"],
                "user_message": raw["prompt"],
                "answer": None,
                "domain": "agent_recorded",
                "title": raw["id"],
                # Side-band: grader reads these via `case["verifier"]` etc.
                "initial_state": raw.get("initial_state", {}),
                "reference_trace": raw.get("reference_trace", {}),
                "verifier": raw.get("verifier", {}),
            }
        )
    return out


def load_agent_recorded_multi_turn_cases(
    path: Path = MULTI_TURN_FIXTURE_PATH,
) -> list[dict[str, Any]]:
    """Return multi-turn replay cases for the coding-agent-loop autotune sweep.

    Distinct from :func:`load_agent_recorded_cases` (the single-prompt
    tool-schema-coverage fixture). Multi-turn cases ship an OpenAI-shape
    ``messages`` list — sendable verbatim to ``/v1/chat/completions`` —
    plus a ``target_bucket_tokens`` field that lets a caller pick the
    longest case fitting under a given ``max_ctx − reply_budget`` cap.
    See ``scripts/extract-agentic-fixture.py --multi-turn``.

    The fixture is OPTIONAL: returns ``[]`` when absent. Callers that
    require it should check the result and surface their own error.
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
                "context_tokens_approx": raw["context_tokens_approx"],
                "target_bucket_tokens": raw["target_bucket_tokens"],
                "n_messages": raw.get("n_messages", len(raw["messages"])),
                "initial_state": raw.get("initial_state", {}),
                "verifier": raw.get("verifier", {}),
            }
        )
    # Sorted ascending by bucket so callers can iterate or bisect to
    # find the largest case fitting a budget.
    cases.sort(key=lambda c: c["target_bucket_tokens"])
    return cases


def pick_multi_turn_case_for_budget(
    cases: list[dict[str, Any]],
    prompt_budget_tokens: int,
    *,
    safety_factor: float = 0.7,
) -> dict[str, Any] | None:
    """Pick the largest multi-turn case that fits within ``prompt_budget_tokens``.

    The sweep uses this to choose the right trace per cell: for a cell
    with ``max_ctx = N`` and a reply budget of ``r``, prompt_budget is
    ``N − r``. Returns ``None`` when no case fits (every case
    over-budget — caller should skip the cell or shrink the case).

    ``safety_factor`` (default 0.7) accounts for the gap between the
    extractor's ``chars / 4`` token approximation and the real
    tokenizer + chat template expansion. Empirical evidence from the
    gemma4-26b sweep on 2026-05-30 (see
    ``docs/experiments/gemma4-26b-coding-agent-loop-sweep-2026-05-30.md``):
    a 65205-approx-token Claude session tokenized to **90799 real
    tokens** through the gemma chat template — a 1.39× expansion. The
    102397-approx-token case overshoots a 126976-token budget at
    max_ctx=131072 and triggers HTTP 400 server-side. Without a safety
    margin the sweep's 131K cells fail uniformly. ``0.7`` corresponds
    to a 1.43× expansion guard; tune downward if a future fixture is
    even denser per char (e.g. heavy multibyte content).
    """
    effective_budget = int(prompt_budget_tokens * safety_factor)
    fit = [c for c in cases if c["context_tokens_approx"] <= effective_budget]
    if not fit:
        return None
    return max(fit, key=lambda c: c["context_tokens_approx"])


def grade_prefill_and_decode(
    row: dict[str, Any], *, min_response_chars: int = 1, max_wall_seconds: float = 300.0
) -> dict[str, Any]:
    """Pass/fail grader for the multi-turn prefill-and-decode verifier.

    Pass criterion: server returned content within ``max_wall_seconds``,
    the assistant emitted at least ``min_response_chars`` (combining
    visible content and any reasoning_content), and no HTTP/server
    error was reported. The verifier exists to score "this max_ctx
    setting actually serves a trace of this length", not the quality
    of the model's reply — anything coherent enough to render passes.
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


def _normalize(text: str) -> str:
    """Lowercased text with collapsed whitespace, used for substring
    checks against tool synonyms and file basenames."""
    return re.sub(r"\s+", " ", (text or "").lower())


def _refused(text: str) -> bool:
    head = (text or "")[:300]
    return any(p.search(head) for p in _REFUSAL_PATTERNS)


def _tool_mentioned(text: str, tool: str) -> bool:
    """True if the model named ``tool`` either by canonical name, by
    one of the loose synonyms in ``_TOOL_SYNONYMS``, or as the verb of a
    ``call:<...>:<verb>{...}`` structured-tool-call emission.

    The canonical-name check is case-sensitive so plain English
    sentences (\"would edit the file\") don't accidentally match the
    capitalized ``Edit`` token; the synonym list covers the
    lowercase / paraphrased cases; the call-verb pass covers models
    that emit their own structured format when given a custom tool
    namespace in the prompt.
    """
    if not text:
        return False
    if re.search(rf"\b{re.escape(tool)}\b", text):
        return True
    haystack = _normalize(text)
    synonyms = _TOOL_SYNONYMS.get(tool, ())
    for syn in synonyms:
        if syn in haystack:
            return True
    # Structured tool-call emissions: pull out every ``call:<verb>{...}``
    # invocation and treat its verb as a synonym candidate. This catches
    # models that invented their own tool format following a prompt's
    # custom namespace (DDX bead verbs etc.) without forcing the grader
    # to enumerate every prompt-driven naming convention.
    if synonyms:
        for m in _CALL_VERB_RE.finditer(text):
            verb = m.group(1).lower()
            if verb in synonyms:
                return True
    return False


def _file_mentioned(text: str, file_path: str) -> bool:
    """True if the basename or full path appears in ``text``."""
    if not text or not file_path:
        return False
    haystack = _normalize(text)
    base = file_path.split("/")[-1]
    if base and base.lower() in haystack:
        return True
    if file_path.lower() in haystack:
        return True
    return False


def grade_agent_recorded_case(case: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    """Three-bin tool-schema-coverage grader.

    Reads the verifier off the case (which makes it trivial to add
    new ``verifier.type`` branches later); the v0 branch is
    ``tool-schema-coverage``. The output shape matches the lucebench
    runner's grader contract (``pass``, ``given``, ``correct``,
    ``status``, ``format_pass``, ``semantic_hint``); the extra
    ``coverage`` / ``bin`` fields are surfaced so the row inspector
    can show which subgoal passed without re-running the grader.
    """
    verifier = case.get("verifier") or {}
    completion = (row.get("content") or "")
    # Some servers route the answer to reasoning_content when
    # max_tokens trips mid-think (same fallback as the smoke grader).
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
        # Both axes available — full pass requires at least one of each.
        if tools_hit and files_hit:
            bin_ = "pass"
        elif tools_hit or files_hit:
            bin_ = "partial"
        else:
            bin_ = "fail"
    else:
        # File-list empty (codex sessions where we couldn't recover
        # paths from the patch envelope). Grade purely on tools.
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
