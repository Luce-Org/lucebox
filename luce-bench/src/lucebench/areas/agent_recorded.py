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

# Tool name → set of phrases that count as "the model meant this tool".
# Intentionally loose so a model that says "use a Bash command" passes
# even though it didn't literally say "Bash". Keys must match the
# canonical names emitted by the collector (Claude tool names + the
# normalized Codex shapes — see ``scripts/extract-agentic-fixture.py``).
_TOOL_SYNONYMS: dict[str, tuple[str, ...]] = {
    "Bash": ("bash", "shell command", "shell", "run a command", "execute a command", "command line"),
    "Read": ("read the file", "read file", "open the file", "view the file", "cat ", "look at"),
    "Edit": ("edit the file", "modify the file", "edit ", "change the file", "patch the file"),
    "Write": ("write the file", "create the file", "create a file", "write a new file"),
    "Grep": ("grep", "search for", "search the code", "ripgrep", "rg "),
    "Glob": ("glob", "find files", "list files matching"),
    "MultiEdit": ("multiple edits", "multi-edit"),
    "NotebookEdit": ("notebook edit", "edit the notebook"),
    "WebFetch": ("fetch the url", "fetch the page", "web request"),
    "WebSearch": ("web search", "search the web"),
    "Task": ("subagent", "spawn an agent", "task tool"),
    "apply_patch": ("apply_patch", "apply patch", "*** begin patch", "patch envelope"),
}

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


def _normalize(text: str) -> str:
    """Lowercased text with collapsed whitespace, used for substring
    checks against tool synonyms and file basenames."""
    return re.sub(r"\s+", " ", (text or "").lower())


def _refused(text: str) -> bool:
    head = (text or "")[:300]
    return any(p.search(head) for p in _REFUSAL_PATTERNS)


def _tool_mentioned(text: str, tool: str) -> bool:
    """True if the model named ``tool`` either by canonical name or by
    one of the loose synonyms in ``_TOOL_SYNONYMS``.

    The canonical-name check is case-sensitive so plain English
    sentences (\"would edit the file\") don't accidentally match the
    capitalized ``Edit`` token; the synonym list covers the
    lowercase / paraphrased cases.
    """
    if not text:
        return False
    if re.search(rf"\b{re.escape(tool)}\b", text):
        return True
    haystack = _normalize(text)
    for syn in _TOOL_SYNONYMS.get(tool, ()):  # noqa: SIM118
        if syn in haystack:
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
