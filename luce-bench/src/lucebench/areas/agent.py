r"""Agent-style probes for `--area agent`.

Pairs a real Codex-style system prompt (loaded from
``fixtures/agent_prompts/codex_*.md``) with a coding-task user message,
and checks whether the model produces agent-shaped output (tool calls,
code blocks, ``apply_patch`` envelopes). Complement to ``--area forge``
which exercises tool-calling protocol reliability with mock scenarios.
This probe measures the simpler upstream question: given a realistic
agent context, does the model engage as an agent at all?

Three classes of output count as PASS:

1. **Code block**: response contains a Markdown fence (\`\`\`).
2. **JSON tool-call envelope**: response contains a JSON object with a
   ``name`` field looking like a tool ("Read", "Edit", "Bash", etc.).
3. **apply_patch envelope**: response contains an ``apply_patch`` or
   ``*** Begin Patch`` string (the Codex apply-patch convention).

Failure modes the grader catches:

- Model regressed to narrative prose ("To do this, you would first...")
  without producing any code or tool envelope.
- Model produced markdown headers/paragraphs only.
- Model refused or echoed the prompt back.

Real SWE-bench-style execution grading is the follow-up (separate
``--area swe`` run-request); this probe is the lightweight signal that
the agent path is even wired up.

Why not bench_agent_loop.py: it reads ``~/.claude/projects/`` session
JSONL on the host (non-portable across machines). Why not
bench_agentic_session.py: it overlaps forge (tool-call protocol). This
module picks the gap forge doesn't cover: agent-shape on a realistic
agent system prompt.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent.parent
FIXTURE_PATH = SCRIPT_DIR / "fixtures" / "agent_cases" / "cases.json"
SYSTEM_PROMPT_DIR = SCRIPT_DIR / "fixtures" / "agent_prompts"


def load_agent_cases(path: Path = FIXTURE_PATH) -> list[dict[str, Any]]:
    """Load the vendored agent-probe cases.

    Each case loads its system prompt from ``fixtures/agent_prompts/`` at
    load time so the prompt text travels with the case (the bench row
    can include the full prompt for trace inspection without
    re-resolving paths at run time).
    """
    payload = json.loads(path.read_text())
    out: list[dict[str, Any]] = []
    for raw in payload["cases"]:
        sys_file = SYSTEM_PROMPT_DIR / raw["system_prompt_file"]
        system_prompt = sys_file.read_text() if sys_file.exists() else ""
        out.append(
            {
                "area": "agent",
                "source": "agent-shape-probe",
                "id": raw["id"],
                "kind": raw.get("kind", "agent-prompt"),
                "system_prompt": system_prompt,
                "user_message": raw["user_message"],
                # Resolved prompt = system + user; the dispatcher will send as
                # a two-message chat (system + user) when building the request.
                # Stored here for traces / readability of the JSON snapshot.
                "answer": None,
                "domain": "agent",
                "title": raw["id"],
                "_system_prompt_file": raw["system_prompt_file"],
            }
        )
    return out


AGENT_CASES = load_agent_cases()


_CODE_FENCE = re.compile(r"```")
_JSON_TOOL_CALL = re.compile(
    r'"name"\s*:\s*"(?:Read|Edit|Write|Grep|Bash|Glob|Update|Search|'
    r'Apply|apply_patch|Run|Execute|Shell)"',
    re.IGNORECASE,
)
_APPLY_PATCH = re.compile(
    r"apply_patch|\*\*\* Begin Patch",
    re.IGNORECASE,
)


def grade_agent(user_message: str, completion: str) -> dict[str, Any]:
    """Pass if the response is agent-shaped.

    See module docstring for the three PASS classes. We deliberately
    don't grade *correctness* of the agent's plan — this probe is purely
    "is the model engaging as an agent at all". Correctness is what
    --area swe is for (when it lands).
    """
    text = completion or ""
    has_code_fence = bool(_CODE_FENCE.search(text))
    has_tool_call = bool(_JSON_TOOL_CALL.search(text))
    has_apply_patch = bool(_APPLY_PATCH.search(text))
    pass_any = has_code_fence or has_tool_call or has_apply_patch
    nonempty = len(text.strip()) >= 16
    return {
        "graded_pass": pass_any and nonempty,
        "strict_pass": pass_any and nonempty,
        "format_pass": pass_any,
        "semantic_pass": pass_any and nonempty,
        # Hint = "model at least produced something non-trivial"; lets a
        # trace reader see "model talked but didn't go into agent mode".
        "semantic_hint": nonempty,
        "status": "passed" if (pass_any and nonempty) else "failed",
        "ok": pass_any and nonempty,
    }


def grade_agent_case(case: dict[str, Any], row: dict[str, Any]) -> dict[str, Any]:
    """Wrap grade_agent to match the lucebench.cli runner shape."""
    completion = row.get("content") or ""
    g = grade_agent(case.get("user_message", ""), completion)
    return {
        "pass": g["graded_pass"],
        "given": g.get("agent_shape") or ("agent_shape_ok" if g["graded_pass"] else "narrative"),
        "correct": "code_block | json_tool_call | apply_patch",
        "status": g["status"],
        "format_pass": g["format_pass"],
        "semantic_hint": g.get("semantic_hint", False),
    }
