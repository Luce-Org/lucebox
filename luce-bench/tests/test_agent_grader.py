"""Tests for the synthetic ``agent`` area's shape-detection grader.

The 2026-05-30 gemma full bench showed the agent grader missing real
agent engagement from models that emit ``call:<verb>{...}`` structured
tool calls (codex-mini, DDX bead executor, etc.) — that pattern is now
recognized as a fourth pass class alongside code fences, JSON tool
calls, and apply_patch envelopes.
"""

from __future__ import annotations

from lucebench.areas.agent import grade_agent


def test_code_fence_passes():
    """Triple-backtick code fence is an agent-shape indicator."""
    g = grade_agent("", "Here's a fix:\n\n```python\nprint('hi')\n```\n")
    assert g["graded_pass"] is True
    assert g["format_pass"] is True


def test_json_tool_call_passes():
    """OpenAI-style tool-use shape — ``\"name\": \"Read\"`` — passes."""
    g = grade_agent("", '{"tool_calls": [{"function": {"name": "Read", "arguments": "..."}}]}')
    assert g["graded_pass"] is True


def test_apply_patch_envelope_passes():
    """The ``*** Begin Patch`` envelope is the codex apply_patch shape."""
    g = grade_agent(
        "",
        "*** Begin Patch\n*** Update File: foo.py\n@@\n-old\n+new\n*** End Patch",
    )
    assert g["graded_pass"] is True


def test_call_invocation_with_namespace_passes():
    """Regression for the 2026-05-30 gemma codex-large-explore case:
    response built from ``call:shell{...}`` + ``call:update_plan{...}``
    is real tool engagement and must pass."""
    response = (
        "I'll start by searching the codebase.\n\n"
        'call:update_plan{steps:[{status:in_progress,task:"search"}]}\n'
        'call:shell{command: "rg -i auth --type rust"}\n'
    )
    g = grade_agent("", response)
    assert g["graded_pass"] is True
    assert g["format_pass"] is True


def test_call_invocation_bare_verb_passes():
    """Without namespace prefix the bare ``call:<verb>{}`` still counts."""
    g = grade_agent("", "I'd use call:read_file{path: \"foo.py\"} to start.")
    assert g["graded_pass"] is True


def test_narrative_only_fails():
    """Pure prose with no agent-shape token fails — same as before."""
    g = grade_agent(
        "",
        "I would read the file and analyze its responsibilities — the way "
        "this server handles HTTP requests is the key question.",
    )
    assert g["graded_pass"] is False, "narrative without any tool shape must fail"


def test_inline_backtick_alone_does_not_pass():
    """A response that quotes a shell command inline with single
    backticks but never opens a code fence or structured call should
    still fail — single backticks are too easy a bar."""
    g = grade_agent("", "I'll use `cat foo.cpp` to read the file, then `grep` for things.")
    assert g["graded_pass"] is False


def test_too_short_fails():
    """Stub responses (< 16 chars stripped) fail regardless of shape."""
    g = grade_agent("", "```\n```")
    assert g["graded_pass"] is False
