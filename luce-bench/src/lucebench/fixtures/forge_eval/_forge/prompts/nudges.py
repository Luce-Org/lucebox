# Vendored from antoinezambelli/forge-guardrails v0.7.1.
# See dflash/scripts/fixtures/forge_eval/_forge/LICENSE for the upstream MIT.
# Local modifications: import paths rewritten from `forge.X` to relative imports.
"""Nudge message templates for the WorkflowRunner."""

from __future__ import annotations


def retry_nudge(raw_response: str) -> str:
    """Nudge for when the model returns text instead of a tool call.

    Args:
        raw_response: The raw text the model produced (unused — kept for
            signature compatibility).
    """
    return (
        "Your previous response was not a valid tool call. "
        "You must respond with a tool call, not free text. "
        "Please try again with a valid tool call."
    )


def unknown_tool_nudge(tool_name: str, available_tools: list[str]) -> str:
    """Nudge for when the model calls a tool that doesn't exist.

    Args:
        tool_name: The tool name the model tried to call.
        available_tools: The list of valid tool names.
    """
    tools_list = ", ".join(available_tools)
    return (
        f"Tool '{tool_name}' does not exist. "
        f"Available tools: {tools_list}. "
        "Call one of them."
    )


def step_nudge(terminal_tool: str, pending_steps: list[str], tier: int = 1) -> str:
    """Escalating nudge for premature terminal tool attempts.

    Args:
        terminal_tool: The name of the terminal tool the model tried to call.
        pending_steps: The required steps that must be completed first.
        tier: Escalation level (1=polite, 2=direct, 3=aggressive). Clamped to 1-3.
    """
    tier = max(1, min(3, tier))
    steps = ", ".join(pending_steps)
    if tier == 1:
        return (
            f"You cannot call {terminal_tool} yet. "
            f"You must first complete these required steps: {steps}. "
            "Call one of them now."
        )
    if tier == 2:
        return (
            f"You must call one of these tools now: {steps}. "
            "Pick one."
        )
    return (
        f"STOP. You MUST call one of: {steps}. "
        f"Do NOT call {terminal_tool}. "
        f"Your next response MUST be a tool call to one of: {steps}."
    )


def prerequisite_nudge(tool_name: str, missing_prereqs: list[str]) -> str:
    """Nudge for when a tool is called without its prerequisites.

    Args:
        tool_name: The tool the model tried to call.
        missing_prereqs: The prerequisite tool names that haven't been called.
    """
    prereqs = ", ".join(missing_prereqs)
    return (
        f"You cannot call {tool_name} yet. "
        f"You must first call: {prereqs}. "
        "Call the prerequisite tool now."
    )
