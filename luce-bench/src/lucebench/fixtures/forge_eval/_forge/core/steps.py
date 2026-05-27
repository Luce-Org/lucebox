# Vendored from antoinezambelli/forge-guardrails v0.7.1.
# See dflash/scripts/fixtures/forge_eval/_forge/LICENSE for the upstream MIT.
# Local modifications: import paths rewritten from `forge.X` to relative imports.
"""Required-step tracking and prerequisite enforcement."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class PrerequisiteCheck:
    """Result of checking prerequisites for a tool call.

    If ``satisfied`` is False, ``missing`` lists the prerequisite tool names
    that have not been called (or not called with matching args).
    """

    satisfied: bool
    missing: list[str]


@dataclass
class StepTracker:
    """Tracks which required steps have been completed and which tools
    have been executed (with args) for prerequisite enforcement.

    Lives on the WorkflowRunner, outside the message history.
    Compaction cannot invalidate step completion. See P0-1.
    """

    required_steps: list[str]
    completed_steps: dict[str, None] = field(default_factory=dict)
    executed_tools: dict[str, list[dict[str, Any]]] = field(default_factory=dict)

    def record(self, tool_name: str, args: dict[str, Any] | None = None) -> None:
        """Record a successful tool execution.

        Args:
            tool_name: The tool that was executed.
            args: The arguments the tool was called with. Stored for
                arg-matched prerequisite checking.
        """
        self.completed_steps[tool_name] = None
        self.executed_tools.setdefault(tool_name, []).append(args or {})

    def is_satisfied(self) -> bool:
        """True if all required steps have been called."""
        return all(s in self.completed_steps for s in self.required_steps)

    def pending(self) -> list[str]:
        """Return required steps not yet completed, preserving original order."""
        return [s for s in self.required_steps if s not in self.completed_steps]

    def check_prerequisites(
        self,
        tool_name: str,
        args: dict[str, Any],
        prerequisites: list[str | dict[str, str]],
    ) -> PrerequisiteCheck:
        """Check whether prerequisites are satisfied for a tool call.

        Args:
            tool_name: The tool about to be called (for error context).
            args: The arguments the tool is being called with.
            prerequisites: The prerequisite definitions from ToolDef.

        Returns:
            PrerequisiteCheck with satisfied=True if all prereqs are met,
            or satisfied=False with the list of unsatisfied prereq tool names.
        """
        missing: list[str] = []
        for prereq in prerequisites:
            if isinstance(prereq, str):
                # Name-only: any prior call to this tool satisfies it
                if prereq not in self.executed_tools:
                    missing.append(prereq)
            elif isinstance(prereq, dict):
                # Arg-matched: a prior call with the same arg value satisfies it
                prereq_tool = prereq["tool"]
                match_arg = prereq["match_arg"]
                required_value = args.get(match_arg)
                if prereq_tool not in self.executed_tools:
                    missing.append(prereq_tool)
                    continue
                if not any(
                    call.get(match_arg) == required_value
                    for call in self.executed_tools[prereq_tool]
                ):
                    missing.append(prereq_tool)

        return PrerequisiteCheck(satisfied=len(missing) == 0, missing=missing)

    def summary_hint(self) -> str:
        """Human-readable hint for injection into compacted summaries."""
        if not self.completed_steps:
            return "[No steps completed yet]"
        return f"[Steps completed: {', '.join(self.completed_steps)}]"
