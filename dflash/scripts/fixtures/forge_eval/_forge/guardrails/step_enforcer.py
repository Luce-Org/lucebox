# Vendored from antoinezambelli/forge-guardrails v0.7.1.
# See dflash/scripts/fixtures/forge_eval/_forge/LICENSE for the upstream MIT.
# Local modifications: import paths rewritten from `forge.X` to relative imports.
"""Step enforcement — required step tracking, premature terminal nudges,
and prerequisite enforcement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ..core.steps import StepTracker
from ..core.workflow import ToolCall, ToolDef
from .nudge import Nudge
from ..prompts.nudges import prerequisite_nudge, step_nudge


@dataclass
class StepCheck:
    """Result of checking tool calls against step requirements.

    If ``needs_nudge`` is True, ``nudge`` contains the message to inject.
    """

    nudge: Nudge | None
    needs_nudge: bool


class StepEnforcer:
    """Tracks required steps and enforces them with escalating nudges.

    Also enforces tool prerequisites — conditional dependencies between tools.

    Stateful — instantiate per session/task.

    Args:
        required_steps: Tool names that must be called before the terminal tool.
        terminal_tools: The tools that can end the workflow.
        tool_prerequisites: Map of tool name to its ToolDef.prerequisites list.
        max_premature_attempts: How many premature terminal attempts before
            the enforcer signals exhaustion (via StepCheck or raising).
        max_prereq_violations: How many consecutive prerequisite violations
            before the enforcer signals exhaustion.
    """

    def __init__(
        self,
        required_steps: list[str],
        terminal_tools: frozenset[str],
        tool_prerequisites: dict[str, list[str | dict[str, str]]] | None = None,
        max_premature_attempts: int = 3,
        max_prereq_violations: int = 2,
    ) -> None:
        self._tracker = StepTracker(required_steps=required_steps)
        self.terminal_tools = terminal_tools
        self._tool_prerequisites = tool_prerequisites or {}
        self.max_premature_attempts = max_premature_attempts
        self.max_prereq_violations = max_prereq_violations
        self._premature_attempts = 0
        self._consecutive_prereq_violations = 0

    def check(self, tool_calls: list[ToolCall]) -> StepCheck:
        """Check whether tool calls include a premature terminal call.

        If a terminal tool is in the batch and required steps aren't
        satisfied, returns a StepCheck with an escalating nudge. The
        escalation tier increments on each premature attempt (1=polite,
        2=direct, 3=aggressive).

        Args:
            tool_calls: The tool calls the model wants to execute.

        Returns:
            StepCheck with nudge if premature, or no nudge if clear to proceed.
        """
        has_terminal = any(tc.tool in self.terminal_tools for tc in tool_calls)

        if has_terminal and not self._tracker.is_satisfied():
            self._premature_attempts += 1
            tier = min(self._premature_attempts, 3)
            # Find which terminal tool was attempted for the nudge message
            attempted = next(
                tc.tool for tc in tool_calls if tc.tool in self.terminal_tools
            )
            return StepCheck(
                nudge=Nudge(
                    role="user",
                    content=step_nudge(
                        attempted,
                        self._tracker.pending(),
                        tier=tier,
                    ),
                    kind="step",
                    tier=tier,
                ),
                needs_nudge=True,
            )

        return StepCheck(nudge=None, needs_nudge=False)

    def check_prerequisites(self, tool_calls: list[ToolCall]) -> StepCheck:
        """Check whether any tool call has unsatisfied prerequisites.

        Evaluates against pre-batch state. Any violation in the batch blocks
        the entire batch (whole-batch blocking).

        Args:
            tool_calls: The tool calls the model wants to execute.

        Returns:
            StepCheck with nudge if any prereq is unsatisfied.
        """
        for tc in tool_calls:
            prereqs = self._tool_prerequisites.get(tc.tool)
            if not prereqs:
                continue
            result = self._tracker.check_prerequisites(tc.tool, tc.args, prereqs)
            if not result.satisfied:
                self._consecutive_prereq_violations += 1
                return StepCheck(
                    nudge=Nudge(
                        role="user",
                        content=prerequisite_nudge(
                            tc.tool, result.missing,
                        ),
                        kind="prerequisite",
                    ),
                    needs_nudge=True,
                )

        return StepCheck(nudge=None, needs_nudge=False)

    def record(self, tool_name: str, args: dict[str, Any] | None = None) -> None:
        """Record a successful tool execution."""
        self._tracker.record(tool_name, args)

    def is_satisfied(self) -> bool:
        """True if all required steps have been completed."""
        return self._tracker.is_satisfied()

    def pending(self) -> list[str]:
        """Return required steps not yet completed."""
        return self._tracker.pending()

    def terminal_reached(self, tool_calls: list[ToolCall]) -> bool:
        """True if a terminal tool is in the batch and steps are satisfied."""
        has_terminal = any(tc.tool in self.terminal_tools for tc in tool_calls)
        return has_terminal and self._tracker.is_satisfied()

    @property
    def premature_attempts(self) -> int:
        """Number of premature terminal attempts so far."""
        return self._premature_attempts

    @property
    def premature_exhausted(self) -> bool:
        """True if premature attempts exceed the limit."""
        return self._premature_attempts > self.max_premature_attempts

    @property
    def prereq_violations(self) -> int:
        """Number of consecutive prerequisite violations."""
        return self._consecutive_prereq_violations

    @property
    def prereq_exhausted(self) -> bool:
        """True if consecutive prereq violations exceed the limit."""
        return self._consecutive_prereq_violations > self.max_prereq_violations

    def reset_premature(self) -> None:
        """Reset premature attempt counter (call after a clean batch)."""
        self._premature_attempts = 0

    def reset_prereq_violations(self) -> None:
        """Reset consecutive prereq violation counter (call after a clean batch)."""
        self._consecutive_prereq_violations = 0

    @property
    def completed_steps(self) -> dict[str, None]:
        """Steps completed so far (for diagnostics / error reporting)."""
        return self._tracker.completed_steps

    def summary_hint(self) -> str:
        """Human-readable hint for context compaction."""
        return self._tracker.summary_hint()
