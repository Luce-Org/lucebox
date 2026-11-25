# Vendored from antoinezambelli/forge-guardrails v0.7.1.
# See dflash/scripts/fixtures/forge_eval/_forge/LICENSE for the upstream MIT.
# Local modifications: import paths rewritten from `forge.X` to relative imports.
"""Lightweight nudge message returned by guardrail components."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Nudge:
    """A message to inject into conversation history.

    Returned by guardrail components when the model needs correction.
    The consumer maps this to their framework's message format::

        # OpenAI-style
        messages.append({"role": nudge.role, "content": nudge.content})

        # LangChain
        messages.append(HumanMessage(content=nudge.content))

    Attributes:
        role: Message role for injection ("user", "system", or "tool").
        content: The nudge text.
        kind: Identifies what generated the nudge ("retry", "unknown_tool",
            "step"). Useful for logging/metrics and for WorkflowRunner to
            map back to MessageType for compaction prioritization.
        tier: Escalation level for step nudges (0 = N/A, 1-3 = escalating).
    """

    role: str
    content: str
    kind: str
    tier: int = 0
