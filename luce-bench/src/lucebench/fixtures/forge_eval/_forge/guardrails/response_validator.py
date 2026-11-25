# Vendored from antoinezambelli/forge-guardrails v0.7.1.
# See dflash/scripts/fixtures/forge_eval/_forge/LICENSE for the upstream MIT.
# Local modifications: import paths rewritten from `forge.X` to relative imports.
"""Response validation — rescue, retry, and unknown-tool nudges."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from ..core.workflow import LLMResponse, TextResponse, ToolCall
from .nudge import Nudge
from ..prompts.nudges import retry_nudge, unknown_tool_nudge
from ..prompts.templates import rescue_tool_call


@dataclass
class ValidationResult:
    """Result of validating an LLM response.

    Exactly one of ``tool_calls`` or ``nudge`` is set:
    - If ``needs_retry`` is False: ``tool_calls`` contains validated tool calls.
    - If ``needs_retry`` is True: ``nudge`` contains the message to inject.
    """

    tool_calls: list[ToolCall] | None
    nudge: Nudge | None
    needs_retry: bool


class ResponseValidator:
    """Validates LLM responses: rescues tool calls from text, checks tool names.

    Stateless — safe to reuse across turns and sessions.

    Args:
        tool_names: Valid tool names for this workflow.
        rescue_enabled: If True, attempt to parse tool calls from TextResponse
            before generating a retry nudge.
        retry_nudge_fn: Custom nudge function for bare text responses. Takes
            the raw response text and returns the nudge message. If None,
            uses the default retry nudge from ``forge.prompts.nudges``.
    """

    def __init__(
        self,
        tool_names: list[str],
        rescue_enabled: bool = True,
        retry_nudge_fn: Callable[[str], str] | None = None,
    ) -> None:
        self.tool_names = tool_names
        self.rescue_enabled = rescue_enabled
        self._retry_nudge_fn = retry_nudge_fn or retry_nudge

    def validate(
        self, response: LLMResponse,
    ) -> ValidationResult:
        """Validate an LLM response.

        Args:
            response: Either a TextResponse or a list of ToolCall objects.

        Returns:
            ValidationResult with tool_calls on success, or a Nudge on failure.
        """
        # TextResponse: rescue, then retry nudge
        if isinstance(response, TextResponse):
            if self.rescue_enabled:
                rescued = rescue_tool_call(response.content, self.tool_names)
                if rescued:
                    return ValidationResult(
                        tool_calls=rescued, nudge=None, needs_retry=False
                    )
            return ValidationResult(
                tool_calls=None,
                nudge=Nudge(
                    role="user",
                    content=self._retry_nudge_fn(response.content),
                    kind="retry",
                ),
                needs_retry=True,
            )

        # list[ToolCall]: check for unknown tools
        tool_calls = response
        unknown = [tc for tc in tool_calls if tc.tool not in self.tool_names]
        if unknown:
            return ValidationResult(
                tool_calls=None,
                nudge=Nudge(
                    role="user",
                    content=unknown_tool_nudge(unknown[0].tool, self.tool_names),
                    kind="unknown_tool",
                ),
                needs_retry=True,
            )

        return ValidationResult(
            tool_calls=tool_calls, nudge=None, needs_retry=False
        )
