# Vendored from antoinezambelli/forge-guardrails v0.7.1.
# See dflash/scripts/fixtures/forge_eval/_forge/LICENSE for the upstream MIT.
# Local modifications: import paths rewritten from `forge.X` to relative imports.
"""Error budget tracking — consecutive retries and tool errors."""

from __future__ import annotations


class ErrorTracker:
    """Tracks consecutive retry and tool error counts against limits.

    Stateful — instantiate per session/task.

    Args:
        max_retries: Consecutive formatting/validation failures before
            exhaustion.
        max_tool_errors: Consecutive tool execution errors before
            exhaustion. Soft errors (ToolResolutionError equivalent)
            do not count.
    """

    def __init__(self, max_retries: int = 3, max_tool_errors: int = 2) -> None:
        self.max_retries = max_retries
        self.max_tool_errors = max_tool_errors
        self._consecutive_retries = 0
        self._consecutive_tool_errors = 0

    def record_retry(self) -> None:
        """Record a validation failure (TextResponse or unknown tool)."""
        self._consecutive_retries += 1

    def reset_retries(self) -> None:
        """Reset retry counter (call on successful validation)."""
        self._consecutive_retries = 0

    def record_result(self, success: bool, is_soft_error: bool = False) -> None:
        """Record a tool execution result.

        Args:
            success: True if the tool executed without error.
            is_soft_error: True if the error is a resolution/soft error
                that should not count toward the error budget (e.g.,
                ToolResolutionError). Ignored when success is True.
        """
        if success:
            # Individual success doesn't reset — only a fully clean batch does.
            # Call reset_errors() after a batch with zero errors.
            return
        if not is_soft_error:
            self._consecutive_tool_errors += 1

    def reset_errors(self) -> None:
        """Reset tool error counter (call after a fully clean batch)."""
        self._consecutive_tool_errors = 0

    @property
    def retries_exhausted(self) -> bool:
        """True if consecutive retries exceed the limit."""
        return self._consecutive_retries > self.max_retries

    @property
    def tool_errors_exhausted(self) -> bool:
        """True if consecutive tool errors exceed the limit."""
        return self._consecutive_tool_errors > self.max_tool_errors

    @property
    def consecutive_retries(self) -> int:
        """Current consecutive retry count."""
        return self._consecutive_retries

    @property
    def consecutive_tool_errors(self) -> int:
        """Current consecutive tool error count."""
        return self._consecutive_tool_errors
