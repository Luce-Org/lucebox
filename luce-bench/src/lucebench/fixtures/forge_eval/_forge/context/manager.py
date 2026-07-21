# Vendored from antoinezambelli/forge-guardrails v0.7.1.
# See dflash/scripts/fixtures/forge_eval/_forge/LICENSE for the upstream MIT.
# Local modifications: import paths rewritten from `forge.X` to relative imports.
"""Context manager for budget tracking and compaction triggering."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .strategies import CompactStrategy
from ..core.messages import Message


@dataclass(frozen=True)
class CompactEvent:
    """Emitted by ContextManager when compaction fires."""

    step_index: int
    tokens_before: int
    tokens_after: int
    budget_tokens: int
    messages_before: int
    messages_after: int
    phase_reached: int


# ── Default context warning ──────────────────────────────────────


def default_context_warning(tokens: int, budget: int, pct: float) -> str | None:
    """Default context threshold callback.

    Returns an escalating warning based on how full the context is.
    """
    if pct >= 0.80:
        return (
            f"[Context usage: {pct:.0%} ({tokens:,} / {budget:,} tokens). "
            "Context is nearly full. Older tool results and reasoning will be "
            "compacted soon — key information may be lost. Summarize critical "
            "findings now and prioritize completing the current task.]"
        )
    if pct >= 0.65:
        return (
            f"[Context usage: {pct:.0%} ({tokens:,} / {budget:,} tokens). "
            "Context is filling up. When compaction triggers, older tool results "
            "and reasoning will be condensed. Be concise in your responses and "
            "front-load important information.]"
        )
    return (
        f"[Context usage: {pct:.0%} ({tokens:,} / {budget:,} tokens). "
        "Be mindful of context usage.]"
    )


class ContextManager:
    """Manages context window budget and triggers compaction."""

    def __init__(
        self,
        strategy: CompactStrategy,
        budget_tokens: int,
        on_compact: Callable[[CompactEvent], None] | None = None,
        context_thresholds: list[float] | None = None,
        on_context_threshold: Callable[[int, int, float], str | None] | None = None,
    ) -> None:
        """
        Args:
            strategy: Compaction strategy to use. The strategy owns its own
                compaction thresholds (e.g. ``TieredCompact(compact_threshold=0.75)``
                or ``TieredCompact(phase_thresholds=(0.6, 0.75, 0.9))``).
            budget_tokens: Maximum context budget in tokens.
            on_compact: Callback invoked when compaction fires. Receives a
                CompactEvent with before/after token counts, phase reached,
                and which messages were affected. Use for logging, debugging,
                or surfacing compaction to a UI.
            context_thresholds: Sorted list of budget fractions (e.g.
                ``[0.5, 0.65, 0.8]``) at which to fire the context
                threshold callback. Each threshold fires at most once per
                session (resets if usage drops below it after compaction).
                Defaults to None (disabled).
            on_context_threshold: Callback invoked when a context threshold
                is crossed. Receives ``(tokens, budget, pct)`` and returns
                an optional string to inject as a transient system message
                before the next inference call. Return None to skip
                injection. Defaults to None (disabled).
        """
        self.strategy = strategy
        self.budget_tokens = budget_tokens
        self.on_compact = on_compact
        self._context_thresholds = sorted(context_thresholds) if context_thresholds else []
        self._on_context_threshold = on_context_threshold
        self._fired_thresholds: set[float] = set()
        self._last_known_tokens: int | None = None

    def update_token_count(self, total_tokens: int) -> None:
        """Record actual token count from the backend.

        Called after each LLM response when the backend reports usage.
        Subsequent calls to ``estimate_tokens`` will return this value
        until the next update.
        """
        self._last_known_tokens = total_tokens

    def estimate_tokens(self, messages: list[Message]) -> int:
        """Return actual token count if available, else char/4 heuristic."""
        if self._last_known_tokens is not None:
            return self._last_known_tokens
        return sum(len(m.content) for m in messages) // 4

    def check_thresholds(self, messages: list[Message]) -> str | None:
        """Check context thresholds and return an optional warning to inject.

        Fires the ``on_context_threshold`` callback when usage crosses a
        configured threshold for the first time. Thresholds reset if usage
        drops below them (e.g. after compaction).

        Returns:
            A string to inject as a transient system message, or None.
        """
        if not self._context_thresholds or not self._on_context_threshold:
            return None

        tokens = self.estimate_tokens(messages)
        if self.budget_tokens <= 0:
            return None

        pct = tokens / self.budget_tokens

        # Reset thresholds that usage has dropped below (after compaction)
        self._fired_thresholds = {t for t in self._fired_thresholds if pct >= t}

        # Find the highest unfired threshold that has been crossed
        highest_crossed: float | None = None
        for threshold in self._context_thresholds:
            if pct >= threshold and threshold not in self._fired_thresholds:
                highest_crossed = threshold

        if highest_crossed is None:
            return None

        self._fired_thresholds.add(highest_crossed)
        return self._on_context_threshold(tokens, self.budget_tokens, pct)

    def maybe_compact(
        self,
        messages: list[Message],
        step_index: int = 0,
        step_hint: str = "",
    ) -> list[Message]:
        """Delegate to the strategy, which owns threshold logic."""
        tokens_before = self.estimate_tokens(messages)

        result, phase = self.strategy.compact(
            messages, self.budget_tokens, step_hint=step_hint
        )

        if phase == 0:
            return messages

        if self.on_compact is not None:
            event = CompactEvent(
                step_index=step_index,
                tokens_before=tokens_before,
                tokens_after=self.estimate_tokens(result),
                budget_tokens=self.budget_tokens,
                messages_before=len(messages),
                messages_after=len(result),
                phase_reached=phase,
            )
            self.on_compact(event)

        return result
