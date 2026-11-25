# Vendored from antoinezambelli/forge-guardrails v0.7.1.
# See dflash/scripts/fixtures/forge_eval/_forge/LICENSE for the upstream MIT.
# Local modifications: import paths rewritten from `forge.X` to relative imports.
"""Compaction strategies for context window management."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..core.messages import Message, MessageMeta, MessageRole, MessageType


def _estimate_tokens(messages: list[Message]) -> int:
    return sum(len(m.content) for m in messages) // 4


class CompactStrategy(ABC):
    """Interface for context compaction strategies.

    Recommended compaction priority (cut first -> preserve longest):
    1. step_nudge, retry_nudge — ephemeral corrections, no long-term value
    2. tool_result — truncate to first line; raw data is expendable once processed
    3. tool_call — collapse to one-liner (tool name + args)
    4. reasoning — preserve as long as possible; this is the model's interpretive context
    5. Recent iterations (within keep_recent window) — fully intact
    """

    @abstractmethod
    def compact(
        self,
        messages: list[Message],
        budget_tokens: int,
        *,
        step_hint: str = "",
    ) -> tuple[list[Message], int]:
        """Return a compacted copy of the message history and the phase reached.

        Returns a tuple of (compacted_messages, phase_reached). The phase int
        indicates how aggressively the strategy compacted: 0 means no
        compaction was applied, 1+ is implementation-defined. Strategies
        without internal phases should return 1.

        The strategy owns its own threshold logic. It receives the full
        budget_tokens and decides whether to compact and how aggressively.
        Return phase 0 if no compaction was needed.

        Must preserve (never cut):
        - The system prompt (messages[0])
        - The original user input (messages[1])
        """
        ...


class NoCompact(CompactStrategy):
    """Passthrough strategy. Returns messages unchanged.

    Use when VRAM is abundant (32GB+) or workflows are short.
    """

    def compact(
        self,
        messages: list[Message],
        budget_tokens: int,
        *,
        step_hint: str = "",
    ) -> tuple[list[Message], int]:
        return list(messages), 0


class SlidingWindowCompact(CompactStrategy):
    """Keeps the system prompt, original user input, and the last N iterations.

    Simple and predictable. Good baseline for testing. Uses step_index to
    identify iteration boundaries (handles variable-size parallel tool batches).
    """

    def __init__(self, keep_recent: int, compact_threshold: float = 0.75) -> None:
        self.keep_recent = keep_recent
        self.compact_threshold = compact_threshold

    def compact(
        self,
        messages: list[Message],
        budget_tokens: int,
        *,
        step_hint: str = "",
    ) -> tuple[list[Message], int]:
        trigger = int(budget_tokens * self.compact_threshold)
        if _estimate_tokens(messages) < trigger:
            return list(messages), 0
        eligible_end = TieredCompact._find_eligible_end(messages, self.keep_recent)
        if eligible_end <= 2:
            return list(messages), 1
        return [messages[0], messages[1]] + messages[eligible_end:], 1


class TieredCompact(CompactStrategy):
    """Three-phase compaction with explicit priority order.

    Each phase fires only if the previous phase didn't reduce tokens below
    the trigger threshold. keep_recent controls how many recent loop iterations
    (each iteration = one assistant message + N tool result messages) are
    fully preserved before older content is eligible for compaction.

    Phase priority (cut first -> preserve longest):
    1. Nudges/retries dropped, tool_results truncated to first ~200 chars
    2. Tool_results dropped entirely — reasoning and text_response preserved
    3. Reasoning and text_response dropped — only tool_call skeleton remains
    """

    TRUNCATE_CHARS = 200

    def __init__(
        self,
        keep_recent: int = 2,
        compact_threshold: float = 0.75,
        phase_thresholds: tuple[float, float, float] | None = None,
    ) -> None:
        """
        Args:
            keep_recent: Number of recent loop iterations to keep fully intact.
                Tune based on workflow depth — shallow workflows (3-5 steps)
                can use 2-3, deep workflows (8-10+) may need 4-6.
            compact_threshold: Fraction of budget that triggers compaction.
                Used as the threshold for all three phases when
                phase_thresholds is not set.
            phase_thresholds: Per-phase compaction thresholds as fractions
                of the context budget. A tuple of (phase1, phase2, phase3).
                Example: ``(0.60, 0.75, 0.90)`` means Phase 1 fires at 60%,
                Phase 2 at 75%, Phase 3 at 90%. Overrides compact_threshold.
        """
        self.keep_recent = keep_recent
        if phase_thresholds is not None:
            self._phase_triggers = phase_thresholds
        else:
            self._phase_triggers = (compact_threshold, compact_threshold, compact_threshold)

    @staticmethod
    def _find_eligible_end(messages: list[Message], keep_recent: int) -> int:
        """Find the boundary index: messages before this are eligible for compaction.

        Uses step_index from message metadata to identify iteration boundaries.
        With parallel tool calls, one iteration may produce variable numbers of
        messages (1 TOOL_CALL + N TOOL_RESULTs), so counting by step_index is
        more accurate than a flat message count.
        """
        # Collect distinct step_index values from messages after the protected
        # header (messages[0] and [1] have step_index=None).
        seen_steps: list[int] = []
        for m in messages[2:]:
            si = m.metadata.step_index
            if si is not None and (not seen_steps or seen_steps[-1] != si):
                seen_steps.append(si)

        if len(seen_steps) <= keep_recent:
            # Not enough iterations to compact anything
            return 2

        # Protect the last keep_recent iterations
        cutoff_step = seen_steps[-keep_recent]
        # Find the first message index with step_index >= cutoff_step
        for i in range(2, len(messages)):
            si = messages[i].metadata.step_index
            if si is not None and si >= cutoff_step:
                return i
        return len(messages)

    def compact(
        self,
        messages: list[Message],
        budget_tokens: int,
        *,
        step_hint: str = "",
    ) -> tuple[list[Message], int]:
        """Apply tiered compaction: Phase 1 -> Phase 2 -> Phase 3.

        Each phase has its own threshold (fraction of budget_tokens).
        A phase only runs if estimated tokens exceed its threshold.
        """
        tokens = _estimate_tokens(messages)
        t1 = int(budget_tokens * self._phase_triggers[0])
        t2 = int(budget_tokens * self._phase_triggers[1])
        t3 = int(budget_tokens * self._phase_triggers[2])

        # Nothing to do if below the lowest threshold
        if tokens < t1:
            return list(messages), 0

        # Determine the boundary: everything before this index is eligible
        # messages[0] and messages[1] are always protected
        eligible_end = self._find_eligible_end(messages, self.keep_recent)

        # Phase 1: Drop nudges/retries, truncate tool_results to first line
        result = self._phase1(messages, eligible_end)
        if _estimate_tokens(result) < t2:
            return result, 1

        # Phase 2: Phase 1 + drop tool_results entirely
        result = self._phase2(messages, eligible_end)
        if _estimate_tokens(result) < t3:
            return result, 2

        # Phase 3: Phase 2 + drop reasoning and text_response (tool_call skeleton only)
        result = self._phase3(messages, eligible_end)
        return result, 3

    def _phase1(
        self, messages: list[Message], eligible_end: int
    ) -> list[Message]:
        """Drop nudges/retries and truncate tool_results outside keep_recent."""
        result: list[Message] = []
        for i, msg in enumerate(messages):
            if 2 <= i < eligible_end:
                if msg.metadata.type in (
                    MessageType.STEP_NUDGE,
                    MessageType.PREREQUISITE_NUDGE,
                    MessageType.RETRY_NUDGE,
                ):
                    continue
                if msg.metadata.type == MessageType.TOOL_RESULT:
                    if len(msg.content) > self.TRUNCATE_CHARS:
                        kept = msg.content[:self.TRUNCATE_CHARS]
                        removed = len(msg.content) - self.TRUNCATE_CHARS
                        result.append(
                            Message(
                                role=msg.role,
                                content=f"{kept}\n[Truncated — {removed} chars removed]",
                                metadata=msg.metadata,
                            )
                        )
                        continue
            result.append(msg)
        return result

    def _phase2(
        self, messages: list[Message], eligible_end: int
    ) -> list[Message]:
        """Phase 1 + drop tool_results entirely. Reasoning and text preserved."""
        result: list[Message] = []
        for i, msg in enumerate(messages):
            if 2 <= i < eligible_end:
                if msg.metadata.type in (
                    MessageType.STEP_NUDGE,
                    MessageType.PREREQUISITE_NUDGE,
                    MessageType.RETRY_NUDGE,
                    MessageType.TOOL_RESULT,
                ):
                    continue
            result.append(msg)
        return result

    def _phase3(
        self, messages: list[Message], eligible_end: int
    ) -> list[Message]:
        """Phase 2 + drop reasoning and text_response. Tool_call skeleton only."""
        result: list[Message] = []
        for i, msg in enumerate(messages):
            if 2 <= i < eligible_end:
                if msg.metadata.type in (
                    MessageType.STEP_NUDGE,
                    MessageType.PREREQUISITE_NUDGE,
                    MessageType.RETRY_NUDGE,
                    MessageType.TOOL_RESULT,
                    MessageType.REASONING,
                    MessageType.TEXT_RESPONSE,
                ):
                    continue
            result.append(msg)
        return result
