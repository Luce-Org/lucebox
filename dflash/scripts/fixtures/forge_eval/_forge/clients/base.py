# Vendored from antoinezambelli/forge-guardrails v0.7.1.
# See dflash/scripts/fixtures/forge_eval/_forge/LICENSE for the upstream MIT.
# Local modifications: import paths rewritten from `forge.X` to relative imports.
"""Streaming types and LLM client protocol."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from ..core.workflow import LLMResponse, ToolCall, TextResponse, ToolSpec


@dataclass(frozen=True)
class TokenUsage:
    """Token counts from a single LLM response.

    Populated from the server's ``usage`` field when available (e.g.
    llama-server).  Backends that don't report usage leave the client's
    ``last_usage`` empty and the context manager falls back to heuristic
    estimation.
    """

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


# Both Ollama and llama-server use the OpenAI tool schema format today.
# If a backend diverges, move this back into the relevant client module.
def format_tool(spec: ToolSpec) -> dict[str, Any]:
    """Convert a ToolSpec into the OpenAI-compatible tool schema."""
    return {
        "type": "function",
        "function": {
            "name": spec.name,
            "description": spec.description,
            "parameters": spec.get_json_schema(),
        },
    }


class ChunkType(str, Enum):
    """What kind of partial data a stream chunk carries."""

    TEXT_DELTA = "text_delta"
    TOOL_CALL_DELTA = "tool_call_delta"
    FINAL = "final"
    RETRY = "retry"


@dataclass(frozen=True)
class StreamChunk:
    """A single chunk from a streaming LLM response.

    Consumers (UI, logging) process TEXT_DELTA and TOOL_CALL_DELTA as they
    arrive. The runner ignores all chunks except FINAL, which carries the
    resolved response. On RETRY, consumers should discard the partial output
    from the failed attempt.
    """

    type: ChunkType
    content: str = ""
    response: LLMResponse | None = None


@runtime_checkable
class LLMClient(Protocol):
    """Interface that client adapters implement.

    The client is responsible for:
    1. Sending messages to the LLM backend
    2. Parsing the response into ToolCall or TextResponse
    3. Handling native FC or prompt-injected calling internally
    4. Optionally streaming partial responses via send_stream()

    The client does NOT retry. Retry logic lives in the WorkflowRunner.
    """

    api_format: str
    """Wire format for Message.to_api_dict(): 'ollama' or 'openai'."""

    async def send(
        self,
        messages: list[dict[str, str]],
        tools: list[ToolSpec] | None = None,
        sampling: dict[str, Any] | None = None,
        passthrough: dict[str, Any] | None = None,
        inbound_anthropic_body: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Send messages and return a parsed response.

        Returns list[ToolCall] if the model produced valid tool invocations.
        Returns TextResponse if the model produced text (reasoning, refusal,
        or malformed output that couldn't be parsed as a tool call).

        The runner inspects the response and decides whether to retry.

        Args:
            messages: API-format messages to send.
            tools: Tool specs to include with the request.
            sampling: Optional per-call sampling overrides
                (``temperature``, ``top_p``, ``top_k``, ``min_p``,
                ``repeat_penalty``, ``presence_penalty``, ``seed``).
                Per-call values win over instance state for this call only;
                the client's instance fields are not mutated.
            passthrough: Optional dict of inbound body fields forge doesn't
                own. The client merges these into the outbound body before
                overlaying its own fields (model, messages, tools, sampling).
                Used by the proxy to preserve user intent (max_tokens, stop,
                tool_choice, etc.) without forge having to enumerate every
                supported field. None = no extras to merge.
            inbound_anthropic_body: Path-1 only — when set, the AnthropicClient
                will send this body verbatim (bypassing its deconstruct/rebuild
                path) to preserve block-level Anthropic fields like
                ``cache_control``. The runner clears this kwarg on any
                forge-mutation (retry / compaction / context warning) so
                only the clean first-attempt call rides verbatim. Other
                clients accept and ignore. See ADR-015.
        """
        ...

    async def send_stream(
        self,
        messages: list[dict[str, str]],
        tools: list[ToolSpec] | None = None,
        sampling: dict[str, Any] | None = None,
        passthrough: dict[str, Any] | None = None,
        inbound_anthropic_body: dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Send messages and yield streaming chunks.

        Yields TEXT_DELTA or TOOL_CALL_DELTA chunks as they arrive.
        The final chunk has type FINAL and carries the resolved LLMResponse
        (same list[ToolCall] | TextResponse as send() would return).

        The runner forwards chunks to its on_chunk callback for UI/logging,
        then inspects the FINAL chunk and decides whether to retry.

        Args:
            messages: API-format messages to send.
            tools: Tool specs to include with the request.
            sampling: Optional per-call sampling overrides (see ``send``).
                Per-call values win over instance state without mutating self.
            passthrough: Optional inbound-body extras dict (see ``send``).
            inbound_anthropic_body: Optional path-1 verbatim body (see ``send``).
        """
        ...

    async def get_context_length(self) -> int | None:
        """Query the backend for its configured context window size."""
        ...
