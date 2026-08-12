# Vendored from antoinezambelli/forge-guardrails v0.7.1.
# See dflash/scripts/fixtures/forge_eval/_forge/LICENSE for the upstream MIT.
# Local modifications: import paths rewritten from `forge.X` to relative imports.
"""Inference loop — compact, fold, serialize, send, validate, retry.

Extracted from WorkflowRunner so both the runner and the proxy can share
the same input-processing and validation logic. This is the "front half"
of the agentic loop: everything up to and including getting a clean
response from the LLM. The "back half" (step enforcement, tool execution,
terminal check) stays in the caller.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from ..clients.base import ChunkType, LLMClient, StreamChunk, TokenUsage
from ..context.manager import ContextManager
from .messages import Message, MessageMeta, MessageRole, MessageType, ToolCallInfo
from .workflow import LLMResponse, TextResponse, ToolCall, ToolSpec
from ..errors import StreamError, ToolCallError
from ..guardrails import ErrorTracker, ResponseValidator

# Maps Nudge.kind → MessageType for message emission.
_NUDGE_KIND_TO_TYPE: dict[str, MessageType] = {
    "retry": MessageType.RETRY_NUDGE,
    "unknown_tool": MessageType.RETRY_NUDGE,
    "step": MessageType.STEP_NUDGE,
    "prerequisite": MessageType.PREREQUISITE_NUDGE,
}


def _get_usage(client: LLMClient) -> TokenUsage | None:
    """Extract actual token count from the client."""
    last_usage = getattr(client, "last_usage", None)
    if not isinstance(last_usage, dict):
        return None
    slot_id = getattr(client, "_slot_id", None) or 0
    return last_usage.get(slot_id)


def _sync_token_count(client: LLMClient, context_manager: ContextManager) -> None:
    """Feed actual token count from the client into the context manager."""
    usage = _get_usage(client)
    if usage is not None:
        context_manager.update_token_count(usage.total_tokens)


@dataclass
class InferenceResult:
    """Result of a single inference call (may include transparent retries).

    Attributes:
        response: The validated LLM response — tool calls or text.
        new_messages: Messages generated during this call (assistant text from
            failed attempts, nudges, and the final assistant response). The
            caller should append these to their message history.
        usage: Token usage for the final successful attempt.
        tool_call_counter: Updated counter for generating unique call IDs.
    """

    response: list[ToolCall] | TextResponse
    new_messages: list[Message] = field(default_factory=list)
    usage: TokenUsage | None = None
    tool_call_counter: int = 0
    attempts: int = 1


def fold_and_serialize(
    messages: list[Message],
    api_format: str,
) -> list[dict[str, Any]]:
    """Reasoning-fold and serialize forge Messages to API dicts.

    Folds REASONING messages into the following TOOL_CALL message's content
    field so the wire format has one assistant message with both content and
    tool_calls (valid OpenAI format). Internal Message list stays separate
    for compaction.
    """
    api_messages: list[dict[str, Any]] = []
    pending_reasoning: str | None = None

    for m in messages:
        if m.metadata.type == MessageType.REASONING and m.role == MessageRole.ASSISTANT:
            pending_reasoning = m.content
            continue
        d = m.to_api_dict(format=api_format)
        if pending_reasoning is not None and m.tool_calls is not None:
            d["content"] = pending_reasoning
            pending_reasoning = None
        elif pending_reasoning is not None:
            api_messages.append({"role": "assistant", "content": pending_reasoning})
            pending_reasoning = None
        api_messages.append(d)

    if pending_reasoning is not None:
        api_messages.append({"role": "assistant", "content": pending_reasoning})

    return api_messages


def _build_tool_call_infos(
    tool_calls: list[ToolCall],
    tool_call_counter: int,
) -> tuple[list[ToolCallInfo], int]:
    """Assign call IDs to tool calls, returning infos and updated counter."""
    tc_infos = []
    for tc in tool_calls:
        tc_id = f"call_{tool_call_counter:09d}"
        tool_call_counter += 1
        tc_infos.append(ToolCallInfo(name=tc.tool, args=tc.args, call_id=tc_id))
    return tc_infos, tool_call_counter


async def run_inference(
    messages: list[Message],
    client: LLMClient,
    context_manager: ContextManager,
    validator: ResponseValidator,
    error_tracker: ErrorTracker,
    tool_specs: list[ToolSpec],
    tool_call_counter: int = 0,
    step_index: int = 0,
    step_hint: str = "",
    max_attempts: int | None = None,
    stream: bool = False,
    on_chunk: Callable[[StreamChunk], Awaitable[None]] | None = None,
    sampling: dict[str, Any] | None = None,
    passthrough: dict[str, Any] | None = None,
    inbound_anthropic_body: dict[str, Any] | None = None,
) -> InferenceResult | None:
    """Send messages to the LLM with compaction, folding, validation, and retry.

    Retries are handled internally — the caller gets a clean response or an
    exception. On each retry, the model's failed output and a corrective
    nudge are appended to the message list and returned in
    ``InferenceResult.new_messages`` so the caller can track history.

    Args:
        messages: The current conversation history (forge Messages). This
            list is mutated during compaction and retry (messages may be
            removed by compaction and added by retries).
        client: The LLM backend client.
        context_manager: For context budget compaction.
        validator: For rescue parsing, retry nudges, unknown tool checks.
        error_tracker: Tracks consecutive retry budget. The caller owns
            this object and passes it in so budget persists across calls.
        tool_specs: Available tools to send to the LLM.
        tool_call_counter: Current counter for generating unique call IDs.
            The updated value is returned in the result.
        step_index: Current iteration index (for compaction and message metadata).
        step_hint: Hint for compaction summarization.
        max_attempts: Maximum LLM calls this invocation may make (including
            retries). When None, bounded only by max_retries. The runner
            passes remaining iteration budget here so retries don't exceed
            max_iterations.
        stream: If True, use send_stream() instead of send().
        on_chunk: Async callback for streaming chunks.

    Returns:
        InferenceResult with validated tool calls, new messages, and
        updated tool_call_counter. Returns None if max_attempts is
        exhausted without a valid response (caller should treat this
        as iteration budget spent).

    Raises:
        ToolCallError: If retry budget (max_retries) is exhausted.
        StreamError: If streaming ends without a FINAL chunk.
    """
    api_format = getattr(client, "api_format", "ollama")
    new_messages: list[Message] = []
    max_retries = error_tracker.max_retries
    attempt_limit = max_retries + 1
    if max_attempts is not None:
        attempt_limit = min(attempt_limit, max_attempts)
    attempts = 0

    # Path-1 verbatim opt-in: drop on any forge mutation (compaction,
    # context warning, retry) so cache_control is only preserved on the
    # clean first-attempt call. ADR-015.
    verbatim_body = inbound_anthropic_body

    for _attempt in range(attempt_limit):
        attempts += 1

        # Compact
        compacted = context_manager.maybe_compact(
            messages, step_index=step_index, step_hint=step_hint,
        )
        # Update the caller's list in-place if compaction changed it
        if compacted is not messages:
            messages.clear()
            messages.extend(compacted)
            verbatim_body = None  # mutation

        # Check context thresholds — inject warning if crossed
        context_warning = context_manager.check_thresholds(messages)
        if context_warning:
            verbatim_body = None  # mutation

        # Fold and serialize
        api_messages = fold_and_serialize(messages, api_format)

        # Inject context warning as transient user message (not persisted
        # in conversation history). Uses "user" role because mid-conversation
        # "system" messages break Jinja chat templates on llama-server.
        # Also emit as a CONTEXT_WARNING message so on_message consumers
        # (TUI, CLI) can display it to the user.
        if context_warning:
            api_messages.append({"role": "user", "content": context_warning})
            new_messages.append(Message(
                MessageRole.USER,
                context_warning,
                MessageMeta(MessageType.CONTEXT_WARNING, step_index=step_index),
            ))

        # Send
        if stream:
            response = await _send_streaming(
                client, api_messages, tool_specs, on_chunk, sampling, passthrough,
                inbound_anthropic_body=verbatim_body,
            )
        else:
            response = await client.send(
                api_messages, tools=tool_specs, sampling=sampling, passthrough=passthrough,
                inbound_anthropic_body=verbatim_body,
            )
        # Subsequent attempts (retries) are mutations regardless of outcome.
        verbatim_body = None

        # Update context manager with real token count if available.
        _sync_token_count(client, context_manager)

        # Validate
        validation = validator.validate(response)

        if not validation.needs_retry:
            error_tracker.reset_retries()
            # Intentional text response or validated tool calls
            validated = validation.tool_calls
            return InferenceResult(
                response=validated,
                new_messages=new_messages,
                usage=_get_usage(client),
                tool_call_counter=tool_call_counter,
                attempts=attempts,
            )

        # Retry path
        error_tracker.record_retry()
        if error_tracker.retries_exhausted:
            raw = response.content if isinstance(response, TextResponse) else str(
                [(tc.tool, tc.args) for tc in response]
            )
            raise ToolCallError(
                f"Retries exhausted after {max_retries} consecutive failed attempts",
                raw_response=raw,
            )

        # Emit the assistant's failed output, then the corrective signal.
        # Two shapes:
        #   - Bare text (no tool_call to anchor on): assistant(text) + user nudge.
        #   - Unknown tool (assistant emitted a tool_call with a bad name): emit
        #     assistant(tc) + one tool-error result per tc, mirroring step/prereq
        #     enforcement in runner.py. Tool-error rides the canonical channel
        #     the model was pretrained on, surviving heavy-context attention
        #     drop-off and Mistral _merge_consecutive folding far better than a
        #     trailing user-role nudge.
        nudge = validation.nudge
        nudge_type = _NUDGE_KIND_TO_TYPE[nudge.kind]

        if isinstance(response, TextResponse):
            msg = Message(
                MessageRole.ASSISTANT,
                response.content,
                MessageMeta(MessageType.TEXT_RESPONSE, step_index=step_index),
            )
            messages.append(msg)
            new_messages.append(msg)
            # Bare text: no tool_call to attach to, fall back to user nudge.
            nudge_msg = Message(
                MessageRole.USER,
                nudge.content,
                MessageMeta(nudge_type, step_index=step_index),
            )
            messages.append(nudge_msg)
            new_messages.append(nudge_msg)
        else:
            # Unknown tool — emit reasoning + tool_call, then one tool-error
            # result per tool_call so the corrective signal rides the canonical
            # channel.
            tool_calls = response
            if tool_calls[0].reasoning:
                reasoning_msg = Message(
                    MessageRole.ASSISTANT,
                    tool_calls[0].reasoning,
                    MessageMeta(MessageType.REASONING, step_index=step_index),
                )
                messages.append(reasoning_msg)
                new_messages.append(reasoning_msg)
            tc_infos, tool_call_counter = _build_tool_call_infos(tool_calls, tool_call_counter)
            tc_msg = Message(
                MessageRole.ASSISTANT,
                "",
                MessageMeta(MessageType.TOOL_CALL, step_index=step_index),
                tool_calls=tc_infos,
            )
            messages.append(tc_msg)
            new_messages.append(tc_msg)
            for tc_info in tc_infos:
                err_msg = Message(
                    MessageRole.TOOL,
                    f"[UnknownTool] {nudge.content}",
                    MessageMeta(nudge_type, step_index=step_index),
                    tool_name=tc_info.name,
                    tool_call_id=tc_info.call_id,
                )
                messages.append(err_msg)
                new_messages.append(err_msg)

    # max_attempts exhausted without valid response — signal to caller
    return None


async def _send_streaming(
    client: LLMClient,
    api_messages: list[dict[str, Any]],
    tool_specs: list[ToolSpec],
    on_chunk: Callable[[StreamChunk], Awaitable[None]] | None = None,
    sampling: dict[str, Any] | None = None,
    passthrough: dict[str, Any] | None = None,
    inbound_anthropic_body: dict[str, Any] | None = None,
) -> LLMResponse:
    """Send via streaming, forwarding chunks to on_chunk callback."""
    response = None
    async for chunk in client.send_stream(
        api_messages, tools=tool_specs, sampling=sampling, passthrough=passthrough,
        inbound_anthropic_body=inbound_anthropic_body,
    ):
        if on_chunk is not None:
            await on_chunk(chunk)
        if chunk.type == ChunkType.FINAL:
            response = chunk.response
    if response is None:
        raise StreamError(
            "Stream ended without FINAL chunk — the client adapter "
            "may be malformed or the connection was interrupted"
        )
    return response
