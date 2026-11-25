# Vendored from antoinezambelli/forge-guardrails v0.7.1.
# See dflash/scripts/fixtures/forge_eval/_forge/LICENSE for the upstream MIT.
# Local modifications: import paths rewritten from `forge.X` to relative imports.
"""Anthropic API client adapter for frontier model baselines.

Translates between forge's OpenAI-style message format (what the runner
produces) and Anthropic's native Messages API format internally.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

import anthropic

from .base import ChunkType, StreamChunk, TokenUsage
from ..core.workflow import LLMResponse, TextResponse, ToolCall, ToolSpec
from ..errors import BackendError

log = logging.getLogger(__name__)


class AnthropicClient:
    """Anthropic Messages API client for Claude models.

    Uses the official anthropic SDK.  The runner serializes messages in
    OpenAI format (``api_format = "openai"``); this client converts them
    to Anthropic format before each API call.
    """

    api_format: str = "openai"

    def __init__(
        self,
        model: str,
        api_key: str | None = None,
        max_tokens: int = 4096,
        timeout: float = 300.0,
        max_retries: int = 3,
        tool_choice: str | None = None,
        recommended_sampling: bool = False,
        base_url: str | None = None,
    ) -> None:
        self.model = model
        self.max_tokens = max_tokens
        self._tool_choice = tool_choice  # "auto", "any", or None (default=auto)
        # Accepted for API symmetry across clients but currently a no-op:
        # AnthropicClient does not expose sampling kwargs through forge today.
        # The Anthropic SDK manages sampling internally.
        if recommended_sampling:
            log.debug(
                "AnthropicClient ignores recommended_sampling=True — no sampling kwargs are exposed."
            )
        sdk_kwargs: dict[str, Any] = {
            "api_key": api_key,
            "timeout": timeout,
            "max_retries": max_retries,
        }
        # base_url retargets the SDK at an Anthropic-shape downstream
        # (LiteLLM, a self-hosted proxy, etc.) — proxy path 1.
        if base_url is not None:
            sdk_kwargs["base_url"] = base_url
        self._client = anthropic.AsyncAnthropic(**sdk_kwargs)
        # Populated after each send()/send_stream() call. Slot-keyed
        # ``{slot_id: TokenUsage}`` to match LlamafileClient / OllamaClient so
        # ``inference._get_usage`` reads every client uniformly. The Anthropic
        # SDK is per-call (no shared inference slot), so we always use slot 0 —
        # same convention as OllamaClient.
        self.last_usage: dict[int, TokenUsage] = {}

    # ── Tool schema conversion ───────────────────────────────────

    @staticmethod
    def _convert_tools(tools: list[ToolSpec]) -> list[dict[str, Any]]:
        """ToolSpec list → Anthropic tool definitions."""
        return [
            {
                "name": spec.name,
                "description": spec.description,
                "input_schema": spec.get_json_schema(),
            }
            for spec in tools
        ]

    # ── Message format conversion ────────────────────────────────

    @staticmethod
    def _convert_messages(
        messages: list[dict[str, Any]],
    ) -> tuple[str | None, list[dict[str, Any]]]:
        """OpenAI-format dicts → (system_prompt, anthropic_messages).

        Handles:
        - System message extraction (→ separate ``system=`` kwarg)
        - assistant tool_calls → ``tool_use`` content blocks
        - role=tool → ``tool_result`` content blocks inside user messages
        - Unpaired tool_use (step/unknown-tool nudges) → synthetic error
          tool_results injected before the next user text
        - Consecutive same-role merging (Anthropic requires strict alternation)
        """
        system: str | None = None
        converted: list[dict[str, Any]] = []
        # Track tool_use IDs that haven't received a tool_result yet.
        pending_tool_use_ids: list[str] = []

        for msg in messages:
            role = msg["role"]

            if role == "system":
                system = msg["content"]
                continue

            if role == "assistant":
                if "tool_calls" in msg:
                    blocks: list[dict[str, Any]] = []
                    content = msg.get("content", "")
                    if content:
                        blocks.append({"type": "text", "text": content})
                    for tc in msg["tool_calls"]:
                        func = tc["function"]
                        args = func.get("arguments", "{}")
                        if isinstance(args, str):
                            args = json.loads(args)
                        tc_id = tc.get("id", f"toolu_{len(converted)}")
                        blocks.append({
                            "type": "tool_use",
                            "id": tc_id,
                            "name": func["name"],
                            "input": args,
                        })
                        pending_tool_use_ids.append(tc_id)
                    converted.append({"role": "assistant", "content": blocks})
                else:
                    converted.append({
                        "role": "assistant",
                        "content": msg.get("content", ""),
                    })
                continue

            if role == "tool":
                tc_id = msg.get("tool_call_id", "unknown")
                block: dict[str, Any] = {
                    "type": "tool_result",
                    "tool_use_id": tc_id,
                    "content": msg.get("content", ""),
                }
                if tc_id in pending_tool_use_ids:
                    pending_tool_use_ids.remove(tc_id)
                converted.append({"role": "user", "content": [block]})
                continue

            if role == "user":
                # Inject error tool_results for any unpaired tool_use blocks
                # (e.g. step nudge or unknown-tool nudge — tool was never executed).
                if pending_tool_use_ids:
                    blocks = []
                    for tc_id in pending_tool_use_ids:
                        blocks.append({
                            "type": "tool_result",
                            "tool_use_id": tc_id,
                            "content": "Not executed.",
                            "is_error": True,
                        })
                    pending_tool_use_ids.clear()
                    text = msg.get("content", "")
                    if text:
                        blocks.append({"type": "text", "text": text})
                    converted.append({"role": "user", "content": blocks})
                else:
                    converted.append({
                        "role": "user",
                        "content": msg.get("content", ""),
                    })
                continue

        # Merge consecutive same-role messages (Anthropic requires strict
        # user/assistant alternation).
        merged: list[dict[str, Any]] = []
        for msg in converted:
            if merged and merged[-1]["role"] == msg["role"]:
                prev_content = merged[-1]["content"]
                curr_content = msg["content"]
                # Normalise to list-of-blocks
                if isinstance(prev_content, str):
                    prev_blocks = [{"type": "text", "text": prev_content}]
                else:
                    prev_blocks = list(prev_content)
                if isinstance(curr_content, str):
                    curr_blocks = [{"type": "text", "text": curr_content}]
                else:
                    curr_blocks = list(curr_content)
                merged[-1] = {
                    "role": msg["role"],
                    "content": prev_blocks + curr_blocks,
                }
            else:
                merged.append(msg)

        return system, merged

    # ── Response parsing ─────────────────────────────────────────

    @staticmethod
    def _parse_response(response: Any) -> LLMResponse:
        """Anthropic Message → list[ToolCall] or TextResponse."""
        tool_uses: list[Any] = []
        text_parts: list[str] = []

        for block in response.content:
            if block.type == "tool_use":
                tool_uses.append(block)
            elif block.type == "text":
                text_parts.append(block.text)

        if tool_uses:
            reasoning = "\n".join(text_parts) if text_parts else None
            return [
                ToolCall(
                    tool=tu.name,
                    args=dict(tu.input),
                    reasoning=reasoning if i == 0 else None,
                )
                for i, tu in enumerate(tool_uses)
            ]
        return TextResponse(content="\n".join(text_parts))

    # ── API methods ──────────────────────────────────────────────

    def _build_kwargs(
        self,
        messages: list[dict[str, Any]],
        tools: list[ToolSpec] | None,
        passthrough: dict[str, Any] | None = None,
        inbound_anthropic_body: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build kwargs dict for messages.create / messages.stream.

        ``passthrough`` (proxy use) supplies inbound-body fields the proxy
        didn't deconstruct (``max_tokens``, ``stop_sequences``, ``tool_choice``,
        ``metadata``, ``thinking``, etc.). Forge-owned fields (``model``,
        ``messages``, ``system``, ``tools``) overlay it.

        ``inbound_anthropic_body`` (proxy path 1) bypasses the deconstruct/
        rebuild path: the SDK is called with the original inbound body
        verbatim, preserving block-level fields like ``cache_control`` that
        forge.Message doesn't represent. The runner only passes this on the
        clean first-attempt call (no compaction, no retries) — see ADR-015.
        """
        if inbound_anthropic_body is not None:
            # Verbatim emit. Drop the proxy-internal ``stream`` field; the
            # SDK call shape (messages.create vs messages.stream) selects
            # streaming. ``model`` defaults to the inbound value but the
            # client's configured model wins if the inbound omitted it.
            kwargs = dict(inbound_anthropic_body)
            kwargs.pop("stream", None)
            kwargs.setdefault("model", self.model)
            return kwargs

        system, converted = self._convert_messages(messages)
        kwargs = dict(passthrough or {})
        kwargs.update({
            "model": self.model,
            "messages": converted,
        })
        kwargs.setdefault("max_tokens", self.max_tokens)
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = self._convert_tools(tools)
            if self._tool_choice and "tool_choice" not in kwargs:
                kwargs["tool_choice"] = {"type": self._tool_choice}
        return kwargs

    async def send(
        self,
        messages: list[dict[str, str]],
        tools: list[ToolSpec] | None = None,
        sampling: dict[str, Any] | None = None,
        passthrough: dict[str, Any] | None = None,
        inbound_anthropic_body: dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Send messages via the Anthropic Messages API.

        ``sampling`` is accepted for protocol symmetry but ignored —
        AnthropicClient does not currently expose sampling kwargs through
        forge. ``passthrough`` merges inbound-body extras into the SDK call.
        ``inbound_anthropic_body`` (path 1) triggers verbatim emit — see
        ADR-015 for the cache_control preservation rationale.
        """
        if sampling:
            log.debug(
                "AnthropicClient ignores per-call sampling overrides: %s",
                sorted(sampling.keys()),
            )
        kwargs = self._build_kwargs(
            messages, tools, passthrough, inbound_anthropic_body,
        )
        try:
            response = await self._client.messages.create(**kwargs)
        except anthropic.APIError as exc:
            raise BackendError(
                getattr(exc, "status_code", 0), str(exc)
            ) from exc
        self.last_usage = {
            0: TokenUsage(
                prompt_tokens=response.usage.input_tokens,
                completion_tokens=response.usage.output_tokens,
                total_tokens=response.usage.input_tokens + response.usage.output_tokens,
            )
        }
        return self._parse_response(response)

    async def send_stream(
        self,
        messages: list[dict[str, str]],
        tools: list[ToolSpec] | None = None,
        sampling: dict[str, Any] | None = None,
        passthrough: dict[str, Any] | None = None,
        inbound_anthropic_body: dict[str, Any] | None = None,
    ) -> AsyncIterator[StreamChunk]:
        """Stream via the Anthropic Messages API.

        ``sampling`` is accepted for protocol symmetry but ignored.
        ``passthrough`` merges inbound-body extras into the SDK call.
        ``inbound_anthropic_body`` (path 1) triggers verbatim emit; see ADR-015.
        """
        if sampling:
            log.debug(
                "AnthropicClient ignores per-call sampling overrides: %s",
                sorted(sampling.keys()),
            )
        kwargs = self._build_kwargs(
            messages, tools, passthrough, inbound_anthropic_body,
        )

        accumulated_text = ""
        # Track multiple tool_use blocks by index.
        tool_blocks: list[dict[str, str]] = []  # [{name, args}, ...]
        _current_tool_idx: int = -1
        try:
            async with self._client.messages.stream(**kwargs) as stream:
                async for event in stream:
                    if event.type == "content_block_start":
                        if event.content_block.type == "tool_use":
                            tool_blocks.append({
                                "name": event.content_block.name,
                                "args": "",
                            })
                            _current_tool_idx = len(tool_blocks) - 1
                    elif event.type == "content_block_delta":
                        if event.delta.type == "text_delta":
                            accumulated_text += event.delta.text
                            yield StreamChunk(
                                type=ChunkType.TEXT_DELTA,
                                content=event.delta.text,
                            )
                        elif event.delta.type == "input_json_delta" and _current_tool_idx >= 0:
                            tool_blocks[_current_tool_idx]["args"] += event.delta.partial_json
                            yield StreamChunk(
                                type=ChunkType.TOOL_CALL_DELTA,
                                content=event.delta.partial_json,
                            )
                    elif event.type == "content_block_stop":
                        # Reset current tool index when a block finishes
                        _current_tool_idx = -1
                    elif event.type == "message_stop":
                        if tool_blocks:
                            reasoning = accumulated_text or None
                            final: LLMResponse = [
                                ToolCall(
                                    tool=tb["name"],
                                    args=json.loads(tb["args"]) if tb["args"] else {},
                                    reasoning=reasoning if i == 0 else None,
                                )
                                for i, tb in enumerate(tool_blocks)
                            ]
                        else:
                            final = TextResponse(content=accumulated_text)
                        yield StreamChunk(
                            type=ChunkType.FINAL, response=final
                        )
                # Grab usage from the final accumulated message.
                final_message = await stream.get_final_message()
                self.last_usage = {
                    0: TokenUsage(
                        prompt_tokens=final_message.usage.input_tokens,
                        completion_tokens=final_message.usage.output_tokens,
                        total_tokens=final_message.usage.input_tokens
                        + final_message.usage.output_tokens,
                    )
                }
        except anthropic.APIError as exc:
            raise BackendError(
                getattr(exc, "status_code", 0), str(exc)
            ) from exc

    async def get_context_length(self) -> int | None:
        """Claude models have 200K context."""
        return 200_000
