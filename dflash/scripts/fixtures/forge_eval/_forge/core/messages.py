# Vendored from antoinezambelli/forge-guardrails v0.7.1.
# See dflash/scripts/fixtures/forge_eval/_forge/LICENSE for the upstream MIT.
# Local modifications: import paths rewritten from `forge.X` to relative imports.
"""Message types and serialization."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Any


class MessageRole(str, Enum):
    """Conversation message roles."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class MessageType(str, Enum):
    """Metadata tag for compaction prioritization."""

    SYSTEM_PROMPT = "system_prompt"
    USER_INPUT = "user_input"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    REASONING = "reasoning"
    TEXT_RESPONSE = "text_response"
    STEP_NUDGE = "step_nudge"
    PREREQUISITE_NUDGE = "prerequisite_nudge"
    RETRY_NUDGE = "retry_nudge"
    CONTEXT_WARNING = "context_warning"
    SUMMARY = "summary"


@dataclass(frozen=True)
class MessageMeta:
    """Metadata attached to a message. Never sent to the API."""

    type: MessageType
    step_index: int | None = None
    original_type: MessageType | None = None
    token_estimate: int | None = None


@dataclass(frozen=True)
class ToolCallInfo:
    """One tool call within an assistant message."""

    name: str
    args: dict[str, Any]
    call_id: str


@dataclass
class Message:
    """Internal message representation with typed metadata.

    For assistant messages with tool calls, ``tool_calls`` holds one or more
    ToolCallInfo entries.  For tool-result messages, ``tool_name`` and
    ``tool_call_id`` pair back to the originating call.
    """

    role: MessageRole
    content: str
    metadata: MessageMeta
    # Tool-result pairing fields
    tool_name: str | None = None
    tool_call_id: str | None = None
    # Assistant tool-call list (length >= 1 when present)
    tool_calls: list[ToolCallInfo] | None = None

    def to_api_dict(self, format: str = "ollama") -> dict[str, Any]:
        """Serialize for LLM API. Strips metadata.

        format="ollama": arguments as dict, no "type" field in tool_calls,
            tool results use "tool_name".
        format="openai": arguments as JSON string, "type": "function" and
            "id" required on tool_calls, tool results use "tool_call_id"
            and "name".
        """
        if self.tool_calls is not None:
            tc_list: list[dict[str, Any]] = []
            for tc in self.tool_calls:
                args: Any = tc.args or {}
                tc_entry: dict[str, Any] = {
                    "function": {
                        "name": tc.name,
                        "arguments": json.dumps(args) if format == "openai" else args,
                    },
                }
                if format == "openai":
                    tc_entry["type"] = "function"
                    tc_entry["id"] = tc.call_id
                tc_list.append(tc_entry)
            return {
                "role": self.role.value,
                "content": self.content,
                "tool_calls": tc_list,
            }
        d: dict[str, Any] = {"role": self.role.value, "content": self.content}
        if self.tool_name is not None:
            if format == "openai":
                d["name"] = self.tool_name
                if self.tool_call_id is not None:
                    d["tool_call_id"] = self.tool_call_id
            else:
                d["tool_name"] = self.tool_name
        return d
