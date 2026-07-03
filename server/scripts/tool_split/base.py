"""Base types for LLM-specific tool / conversation prompt splitting."""
from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class PromptSplit:
    """Tokenized prompt decomposed for split KV + PFlash.

    ``tool_prefix_ids`` is the stable segment pinned in a tool KV slot
    (system header + tool schema tokens). ``conversation_ids`` is the
    suffix that may grow each turn and is eligible for PFlash compression.
    ``full_ids`` is the complete prompt (tool prefix + conversation).
    """

    full_ids: list[int]
    tool_prefix_ids: list[int]
    conversation_ids: list[int]
    tool_prefix_len: int
    profile: str

    def __post_init__(self) -> None:
        if self.tool_prefix_len != len(self.tool_prefix_ids):
            raise ValueError("tool_prefix_len must match len(tool_prefix_ids)")
        if self.full_ids != self.tool_prefix_ids + self.conversation_ids:
            raise ValueError("full_ids must equal tool_prefix_ids + conversation_ids")

    @property
    def conversation_len(self) -> int:
        return len(self.conversation_ids)


@dataclass
class ToolSplitPlan:
    """Daemon command plan produced by :class:`ToolSplitOrchestrator`."""

    prompt_bin_path: str
    prompt_token_count: int
    tool_slot: int | None = None
    conv_restore_slot: int | None = None
    conv_restore_prefix_len: int = 0
    use_restore_chain: bool = False
    thin_slot_ids: list[int] = field(default_factory=list)
    inline_snap: tuple[int, int] | None = None  # (slot_id, cut_pos)
    compression_fired: bool = False
    started_in_thinking: bool = False
    tools_fingerprint: str | None = None
    pending_tool_snap: tuple[int, int] | None = None  # (slot, kv_end) post-response
    tool_prefix_len: int = 0


@dataclass
class ToolRequestContext:
    """Per-request tool-split state for daemon command building."""

    split: PromptSplit | None = None
    fingerprint: str | None = None
    tool_slot_hit: int | None = None
    pending_tool_snap: tuple[int, int] | None = None


class ToolSplitAdapter(ABC):
    """Per-LLM adapter: tokenize tools separately from conversation."""

    #: Registry name (e.g. ``qwen3``, ``laguna``).
    profile_name: str

    @classmethod
    @abstractmethod
    def detect(cls, *, arch: str, tokenizer_id: str) -> bool:
        """Return True if this adapter should handle *arch* / tokenizer."""

    @abstractmethod
    def split_prompt(
        self,
        tokenizer: Any,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None,
        *,
        chat_template_kwargs: Mapping[str, Any] | None = None,
        enable_thinking: bool = False,
    ) -> PromptSplit:
        """Split a chat request into tool-prefix and conversation token spans."""

    def canonical_tools(
        self, tools: Sequence[Mapping[str, Any]] | None
    ) -> list[dict[str, Any]]:
        """Normalize tool defs for stable cache keys (override per model)."""
        if not tools:
            return []
        out: list[dict[str, Any]] = []
        for t in tools:
            if isinstance(t, dict):
                fn = t.get("function") or {}
                name = fn.get("name", "")
                out.append({
                    "type": t.get("type", "function"),
                    "function": {
                        "name": name,
                        "description": fn.get("description", ""),
                        "parameters": fn.get("parameters") or {},
                    },
                })
        out.sort(key=lambda x: (x.get("function") or {}).get("name", ""))
        return out

    def tools_fingerprint(self, tools: Sequence[Mapping[str, Any]] | None) -> str:
        """SHA-256 hex digest of canonical tool definitions."""
        payload = json.dumps(
            self.canonical_tools(tools),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def supports_pflash_on_conversation(self) -> bool:
        """Whether PFlash may run on ``conversation_ids`` only."""
        return True
