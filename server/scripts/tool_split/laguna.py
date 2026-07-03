"""Laguna / Poolside XML chat-template splitter (stub + boundary fallback)."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from tool_split.base import PromptSplit, ToolSplitAdapter
from tool_split.registry import register_adapter

_LAGUNA_ARCHES = frozenset({"laguna"})
_LAGUNA_TOKENIZER_HINTS = ("laguna", "poolside")


@register_adapter("laguna")
class LagunaToolSplitAdapter(ToolSplitAdapter):
    """Laguna ``<system>`` / ``<user>`` XML templates."""

    @classmethod
    def detect(cls, *, arch: str, tokenizer_id: str) -> bool:
        a = (arch or "").lower()
        t = (tokenizer_id or "").lower()
        if a in _LAGUNA_ARCHES:
            return True
        return any(h in t for h in _LAGUNA_TOKENIZER_HINTS)

    def split_prompt(
        self,
        tokenizer: Any,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None,
        *,
        chat_template_kwargs: Mapping[str, Any] | None = None,
        enable_thinking: bool = False,
    ) -> PromptSplit:
        kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
        }
        if tools:
            kwargs["tools"] = list(tools)
        if chat_template_kwargs:
            kwargs.update(dict(chat_template_kwargs))

        prompt = tokenizer.apply_chat_template(list(messages), **kwargs)
        full_ids = tokenizer.encode(prompt, add_special_tokens=False)

        if not tools:
            return PromptSplit(
                full_ids=full_ids,
                tool_prefix_ids=[],
                conversation_ids=full_ids,
                tool_prefix_len=0,
                profile=self.profile_name,
            )

        from prefix_cache import _resolve_chat_markers, find_prefix_boundary_markers

        markers = _resolve_chat_markers(tokenizer)
        boundary = find_prefix_boundary_markers(full_ids, markers)
        if boundary < 0:
            return PromptSplit(
                full_ids=full_ids,
                tool_prefix_ids=[],
                conversation_ids=full_ids,
                tool_prefix_len=0,
                profile=self.profile_name,
            )

        return PromptSplit(
            full_ids=full_ids,
            tool_prefix_ids=full_ids[:boundary],
            conversation_ids=full_ids[boundary:],
            tool_prefix_len=boundary,
            profile=self.profile_name,
        )

    def supports_pflash_on_conversation(self) -> bool:
        # Laguna tool format differs; enable when validated on hardware.
        return False
