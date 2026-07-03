"""Qwen3.x chat-template tool / conversation splitter."""
from __future__ import annotations

import re
from typing import Any, Mapping, Sequence

from tool_split.base import PromptSplit, ToolSplitAdapter
from tool_split.registry import register_adapter

_QWEN_ARCHES = frozenset({"qwen35", "qwen36", "qwen3"})
_QWEN_TOKENIZER_HINTS = ("qwen3", "qwen/qwen3")


@register_adapter("qwen3")
class Qwen3ToolSplitAdapter(ToolSplitAdapter):
    """Qwen ``<|im_start|>`` templates with OpenAI ``tools=`` injection."""

    @classmethod
    def detect(cls, *, arch: str, tokenizer_id: str) -> bool:
        a = (arch or "").lower()
        t = (tokenizer_id or "").lower()
        if a in _QWEN_ARCHES:
            return True
        return any(h in t for h in _QWEN_TOKENIZER_HINTS)

    def split_prompt(
        self,
        tokenizer: Any,
        messages: Sequence[Mapping[str, Any]],
        tools: Sequence[Mapping[str, Any]] | None,
        *,
        chat_template_kwargs: Mapping[str, Any] | None = None,
        enable_thinking: bool = False,
    ) -> PromptSplit:
        if not tools:
            full_ids = self._encode_full(
                tokenizer, messages, None, chat_template_kwargs, enable_thinking,
            )
            return PromptSplit(
                full_ids=full_ids,
                tool_prefix_ids=[],
                conversation_ids=full_ids,
                tool_prefix_len=0,
                profile=self.profile_name,
            )

        canon = self.canonical_tools(tools)
        full_ids = self._encode_full(
            tokenizer, messages, canon, chat_template_kwargs, enable_thinking,
        )

        # Stable prefix: system messages (if any) + tool definitions, no user turn.
        system_msgs = [m for m in messages if m.get("role") == "system"]
        tool_prefix_ids: list[int] = []
        conv_ids: list[int]

        if system_msgs:
            tool_prefix_ids = self._encode_full(
                tokenizer,
                system_msgs,
                canon,
                chat_template_kwargs,
                enable_thinking=False,
                add_generation_prompt=False,
            )
            if tool_prefix_ids and full_ids[: len(tool_prefix_ids)] == tool_prefix_ids:
                conv_ids = full_ids[len(tool_prefix_ids) :]
            else:
                boundary = self._tool_prefix_boundary(tokenizer, full_ids)
                if boundary < 0:
                    tool_prefix_ids = []
                    conv_ids = full_ids
                else:
                    tool_prefix_ids = full_ids[:boundary]
                    conv_ids = full_ids[boundary:]
        else:
            # No explicit system message — tools inject into the default system
            # block. Never call apply_chat_template([]); slice at first user turn.
            boundary = self._tool_prefix_boundary(tokenizer, full_ids)
            if boundary < 0:
                tool_prefix_ids = []
                conv_ids = full_ids
            else:
                tool_prefix_ids = full_ids[:boundary]
                conv_ids = full_ids[boundary:]

        return PromptSplit(
            full_ids=full_ids,
            tool_prefix_ids=tool_prefix_ids,
            conversation_ids=conv_ids,
            tool_prefix_len=len(tool_prefix_ids),
            profile=self.profile_name,
        )

    def _encode_full(
        self,
        tokenizer: Any,
        messages: Sequence[Mapping[str, Any]],
        tools: list[dict[str, Any]] | None,
        chat_template_kwargs: Mapping[str, Any] | None,
        enable_thinking: bool,
        *,
        add_generation_prompt: bool = True,
    ) -> list[int]:
        kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": add_generation_prompt,
            "enable_thinking": enable_thinking,
        }
        if chat_template_kwargs:
            for k, v in chat_template_kwargs.items():
                if k in ("enable_thinking", "add_generation_prompt", "tools"):
                    kwargs[k] = v
        if tools:
            kwargs["tools"] = tools
            kwargs["enable_thinking"] = False
        prompt = tokenizer.apply_chat_template(list(messages), **kwargs)
        if add_generation_prompt and not enable_thinking:
            # Match server_tools: detect thinking prefill at end of template.
            pass
        return tokenizer.encode(prompt, add_special_tokens=False)

    def thinking_prefill_at_end(self, tokenizer: Any, full_ids: list[int]) -> bool:
        prompt = tokenizer.decode(full_ids, skip_special_tokens=False)
        return bool(re.search(r"<think>\s*$", prompt))

    def _tool_prefix_boundary(self, tokenizer: Any, full_ids: list[int]) -> int:
        """Index after tool/system header — right before first user turn."""
        from prefix_cache import (
            _find_first_seq,
            _resolve_chat_markers,
            find_prefix_boundary_markers,
        )

        markers = _resolve_chat_markers(tokenizer)
        boundary = find_prefix_boundary_markers(full_ids, markers)
        if boundary >= 0:
            return boundary

        qs = tokenizer.encode("<|im_start|>", add_special_tokens=False)
        user_t = tokenizer.encode("user", add_special_tokens=False)
        if len(qs) == 1 and len(user_t) == 1:
            user_seq = (qs[0], user_t[0])
            idx = _find_first_seq(full_ids, user_seq)
            if idx >= 0:
                return idx + len(user_seq)
        return -1
