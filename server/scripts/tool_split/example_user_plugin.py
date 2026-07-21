"""Example user plugin — copy to your ``--tool-split-plugin-dir`` and edit.

Usage:
  mkdir -p ~/.lucebox/tool_split_plugins
  cp example_user_plugin.py ~/.lucebox/tool_split_plugins/my_vendor.py
  # server: --tool-split on --tool-split-profile plugin:my_vendor \\
  #          --tool-split-plugin-dir ~/.lucebox/tool_split_plugins
"""
from __future__ import annotations

from typing import Any, Mapping, Sequence

from tool_split.base import PromptSplit, ToolSplitAdapter
from tool_split.registry import register_adapter


@register_adapter("my_vendor")
class MyVendorToolSplitAdapter(ToolSplitAdapter):
    """Template: replace detection + split logic for your model family."""

    @classmethod
    def detect(cls, *, arch: str, tokenizer_id: str) -> bool:
        return "my_vendor" in (arch or "").lower()

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
            "enable_thinking": enable_thinking,
        }
        if tools:
            kwargs["tools"] = list(tools)
        if chat_template_kwargs:
            for k, v in chat_template_kwargs.items():
                if k in ("enable_thinking", "add_generation_prompt", "tools"):
                    kwargs[k] = v
        prompt = tokenizer.apply_chat_template(list(messages), **kwargs)
        full_ids = tokenizer.encode(prompt, add_special_tokens=False)
        # TODO: locate tool-schema token boundary for your template.
        split_at = len(full_ids) // 2  # placeholder — do not ship as-is
        return PromptSplit(
            full_ids=full_ids,
            tool_prefix_ids=full_ids[:split_at],
            conversation_ids=full_ids[split_at:],
            tool_prefix_len=split_at,
            profile=self.profile_name,
        )
