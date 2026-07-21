"""Orchestrate split tool KV cache + conversation PFlash."""
from __future__ import annotations

import json
import os
import struct
import tempfile
from collections import OrderedDict
from pathlib import Path
from typing import Any, Mapping, Sequence

from tool_split.base import PromptSplit, ToolSplitAdapter, ToolSplitPlan, ToolRequestContext
from tool_split.config import ToolSplitConfig


def _ids_to_bin(ids: list[int]) -> Path:
    fd, path = tempfile.mkstemp(suffix=".bin")
    with os.fdopen(fd, "wb") as f:
        for t in ids:
            f.write(struct.pack("<i", int(t)))
    return Path(path)


def _tool_call_args_to_obj(args: Any) -> Any:
    # Qwen chat templates iterate arguments with `| items`, which requires a
    # mapping. OpenAI-shaped requests carry arguments as a JSON string.
    if isinstance(args, str):
        try:
            return json.loads(args)
        except (json.JSONDecodeError, ValueError):
            return {"_raw": args}
    return args


def _messages_for_adapter(messages: Sequence[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in messages:
        if isinstance(m, dict):
            d = dict(m)
            if d.get("tool_calls"):
                d["tool_calls"] = [
                    {
                        **tc,
                        "function": {
                            **(tc.get("function") or {}),
                            "arguments": _tool_call_args_to_obj(
                                (tc.get("function") or {}).get("arguments")),
                        },
                    }
                    for tc in d["tool_calls"]
                ]
            out.append(d)
            continue
        d: dict[str, Any] = {"role": m.role}
        if m.content is not None:
            d["content"] = m.content
        if getattr(m, "name", None) is not None:
            d["name"] = m.name
        if getattr(m, "tool_call_id", None) is not None:
            d["tool_call_id"] = m.tool_call_id
        if getattr(m, "tool_calls", None):
            d["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {
                        "name": tc.function.name,
                        "arguments": _tool_call_args_to_obj(tc.function.arguments),
                    },
                }
                for tc in m.tool_calls
            ]
        out.append(d)
    return out


def _tools_for_adapter(tools: Sequence[Any] | None) -> list[dict[str, Any]] | None:
    if not tools:
        return None
    return [t.model_dump() if hasattr(t, "model_dump") else dict(t) for t in tools]


class ToolSlotCache:
    """LRU index: tools fingerprint → daemon thin-snapshot slot (pinned range).

    Slot IDs stay in ``[slot_base, slot_base + pinned_slots)``. Eviction reuses
    the LRU entry's slot so RESTORE_CHAIN never references out-of-range IDs.

    Eviction is deferred from ``reserve()`` to ``confirm()`` so a failed
    ``SNAPSHOT_THIN`` can call ``release_reservation()`` without losing the
    previous anchor (same pattern as ``PrefixCache.prepare_inline_snap``).
    """

    def __init__(self, *, pinned_slots: int, slot_base: int = 0):
        self.pinned_slots = max(0, pinned_slots)
        self.slot_base = slot_base
        self._confirmed: OrderedDict[str, int] = OrderedDict()
        self._pending: dict[str, int] = {}
        self._pending_evict: tuple[str, int] | None = None

    def lookup(self, fingerprint: str) -> int | None:
        if fingerprint in self._pending:
            return None
        if fingerprint in self._confirmed:
            self._confirmed.move_to_end(fingerprint)
            return self._confirmed[fingerprint]
        return None

    def reserve(self, fingerprint: str) -> int:
        if self.pinned_slots <= 0:
            raise ValueError("ToolSlotCache.reserve requires pinned_slots > 0")
        if fingerprint in self._confirmed:
            self._confirmed.move_to_end(fingerprint)
            return self._confirmed[fingerprint]
        if fingerprint in self._pending:
            return self._pending[fingerprint]
        if len(self._confirmed) >= self.pinned_slots:
            # Peek LRU without evicting — eviction happens in confirm().
            old_fp, slot = next(iter(self._confirmed.items()))
            self._pending_evict = (old_fp, slot)
        else:
            used = set(self._confirmed.values()) | set(self._pending.values())
            slot = next(
                s for s in range(self.slot_base, self.slot_base + self.pinned_slots)
                if s not in used
            )
            self._pending_evict = None
        self._pending[fingerprint] = slot
        return slot

    def confirm(self, fingerprint: str, slot: int) -> None:
        if self._pending_evict is not None:
            evict_fp, _ = self._pending_evict
            self._confirmed.pop(evict_fp, None)
            self._pending_evict = None
        self._pending.pop(fingerprint, None)
        self._confirmed[fingerprint] = slot
        self._confirmed.move_to_end(fingerprint)

    def release_reservation(self, fingerprint: str, slot: int) -> None:
        """Drop a fingerprint reserved before SNAPSHOT_THIN succeeded."""
        if self._pending.get(fingerprint) == slot:
            del self._pending[fingerprint]
        self._pending_evict = None


class ToolSplitOrchestrator:
    """High-level API used by ``server_tools.py``."""

    def __init__(
        self,
        *,
        adapter: ToolSplitAdapter,
        config: ToolSplitConfig,
        tool_slot_cache: ToolSlotCache | None = None,
    ):
        self.adapter = adapter
        self.config = config
        self.tool_slots = tool_slot_cache or ToolSlotCache(
            pinned_slots=config.pinned_tool_slots,
        )

    @property
    def profile(self) -> str:
        return self.adapter.profile_name

    def active_for_request(self, tools: Sequence[Any] | None) -> bool:
        return bool(tools)

    def split_request(
        self,
        tokenizer: Any,
        messages: Sequence[Any],
        tools: Sequence[Any] | None,
        *,
        chat_template_kwargs: Mapping[str, Any] | None = None,
    ) -> PromptSplit:
        return self.adapter.split_prompt(
            tokenizer,
            _messages_for_adapter(messages),
            _tools_for_adapter(tools),
            chat_template_kwargs=chat_template_kwargs or {},
            enable_thinking=False,
        )

    def tools_fingerprint(self, tools: Sequence[Any] | None) -> str | None:
        raw = _tools_for_adapter(tools)
        if not raw:
            return None
        return self.adapter.tools_fingerprint(raw)

    def write_prompt_bin(self, ids: list[int]) -> Path:
        return _ids_to_bin(ids)

    def conversation_compressible(self, split: PromptSplit) -> bool:
        return (
            self.config.compress_conversation
            and self.adapter.supports_pflash_on_conversation()
            and split.conversation_len > 0
        )

    def prepare_request_context(
        self,
        tokenizer: Any,
        messages: Sequence[Any],
        tools: Sequence[Any] | None,
        *,
        chat_template_kwargs: Mapping[str, Any] | None = None,
    ) -> ToolRequestContext | None:
        if not self.active_for_request(tools):
            return None
        split = self.split_request(
            tokenizer, messages, tools,
            chat_template_kwargs=chat_template_kwargs,
        )
        fp = self.tools_fingerprint(tools)
        if not fp or split.tool_prefix_len <= 0:
            return ToolRequestContext(split=split, fingerprint=fp)

        tool_slot_hit = self.tool_slots.lookup(fp)
        pending: tuple[int, int] | None = None
        if tool_slot_hit is None and self.config.pinned_tool_slots > 0:
            slot = self.tool_slots.reserve(fp)
            pending = (slot, split.tool_prefix_len)

        return ToolRequestContext(
            split=split,
            fingerprint=fp,
            tool_slot_hit=tool_slot_hit,
            pending_tool_snap=pending,
        )

    def build_plan(
        self,
        *,
        split: PromptSplit | None,
        tools_fingerprint: str | None,
        prompt_bin: Path,
        prompt_len: int,
        tool_slot_hit: int | None = None,
        conv_hit: tuple[int, int] | None = None,
        snap_prep: tuple[int, int] | None = None,
        compression_fired: bool = False,
        started_in_thinking: bool = False,
        pending_tool_snap: tuple[int, int] | None = None,
    ) -> ToolSplitPlan:
        use_chain = tool_slot_hit is not None
        thick = conv_hit[0] if conv_hit else None
        if use_chain and thick is None:
            thick_arg = -1
        elif use_chain:
            thick_arg = thick
        else:
            thick_arg = None

        thin_ids = [tool_slot_hit] if tool_slot_hit is not None else []

        return ToolSplitPlan(
            prompt_bin_path=str(prompt_bin),
            prompt_token_count=prompt_len,
            tool_slot=tool_slot_hit if use_chain else None,
            conv_restore_slot=thick_arg if use_chain else (conv_hit[0] if conv_hit else None),
            conv_restore_prefix_len=conv_hit[1] if conv_hit else 0,
            use_restore_chain=use_chain,
            thin_slot_ids=thin_ids,
            inline_snap=snap_prep,
            compression_fired=compression_fired,
            started_in_thinking=started_in_thinking,
            tools_fingerprint=tools_fingerprint,
            pending_tool_snap=pending_tool_snap,
            tool_prefix_len=split.tool_prefix_len if split else 0,
        )

    def format_daemon_command(
        self,
        plan: ToolSplitPlan,
        gen_len: int,
    ) -> str:
        """Format stdin line for test_dflash (RESTORE / RESTORE_CHAIN)."""
        path = plan.prompt_bin_path
        if plan.use_restore_chain and plan.thin_slot_ids:
            thick = plan.conv_restore_slot if plan.conv_restore_slot is not None else -1
            thin = ",".join(str(s) for s in plan.thin_slot_ids)
            line = f"RESTORE_CHAIN {thick} {thin} {path} {gen_len}"
        elif plan.conv_restore_slot is not None and not plan.use_restore_chain:
            line = f"RESTORE {plan.conv_restore_slot} {path} {gen_len}"
        else:
            line = f"{path} {gen_len}"
        if plan.inline_snap:
            slot, cut = plan.inline_snap
            line += f" snap={cut}:{slot}"
        return line + "\n"
