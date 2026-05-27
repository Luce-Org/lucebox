from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Iterable, Sequence


@dataclass
class _ToolMemoryBlock:
    raw_text: str
    size_bytes: int
    refs: int = 0


class ToolMemory:
    """Exact assistant-text memory for tool-calling turns.

    The server receives assistant tool calls back as structured JSON on later
    turns. Re-rendering those objects can change key order, spacing, or wrapper
    shape, which changes prompt tokenization and breaks prefix/KV reuse. This
    store remembers the exact assistant text that originally produced a set of
    tool-call IDs so replay can inject the original text verbatim.
    """

    def __init__(self, *, max_entries: int = 50_000, max_bytes: int = 64 * 1024 * 1024):
        self.max_entries = max(0, int(max_entries))
        self.max_bytes = max(0, int(max_bytes))
        self.by_id: dict[str, _ToolMemoryBlock] = {}
        self.by_block: dict[str, _ToolMemoryBlock] = {}
        self._lru: OrderedDict[str, None] = OrderedDict()
        self.total_bytes = 0

    @property
    def disabled(self) -> bool:
        return self.max_entries == 0 or self.max_bytes == 0

    def remember(self, call_ids: Iterable[str], raw_text: str) -> None:
        if self.disabled or not raw_text:
            return
        unique_ids = []
        seen: set[str] = set()
        for call_id in call_ids:
            if not isinstance(call_id, str) or not call_id or call_id in seen:
                continue
            seen.add(call_id)
            unique_ids.append(call_id)
        if not unique_ids:
            return

        block = self.by_block.get(raw_text)
        if block is None:
            block = _ToolMemoryBlock(
                raw_text=raw_text,
                size_bytes=len(raw_text.encode("utf-8")),
            )
            self.by_block[raw_text] = block
            self.total_bytes += block.size_bytes

        for call_id in unique_ids:
            current = self.by_id.get(call_id)
            if current is block:
                self._touch(call_id)
                continue
            if current is not None:
                self._drop_entry(call_id, current)
            self.by_id[call_id] = block
            block.refs += 1
            self._touch(call_id)

        self._prune()

    def lookup_message(self, tool_calls: Sequence[Any]) -> str | None:
        raw_text: str | None = None
        touched: list[str] = []
        for item in tool_calls:
            call_id = self._extract_call_id(item)
            if not call_id:
                return None
            block = self.by_id.get(call_id)
            if block is None:
                return None
            touched.append(call_id)
            if raw_text is None:
                raw_text = block.raw_text
            elif raw_text != block.raw_text:
                return None
        if raw_text is None:
            return None
        for call_id in touched:
            self._touch(call_id)
        return raw_text

    def _touch(self, call_id: str) -> None:
        self._lru[call_id] = None
        self._lru.move_to_end(call_id)

    def _prune(self) -> None:
        while self.by_id and (
            (self.max_entries > 0 and len(self.by_id) > self.max_entries)
            or (self.max_bytes > 0 and self.total_bytes > self.max_bytes)
        ):
            oldest_id, _ = self._lru.popitem(last=False)
            block = self.by_id.get(oldest_id)
            if block is not None:
                self._drop_entry(oldest_id, block)

    def _drop_entry(self, call_id: str, block: _ToolMemoryBlock) -> None:
        self.by_id.pop(call_id, None)
        self._lru.pop(call_id, None)
        if block.refs > 0:
            block.refs -= 1
        if block.refs == 0:
            self.by_block.pop(block.raw_text, None)
            self.total_bytes -= block.size_bytes
            if self.total_bytes < 0:
                self.total_bytes = 0

    @staticmethod
    def _extract_call_id(item: Any) -> str | None:
        if isinstance(item, dict):
            call_id = item.get("id")
            return call_id if isinstance(call_id, str) and call_id else None
        call_id = getattr(item, "id", None)
        return call_id if isinstance(call_id, str) and call_id else None
