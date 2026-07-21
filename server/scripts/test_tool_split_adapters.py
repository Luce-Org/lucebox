"""Unit tests for tool_split registry and adapters (no GPU)."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from tool_split.base import PromptSplit, ToolSplitAdapter  # noqa: E402
from tool_split.config import ToolSplitConfig, config_from_env_and_args  # noqa: E402
from tool_split.registry import (  # noqa: E402
    builtin_adapter_names,
    get_adapter_class,
    register_adapter,
    resolve_adapter,
)
from tool_split.qwen3 import Qwen3ToolSplitAdapter  # noqa: E402


class _FakeTokenizer:
    """Qwen-shaped tokenizer stub for split tests.

    Renders ``<|im_start|>role ... <|im_end|>`` framing (like the real Qwen3
    chat template) and encodes the special markers + role words as single
    tokens so ``_tool_prefix_boundary`` can find the first user turn.
    """

    _SPECIALS = ["<|im_start|>", "<|im_end|>", "system", "user", "assistant"]

    def apply_chat_template(self, messages, **kwargs):
        parts = []
        first = True
        for m in messages:
            body = m.get("content", "") or ""
            if first and kwargs.get("tools"):
                names = ",".join(t["function"]["name"] for t in kwargs["tools"])
                body += f"<tools>{names}</tools>"
            parts.append(f"<|im_start|>{m['role']}\n{body}<|im_end|>\n")
            first = False
        if kwargs.get("add_generation_prompt", True):
            parts.append("<|im_start|>assistant\n")
        return "".join(parts)

    def encode(self, text, add_special_tokens=False):
        ids = []
        i = 0
        while i < len(text):
            for si, s in enumerate(self._SPECIALS):
                if text.startswith(s, i):
                    ids.append(1000 + si)
                    i += len(s)
                    break
            else:
                ids.append(ord(text[i]))
                i += 1
        return ids


class TestRegistry(unittest.TestCase):
    def test_builtins_registered(self):
        names = builtin_adapter_names()
        self.assertIn("qwen3", names)
        self.assertIn("laguna", names)

    def test_get_adapter_class(self):
        cls = get_adapter_class("qwen3")
        self.assertIs(cls, Qwen3ToolSplitAdapter)

    def test_user_plugin_register(self):
        @register_adapter("test_vendor")
        class _TestVendor(ToolSplitAdapter):
            @classmethod
            def detect(cls, *, arch, tokenizer_id):
                return False

            def split_prompt(self, *a, **k):
                raise NotImplementedError

        self.assertIn("test_vendor", builtin_adapter_names())


class TestQwen3Adapter(unittest.TestCase):
    def setUp(self):
        self.adapter = Qwen3ToolSplitAdapter()
        self.tok = _FakeTokenizer()

    def test_canonical_tools_sorted(self):
        tools = [
            {"type": "function", "function": {"name": "z", "description": "", "parameters": {}}},
            {"type": "function", "function": {"name": "a", "description": "", "parameters": {}}},
        ]
        canon = self.adapter.canonical_tools(tools)
        self.assertEqual([c["function"]["name"] for c in canon], ["a", "z"])

    def test_fingerprint_stable(self):
        tools = [{"type": "function", "function": {"name": "x", "description": "d", "parameters": {}}}]
        self.assertEqual(
            self.adapter.tools_fingerprint(tools),
            self.adapter.tools_fingerprint(tools),
        )

    def test_split_with_tools(self):
        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ]
        tools = [{"type": "function", "function": {"name": "read_file", "description": "", "parameters": {}}}]
        split = self.adapter.split_prompt(self.tok, messages, tools)
        self.assertIsInstance(split, PromptSplit)
        self.assertGreater(split.tool_prefix_len, 0)
        self.assertEqual(split.full_ids, split.tool_prefix_ids + split.conversation_ids)

    def test_split_no_tools(self):
        messages = [{"role": "user", "content": "hi"}]
        split = self.adapter.split_prompt(self.tok, messages, None)
        self.assertEqual(split.tool_prefix_len, 0)
        self.assertEqual(split.conversation_ids, split.full_ids)

    def test_tool_boundary_fallback_matches_marker_path(self):
        from unittest.mock import patch

        messages = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "hi"},
        ]
        tools = [{"type": "function", "function": {"name": "read_file", "description": "", "parameters": {}}}]
        with patch("prefix_cache.find_prefix_boundary_markers", return_value=-1):
            split = self.adapter.split_prompt(self.tok, messages, tools)
        im_start = self.tok.encode("<|im_start|>", add_special_tokens=False)[0]
        user_tok = self.tok.encode("user", add_special_tokens=False)[0]
        seq = (im_start, user_tok)
        idx = next(
            i for i in range(len(split.full_ids) - 1)
            if tuple(split.full_ids[i:i + 2]) == seq
        )
        self.assertEqual(split.tool_prefix_len, idx + 1)


class TestConfig(unittest.TestCase):
    def test_invalid_env_pinned_falls_back_with_cli_override(self):
        import os
        from unittest.mock import patch

        class _Args:
            tool_split_pinned_slots = 5

        with patch.dict(os.environ, {"DFLASH_TOOL_SPLIT_PINNED_SLOTS": "not-a-number"}):
            cfg = config_from_env_and_args(_Args())
        self.assertEqual(cfg.pinned_tool_slots, 5)

    def test_cli_plugin_dir_expands_user(self):
        import os
        from unittest.mock import patch

        class _Args:
            tool_split_plugin_dir = Path("~/tool_plugins")

        with patch.dict(os.environ, {}, clear=False):
            cfg = config_from_env_and_args(_Args())
        self.assertTrue(str(cfg.plugin_dir).endswith("tool_plugins"))
        self.assertNotIn("~", str(cfg.plugin_dir))

    def test_resolve_auto_qwen(self):
        cfg = ToolSplitConfig(enabled=True, profile="auto")
        adapter = resolve_adapter(cfg, arch="qwen36", tokenizer_id="Qwen/Qwen3.6-27B")
        self.assertIsNotNone(adapter)
        self.assertEqual(adapter.profile_name, "qwen3")


class TestToolSlotCache(unittest.TestCase):
    def test_reserve_reuses_evicted_slot_within_range(self):
        from tool_split.orchestrator import ToolSlotCache

        cache = ToolSlotCache(pinned_slots=2, slot_base=3)
        s0 = cache.reserve("a")
        cache.confirm("a", s0)
        s1 = cache.reserve("b")
        cache.confirm("b", s1)
        self.assertEqual({s0, s1}, {3, 4})
        # Reserve "c" — defers eviction of "a" until confirm.
        s2 = cache.reserve("c")
        self.assertEqual(s2, s0)
        self.assertEqual(cache.lookup("a"), s0)
        self.assertIsNone(cache.lookup("c"))
        cache.confirm("c", s2)
        self.assertIsNone(cache.lookup("a"))
        self.assertEqual(cache.lookup("c"), s0)
        self.assertEqual(cache.lookup("b"), s1)
        self.assertTrue(all(3 <= s < 5 for s in cache._confirmed.values()))

    def test_release_reservation_frees_slot(self):
        from tool_split.orchestrator import ToolSlotCache

        cache = ToolSlotCache(pinned_slots=2, slot_base=6)
        slot = cache.reserve("fp")
        cache.release_reservation("fp", slot)
        self.assertIsNone(cache.lookup("fp"))
        # Freed slot is reusable.
        self.assertEqual(cache.reserve("other"), slot)

    def test_release_reservation_restores_evicted_anchor(self):
        from tool_split.orchestrator import ToolSlotCache

        cache = ToolSlotCache(pinned_slots=2, slot_base=3)
        s0 = cache.reserve("a")
        cache.confirm("a", s0)
        s1 = cache.reserve("b")
        cache.confirm("b", s1)
        s2 = cache.reserve("c")
        self.assertEqual(cache.lookup("a"), s0)
        cache.release_reservation("c", s2)
        self.assertEqual(cache.lookup("a"), s0)
        self.assertIsNone(cache.lookup("c"))


class TestCommitPendingToolSnap(unittest.IsolatedAsyncioTestCase):
    async def test_releases_reservation_on_snapshot_failure(self):
        import asyncio

        from tool_split.daemon_bridge import commit_pending_tool_snap
        from tool_split.orchestrator import ToolSlotCache

        class _Orch:
            def __init__(self):
                self.tool_slots = ToolSlotCache(pinned_slots=2, slot_base=3)

        orch = _Orch()
        slot = orch.tool_slots.reserve("fp")

        async def _fail_reply(prefix, timeout=30.0):
            raise asyncio.TimeoutError("daemon timeout")

        class _Stdin:
            def write(self, data):
                return None

            def flush(self):
                return None

        await commit_pending_tool_snap(
            orchestrator=orch,
            daemon_stdin=_Stdin(),
            await_reply=_fail_reply,
            fingerprint="fp",
            slot=slot,
            kv_end=10,
        )
        self.assertIsNone(orch.tool_slots.lookup("fp"))


class TestOrchestratorDaemonCmd(unittest.TestCase):
    def setUp(self):
        from tool_split.config import ToolSplitConfig
        from tool_split.orchestrator import ToolSplitOrchestrator

        self.orch = ToolSplitOrchestrator(
            adapter=Qwen3ToolSplitAdapter(),
            config=ToolSplitConfig(enabled=True, pinned_tool_slots=2),
        )

    def test_restore_chain_tool_only(self):
        plan = self.orch.build_plan(
            split=None,
            tools_fingerprint="abc",
            prompt_bin=Path("/tmp/p.bin"),
            prompt_len=100,
            tool_slot_hit=6,
        )
        cmd = self.orch.format_daemon_command(plan, 32)
        self.assertIn("RESTORE_CHAIN -1 6 /tmp/p.bin 32", cmd)

    def test_restore_chain_tool_and_conv(self):
        plan = self.orch.build_plan(
            split=None,
            tools_fingerprint="abc",
            prompt_bin=Path("/tmp/p.bin"),
            prompt_len=100,
            tool_slot_hit=6,
            conv_hit=(2, 80),
        )
        cmd = self.orch.format_daemon_command(plan, 16)
        self.assertIn("RESTORE_CHAIN 2 6 /tmp/p.bin 16", cmd)


if __name__ == "__main__":
    unittest.main()
