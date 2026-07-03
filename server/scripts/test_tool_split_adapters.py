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
    """Minimal tokenizer stub for split tests."""

    def apply_chat_template(self, messages, **kwargs):
        parts = []
        if kwargs.get("tools"):
            parts.append("<tools>")
            for t in kwargs["tools"]:
                parts.append(t["function"]["name"])
            parts.append("</tools>")
        for m in messages:
            parts.append(f"<{m['role']}>{m.get('content', '')}")
        if kwargs.get("add_generation_prompt", True):
            parts.append("<assistant>")
        return "".join(parts)

    def encode(self, text, add_special_tokens=False):
        # One token per character for easy slicing tests.
        return [ord(c) for c in text]


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


class TestConfig(unittest.TestCase):
    def test_resolve_auto_qwen(self):
        cfg = ToolSplitConfig(enabled=True, profile="auto")
        adapter = resolve_adapter(cfg, arch="qwen36", tokenizer_id="Qwen/Qwen3.6-27B")
        self.assertIsNotNone(adapter)
        self.assertEqual(adapter.profile_name, "qwen3")


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
