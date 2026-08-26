#!/usr/bin/env python3
"""Regression tests for backend-pair client output parsing."""

from __future__ import annotations

import importlib.util
import json
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "clients" / "summarize_backend_pair.py"
SPEC = importlib.util.spec_from_file_location("summarize_backend_pair", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"could not load {SCRIPT}")
SUMMARY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SUMMARY)


def event(payload: dict) -> str:
    return json.dumps(payload, separators=(",", ":"))


class OmpTextExtractionTests(unittest.TestCase):
    def test_assistant_text_is_extracted_without_echoed_user_text(self) -> None:
        output = "\n".join((
            event({
                "type": "message_end",
                "message": {
                    "role": "user",
                    "content": [{"type": "text", "text": "echoed user prompt"}],
                },
            }),
            event({
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "OK_DONE"}],
                    "stopReason": "stop",
                },
            }),
        ))

        self.assertEqual(SUMMARY.extract_generated_text(output), "OK_DONE")

    def test_assistant_text_is_extracted_from_mixed_tool_call_content(self) -> None:
        output = event({
            "type": "message_end",
            "message": {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "I will inspect the file."},
                    {"type": "toolCall", "name": "read", "arguments": {}},
                ],
                "stopReason": "toolUse",
            },
        })

        self.assertEqual(
            SUMMARY.extract_generated_text(output),
            "I will inspect the file.",
        )


class OmpToolCallTests(unittest.TestCase):
    def test_tool_call_only_assistant_does_not_echo_other_messages(self) -> None:
        output = "\n".join((
            event({
                "type": "message_end",
                "message": {
                    "role": "user",
                    "content": [{"type": "text", "text": "echoed user prompt"}],
                },
            }),
            event({
                "type": "message_end",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "toolCall", "name": "read", "arguments": {}}],
                    "stopReason": "toolUse",
                },
            }),
            event({
                "type": "message_end",
                "message": {
                    "role": "toolResult",
                    "content": [{"type": "text", "text": "echoed tool result"}],
                },
            }),
        ))

        self.assertEqual(SUMMARY.extract_generated_text(output), "")

    def test_tool_use_stop_without_call_is_not_success(self) -> None:
        output = event({
            "type": "message_end",
            "message": {
                "role": "assistant",
                "content": [],
                "stopReason": "toolUse",
            },
        })

        self.assertFalse(SUMMARY.omp_tool_call_ok(output))

    def test_tool_call_text_is_not_success(self) -> None:
        output = event({
            "type": "message_end",
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": '{"type":"toolCall"}'}],
                "stopReason": "stop",
            },
        })

        self.assertFalse(SUMMARY.omp_tool_call_ok(output))

    def test_tool_call_content_is_success(self) -> None:
        output = event({
            "type": "message_end",
            "message": {
                "role": "assistant",
                "content": [{"type": "toolCall", "name": "read", "arguments": {}}],
                "stopReason": "toolUse",
            },
        })

        self.assertTrue(SUMMARY.omp_tool_call_ok(output))

    def test_tool_execution_start_is_success(self) -> None:
        output = "\n".join((
            "OMP diagnostic",
            event({"type": "tool_execution_start", "toolName": "read"}),
        ))

        self.assertTrue(SUMMARY.omp_tool_call_ok(output))


if __name__ == "__main__":
    unittest.main()
