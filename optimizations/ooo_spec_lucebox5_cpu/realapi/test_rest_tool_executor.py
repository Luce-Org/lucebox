#!/usr/bin/env python3
from __future__ import annotations

import io
import json
import sys
import unittest
from contextlib import redirect_stdout
from unittest import mock

import rest_tool_executor


class RestToolExecutorTest(unittest.TestCase):
    def invoke(self, raw_result: dict) -> tuple[int, dict]:
        request = {
            "protocol": rest_tool_executor.rest_tools.PROTOCOL,
            "call": {"name": "geocode_city", "arguments": {"city": "Rome"}},
            "cpu_affinity": [7],
        }
        output = io.StringIO()
        with (
            mock.patch.object(sys, "argv", ["rest_tool_executor.py", "--dflash-tool-spec-v1"]),
            mock.patch.object(sys, "stdin", io.StringIO(json.dumps(request) + "\n")),
            mock.patch.object(
                rest_tool_executor.rest_tools, "run_tool", return_value=raw_result
            ),
            mock.patch.object(
                rest_tool_executor.os, "sched_getaffinity", return_value={7},
                create=True,
            ),
            redirect_stdout(output),
        ):
            return rest_tool_executor.main(), json.loads(output.getvalue())

    def test_successful_tool_result_is_exposed(self) -> None:
        raw = {"ok": True, "value": {"latitude": 41.9}}
        status, envelope = self.invoke(raw)
        self.assertEqual(status, 0)
        self.assertEqual(envelope, {"ok": True, "result": raw})

    def test_failed_tool_result_rejects_speculative_hit(self) -> None:
        status, envelope = self.invoke(
            {"ok": False, "value": {"error": "network unavailable"}}
        )
        self.assertEqual(status, 0)
        self.assertFalse(envelope["ok"])
        self.assertNotIn("result", envelope)


if __name__ == "__main__":
    unittest.main()
