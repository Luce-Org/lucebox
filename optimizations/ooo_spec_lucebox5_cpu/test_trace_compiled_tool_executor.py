from __future__ import annotations

import unittest

from benchmark_trace_compiled_workflows import mine_pattern, simulated_tool_result
from bfcl_replay_tool_executor import execute as execute_bfcl
from test_benchmark_trace_compiled_workflows import workflow_trace
from trace_compiled_tool_executor import execute_macro, resolve_items


class TraceCompiledToolExecutorTest(unittest.TestCase):
    def setUp(self) -> None:
        self.pattern = mine_pattern(
            [
                workflow_trace("first@example.test", "Rome"),
                workflow_trace("second@example.test", "Milan"),
            ]
        )
        self.items = [
            {"customer_email": "alice@example.test", "destination": "Turin"},
            {"customer_email": "bob@example.test", "destination": "Naples"},
        ]
        self.workflow_ref = "workflow_alpha"
        self.registry = {
            "schema_version": 1,
            "pattern_fingerprint": self.pattern.fingerprint,
            "workflows": {
                self.workflow_ref: {
                    "pattern_fingerprint": self.pattern.fingerprint,
                    "items": self.items,
                }
            },
        }

    @staticmethod
    def fake_leaf(request: dict) -> dict:
        return {"ok": True, "result": simulated_tool_result(request["call"])}

    def test_executes_every_branch_and_preserves_order(self) -> None:
        call = {
            "name": self.pattern.macro_name,
            "arguments": {"workflow_ref": self.workflow_ref},
        }
        request = {"call": call}
        envelope = execute_macro(
            request, self.pattern, self.fake_leaf, self.registry
        )
        result = envelope["result"]
        self.assertEqual(result["call_count"], 10)
        self.assertEqual([branch["root"] for branch in result["branches"]], self.items)
        self.assertEqual(
            [len(branch["steps"]) for branch in result["branches"]], [5, 5]
        )
        self.assertFalse(result["side_effects"])

    def test_rejects_unknown_or_missing_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "fields"):
            resolve_items(
                {"workflow_ref": self.workflow_ref},
                self.pattern,
                {
                    **self.registry,
                    "workflows": {
                        self.workflow_ref: {
                            "pattern_fingerprint": self.pattern.fingerprint,
                            "items": [{**self.items[0], "undeclared": "value"}],
                        }
                    },
                },
            )
        with self.assertRaisesRegex(ValueError, "workflow_ref"):
            resolve_items({"items": self.items}, self.pattern, self.registry)

    def test_rejects_non_object_leaf_response(self) -> None:
        request = {
            "call": {
                "name": self.pattern.macro_name,
                "arguments": {"workflow_ref": self.workflow_ref},
            }
        }
        with self.assertRaisesRegex(RuntimeError, "invalid result"):
            execute_macro(
                request,
                self.pattern,
                lambda _request: None,  # type: ignore[arg-type,return-value]
                self.registry,
            )

    def test_leaf_rejects_explicit_null_affinity(self) -> None:
        with self.assertRaisesRegex(ValueError, "cpu_affinity"):
            execute_bfcl(
                {
                    "protocol": "dflash.tool-speculation.v1",
                    "cpu_affinity": None,
                    "call": {"name": "lookup", "arguments": {}},
                }
            )


if __name__ == "__main__":
    unittest.main()
