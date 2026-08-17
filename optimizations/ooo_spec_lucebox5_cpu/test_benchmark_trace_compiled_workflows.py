from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from benchmark_trace_compiled_workflows import (
    alphabetic_identifier,
    compact_arm,
    final_answer_correct,
    load_training_traces,
    load_partial_pairs,
    make_task,
    mine_pattern,
    model_observation,
    parse_request_customers,
    post_final,
    production_checks,
    simulated_tool_result,
    stage_batch_tool,
    stage_batched_messages,
    stage_reference,
    workflow_reference,
)


def workflow_trace(email: str, destination: str) -> dict:
    root = {"customer_email": email, "destination": destination}
    calls = []
    results = []

    def add(name: str, arguments: dict) -> None:
        call = {"name": name, "arguments": arguments}
        calls.append(call)
        results.append(simulated_tool_result(call))

    add("resolve_customer", {"customer_email": email})
    add("list_open_orders", {"customer_ref": results[-1]["call_ref"]})
    add("get_order_details", {"orders_ref": results[-1]["call_ref"]})
    add(
        "calculate_shipping",
        {"order_ref": results[-1]["call_ref"], "destination": destination},
    )
    add("prepare_customer_summary", {"shipping_ref": results[-1]["call_ref"]})
    return {"root": root, "calls": calls, "results": results}


class TraceCompiledWorkflowBenchmarkTest(unittest.TestCase):
    def setUp(self) -> None:
        self.pattern = mine_pattern(
            [
                workflow_trace("first@example.test", "Rome"),
                workflow_trace("second@example.test", "Milan"),
            ]
        )

    def test_mines_control_flow_and_late_bound_arguments(self) -> None:
        self.assertEqual(self.pattern.training_traces, 2)
        self.assertEqual(
            [step.tool for step in self.pattern.steps],
            [
                "resolve_customer",
                "list_open_orders",
                "get_order_details",
                "calculate_shipping",
                "prepare_customer_summary",
            ],
        )
        shipping_bindings = dict(self.pattern.steps[3].arguments)
        self.assertEqual(shipping_bindings["order_ref"].source, "previous_result")
        self.assertEqual(shipping_bindings["order_ref"].key, "call_ref")
        self.assertEqual(shipping_bindings["destination"].source, "root")

    def test_loads_compact_training_trace_file(self) -> None:
        traces = [
            workflow_trace("first@example.test", "Rome"),
            workflow_trace("second@example.test", "Milan"),
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "traces.json"
            path.write_text(json.dumps({"traces": traces}), encoding="utf-8")
            loaded = load_training_traces(path, required_steps=5)
        self.assertEqual(loaded, traces)

    def test_compact_arm_keeps_evidence_but_drops_full_graph(self) -> None:
        arm = {
            "task_ms": 12.0,
            "underlying_calls": ["call"],
            "tool_results": ["digest"],
            "graph": {"large": "payload"},
            "call_turns": [{"large": "payload"}],
            "final": {
                "content": "done",
                "content_sha256": "hash",
                "completion_tokens": 1,
                "unused": "payload",
            },
        }
        compact = compact_arm(arm)
        self.assertEqual(compact["underlying_calls_count"], 1)
        self.assertEqual(
            compact["underlying_calls_sha256"],
            "4f2a91df1674ac67599f9835f2d43b0ca94e1e769f6a666ce448ae07ac1d94f7",
        )
        self.assertEqual(compact["final"]["content_sha256"], "hash")
        self.assertNotIn("underlying_calls", compact)
        self.assertNotIn("content", compact["final"])
        self.assertNotIn("graph", compact)
        self.assertNotIn("call_turns", compact)
        self.assertNotIn("unused", compact["final"])

    def test_pattern_expands_unseen_request_without_literals(self) -> None:
        root = {"customer_email": "new@example.test", "destination": "Turin"}
        calls = self.pattern.simulate(root)
        self.assertEqual(calls[0]["arguments"], {"customer_email": root["customer_email"]})
        self.assertEqual(
            calls[1]["arguments"], {"customer_ref": simulated_tool_result(calls[0])["call_ref"]}
        )
        self.assertEqual(calls[3]["arguments"]["destination"], "Turin")

    def test_compiler_rejects_side_effecting_trace(self) -> None:
        traces = [
            workflow_trace("first@example.test", "Rome"),
            workflow_trace("second@example.test", "Milan"),
        ]
        traces[1]["results"][2]["side_effects"] = True
        with self.assertRaisesRegex(ValueError, "side-effect-free"):
            mine_pattern(traces)

    def test_compiler_rejects_literal_argument(self) -> None:
        traces = [
            workflow_trace("first@example.test", "Rome"),
            workflow_trace("second@example.test", "Milan"),
        ]
        for trace in traces:
            trace["calls"][0]["arguments"]["constant"] = "not-in-request"
            trace["results"][0] = simulated_tool_result(trace["calls"][0])
        with self.assertRaisesRegex(ValueError, "literal"):
            mine_pattern(traces)

    def test_macro_schema_is_closed_and_typed(self) -> None:
        tool = self.pattern.macro_tool(4)["function"]
        parameters = tool["parameters"]
        item = parameters["properties"]["customers"]["items"]
        self.assertFalse(parameters["additionalProperties"])
        self.assertFalse(item["additionalProperties"])
        self.assertEqual(set(item["required"]), set(self.pattern.root_fields))
        self.assertEqual(parameters["properties"]["customers"]["maxItems"], 4)

    def test_bound_macro_uses_a_short_single_value_reference(self) -> None:
        task = make_task(4, 4, self.pattern)
        workflow_ref = workflow_reference(task, self.pattern)
        parameters = self.pattern.macro_tool(4, workflow_ref)["function"]["parameters"]
        self.assertEqual(set(parameters["properties"]), {"workflow_ref"})
        self.assertEqual(
            parameters["properties"]["workflow_ref"]["enum"], [workflow_ref]
        )
        self.assertEqual(workflow_ref, "workflow_taske")

    def test_generated_branches_do_not_collapse_on_short_refs(self) -> None:
        task = make_task(3, 4, self.pattern)
        refs_by_stage = list(
            zip(
                *[
                    [simulated_tool_result(call)["call_ref"] for call in self.pattern.simulate(root)]
                    for root in task["items"]
                ],
                strict=True,
            )
        )
        self.assertTrue(all(len(set(refs)) == 4 for refs in refs_by_stage))

    def test_generated_identifiers_avoid_ambiguous_digits(self) -> None:
        self.assertEqual(alphabetic_identifier(0), "taska")
        self.assertEqual(alphabetic_identifier(26), "taskaa")
        task = make_task(50, 2, self.pattern)
        self.assertTrue(
            all(not any(character.isdigit() for character in item["customer_email"])
                for item in task["items"])
        )

    def test_event_extractor_recovers_macro_arguments_without_a_model(self) -> None:
        content = "Customers: a@example.test to Rome; b@example.test to Milan."
        self.assertEqual(
            parse_request_customers(content),
            [
                {"customer_email": "a@example.test", "destination": "Rome"},
                {"customer_email": "b@example.test", "destination": "Milan"},
            ],
        )

    def test_stage_batch_schema_preserves_every_call(self) -> None:
        task = make_task(2, 3, self.pattern)
        tool = stage_batch_tool(self.pattern, 3, 4)["function"]
        calls = tool["parameters"]["properties"]["calls"]
        self.assertEqual(tool["name"], "batch_calculate_shipping")
        self.assertEqual(calls["maxItems"], 4)
        self.assertEqual(
            set(calls["items"]["required"]), {"order_ref", "destination"}
        )
        self.assertIn(
            "exactly one currently-ready batch tool",
            stage_batched_messages(task, self.pattern)[0]["content"],
        )

    def test_bound_stage_uses_the_request_scoped_reference(self) -> None:
        task = make_task(2, 4, self.pattern)
        stage_ref = stage_reference(task, self.pattern, 2)
        parameters = stage_batch_tool(
            self.pattern, 2, 4, stage_ref
        )["function"]["parameters"]
        self.assertEqual(set(parameters["properties"]), {"stage_ref"})
        self.assertEqual(parameters["properties"]["stage_ref"]["enum"], [stage_ref])
        self.assertEqual(stage_ref, "workflow_taskc_stage_three")

    def test_parses_multiple_native_tool_calls(self) -> None:
        response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "resolve_customer",
                                    "arguments": '{"customer_email":"a@example.test"}',
                                },
                            },
                            {
                                "id": "call_2",
                                "type": "function",
                                "function": {
                                    "name": "resolve_customer",
                                    "arguments": {"customer_email": "b@example.test"},
                                },
                            },
                        ],
                    }
                }
            ],
            "usage": {"timings": {}},
        }
        observed = model_observation(response, 12.0)
        self.assertEqual(len(observed["calls"]), 2)
        self.assertEqual(observed["calls"][0]["id"], "call_1")
        self.assertEqual(
            observed["calls"][1]["call"]["arguments"]["customer_email"],
            "b@example.test",
        )

    def test_normalizes_content_format_call_for_conversation_history(self) -> None:
        response = {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": (
                            '{"function":"resolve_customer","parameters":'
                            '{"customer_email":"a@example.test"},"type":"function"}'
                        ),
                    }
                }
            ],
            "usage": {"timings": {}},
        }
        observed = model_observation(response, 12.0)
        self.assertTrue(observed["content_format_call"])
        self.assertEqual(
            observed["calls"][0]["call"],
            {
                "name": "resolve_customer",
                "arguments": {"customer_email": "a@example.test"},
            },
        )
        self.assertEqual(
            observed["assistant_message"]["tool_calls"][0]["id"],
            observed["calls"][0]["id"],
        )

    def test_production_gate_requires_two_x_and_exact_outputs(self) -> None:
        summary = {
            "tasks": 6,
            "stage_batched_to_speculative_speedup_p50": 1.99,
            "stage_batched_to_speculative_bootstrap_95ci": [1.5, 2.4],
            "stage_batched_to_speculative_speedup_p05": 1.6,
            "compiled_to_speculative_speedup_p50": 1.1,
            "compiled_to_speculative_bootstrap_95ci": [1.01, 1.2],
            "pattern_prediction_hit_rate": 1.0,
            "all_predictions_from_qwen": True,
            "model_compute_slowdown_p50_percent": 0.0,
            "model_compute_slowdown_p95_percent": 0.0,
            "decode_slowdown_p50_percent": 0.0,
            "decode_slowdown_p95_percent": 0.0,
            "continuation_cache_hit_rate": 1.0,
            "prefix_cache_configured": True,
            "all_calls_stable": True,
            "all_tool_results_stable": True,
            "macro_output_stability_rate": 1.0,
            "all_final_answers_correct": True,
            "all_final_outputs_stable": True,
            "all_macro_calls_correct": True,
            "all_ds4_active": True,
        }
        args = argparse.Namespace(
            min_e2e_speedup=2.0,
            min_e2e_speedup_p05=1.5,
            min_incremental_speedup=1.05,
            min_production_pairs=6,
            max_model_slowdown_percent=1.0,
            max_model_slowdown_p95_percent=5.0,
            max_decode_slowdown_percent=1.0,
            max_decode_slowdown_p95_percent=5.0,
        )
        checks = production_checks(summary, args)
        self.assertFalse(checks["end_to_end_speedup"])
        self.assertTrue(
            all(value for key, value in checks.items() if key != "end_to_end_speedup")
        )

    def test_final_turn_is_identical_and_context_free_for_every_arm(self) -> None:
        args = argparse.Namespace(final_max_tokens=32)
        with patch(
            "benchmark_trace_compiled_workflows.post_turn",
            return_value={"content": "workflow_complete:plum"},
        ) as mocked:
            post_final(args, "workflow_complete:plum")

        call_args = mocked.call_args.args
        self.assertEqual(call_args[2], [])
        self.assertEqual(call_args[3], "none")
        self.assertEqual(call_args[4], 32)
        self.assertEqual(
            call_args[1][-1],
            {"role": "user", "content": "workflow_complete:plum"},
        )

    def test_final_receipt_accepts_literal_or_equivalent_json(self) -> None:
        expected = "workflow_complete:plum,ivory"
        self.assertTrue(final_answer_correct(expected, expected))
        self.assertTrue(
            final_answer_correct('{"workflow_complete": "plum,ivory"}', expected)
        )
        self.assertTrue(final_answer_correct("plum,ivory", expected))
        self.assertFalse(final_answer_correct('{"workflow_complete": "plum"}', expected))
        self.assertFalse(final_answer_correct("workflow_complete:ivory,plum", expected))

    def test_resume_checkpoint_requires_matching_task_and_arm_order(self) -> None:
        tasks = [make_task(0, 2, self.pattern), make_task(1, 3, self.pattern)]
        orders = [
            ["compiled", "stage_batched", "speculative"],
            ["speculative", "stage_batched", "compiled"],
        ]
        checkpoint = {
            "schema_version": 1,
            "complete": False,
            "pairs": [
                {
                    "pair_index": 0,
                    "task": tasks[0],
                    "arm_order": orders[0],
                    "stage_batched": {
                        "final": {"content": "plum,ivory"},
                        "expected_final": "workflow_complete:plum,ivory",
                    },
                    "compiled": {
                        "final": {"content": "plum,ivory"},
                        "expected_final": "workflow_complete:plum,ivory",
                    },
                    "speculative": {
                        "final": {"content": "plum,ivory"},
                        "expected_final": "workflow_complete:plum,ivory",
                    },
                }
            ],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "run.json.partial"
            path.write_text(json.dumps(checkpoint), encoding="utf-8")
            resumed = load_partial_pairs(path, tasks, orders)
            self.assertEqual(len(resumed), 1)
            self.assertTrue(
                all(
                    resumed[0][arm]["final_correct"]
                    for arm in ("stage_batched", "compiled", "speculative")
                )
            )
            checkpoint["pairs"][0]["arm_order"] = orders[1]
            path.write_text(json.dumps(checkpoint), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "does not match"):
                load_partial_pairs(path, tasks, orders)


if __name__ == "__main__":
    unittest.main()
