#!/usr/bin/env python3

import json
import unittest
from unittest import mock

import dag_bench
import rest_tools


class DagDependencyTests(unittest.TestCase):
    def test_nested_result_paths_work_for_whole_and_embedded_values(self):
        results = {
            1: {"value": {"meta": {"latitude": 12.5, "city": "Rome"}}},
        }
        resolved, deps, error = dag_bench.resolve_args(
            {
                "latitude": "$1.meta.latitude",
                "explicit": "$1.value.meta.latitude",
                "label": "weather for $1.meta.city",
            },
            results,
        )
        self.assertIsNone(error)
        self.assertEqual(sorted(set(deps)), [1])
        self.assertEqual(resolved["latitude"], 12.5)
        self.assertEqual(resolved["explicit"], 12.5)
        self.assertEqual(resolved["label"], "weather for Rome")

    def test_json_null_is_a_value_but_missing_nested_field_is_an_error(self):
        results = {1: {"value": {"meta": {"nullable": None}}}}
        resolved, _, error = dag_bench.resolve_args(
            {"value": "$1.meta.nullable"}, results)
        self.assertIsNone(error)
        self.assertIsNone(resolved["value"])

        _, _, error = dag_bench.resolve_args(
            {"value": "$1.meta.missing"}, results)
        self.assertIn("meta.missing", error)


class LiveResultTests(unittest.TestCase):
    def test_live_tools_attach_a_commit_freshness_deadline(self):
        with mock.patch.object(rest_tools, "_call", return_value={"rate": 1.2}), \
             mock.patch.object(rest_tools.time, "time", return_value=100.0):
            result = rest_tools.run_tool(
                "exchange_rate", {"base": "EUR", "quote": "USD"})
        self.assertTrue(result["ok"])
        self.assertEqual(
            result["_speculation_fresh_until_unix_ms"],
            100_000 + rest_tools.LIVE_RESULT_MAX_AGE_MS["exchange_rate"],
        )

    def test_static_tools_do_not_need_a_freshness_deadline(self):
        with mock.patch.object(rest_tools, "_call", return_value={"latitude": 1.0}):
            result = rest_tools.run_tool("geocode_city", {"city": "Rome"})
        self.assertNotIn("_speculation_fresh_until_unix_ms", result)

    def test_tool_message_format_is_canonical_across_benchmark_arms(self):
        raw = {"value": {"z": 1, "a": "München"}}
        self.assertEqual(
            rest_tools.format_result(raw),
            '{"a":"München","z":1}',
        )


class ResultDerivedValidationTests(unittest.TestCase):
    def test_prompt_keywords_without_returned_value_do_not_pass(self):
        result = {
            "tool_name": "get_weather",
            "ok": True,
            "value": {"temperature_c": 18.7},
        }
        validation = rest_tools.validate_answer_from_results(
            "What is the current temperature in Tokyo?",
            "Tokyo has current weather.",
            [result],
        )
        self.assertFalse(validation["ok"])

    def test_reported_tool_value_passes(self):
        result = {
            "tool_name": "get_weather",
            "ok": True,
            "value": {"temperature_c": 18.7},
        }
        validation = rest_tools.validate_answer_from_results(
            "What is the current temperature in Tokyo?",
            "Tokyo is currently 18.7 °C.",
            [result],
        )
        self.assertTrue(validation["ok"])

    def test_failed_tool_never_passes(self):
        validation = rest_tools.validate_answer_from_results(
            "What is the weather?",
            "It is 18.7 °C.",
            [{"tool_name": "get_weather", "ok": False, "value": {}}],
        )
        self.assertEqual(validation["reason"], "tool_failure")


if __name__ == "__main__":
    unittest.main()
