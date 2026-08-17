from __future__ import annotations

import argparse
import unittest

from benchmark_cpu_tool_speculation import (
    TOOL_NAME,
    expected_arguments,
    normalize_tool_call,
    parse_cpu_list,
    percentile,
    props_url,
    request_body,
)


class CpuToolSpeculationBenchmarkTest(unittest.TestCase):
    def test_cpu_list_parser_canonicalizes_ranges(self) -> None:
        self.assertEqual(parse_cpu_list("30-31,15,14-15"), [14, 15, 30, 31])
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_cpu_list("14,,15")

    def test_request_has_exact_concrete_prediction(self) -> None:
        arguments = expected_arguments(4096, 16, 77, 2, 731)
        self.assertEqual(arguments, {"iterations": 77})
        body = request_body(arguments, 32, prediction=arguments)
        self.assertEqual(
            body["tool_speculation"]["call"],
            {"name": TOOL_NAME, "arguments": arguments},
        )
        self.assertEqual(body["tool_speculation"]["confidence"], 1.0)
        self.assertEqual(body["tools"][0]["name"], TOOL_NAME)

    def test_props_url_uses_server_origin(self) -> None:
        self.assertEqual(
            props_url("http://127.0.0.1:18145/v1/chat/completions?x=1"),
            "http://127.0.0.1:18145/props",
        )

    def test_percentile_interpolates_sorted_values(self) -> None:
        self.assertEqual(percentile([4, 1, 3, 2], 0.0), 1.0)
        self.assertEqual(percentile([4, 1, 3, 2], 0.5), 2.5)
        self.assertEqual(percentile([4, 1, 3, 2], 1.0), 4.0)

    def test_automatic_qwen_arm_has_no_oracle_prediction(self) -> None:
        arguments = {"iterations": 77}
        body = request_body(
            arguments,
            32,
            prediction=None,
            automatic_prediction=True,
            tool_choice="required",
        )
        self.assertNotIn("tool_speculation", body)
        self.assertTrue(body["automatic_tool_speculation"])
        self.assertEqual(body["tool_choice"], "required")

    def test_normalizes_deepseek_single_parameter_envelope(self) -> None:
        result = {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"function":"batch_resolve_customer",'
                            '"parameter":"stage_ref",'
                            '"parameter_value":"workflow_taskf_stage_one"}'
                        )
                    }
                }
            ]
        }
        self.assertEqual(
            normalize_tool_call(result),
            {
                "name": "batch_resolve_customer",
                "arguments": {"stage_ref": "workflow_taskf_stage_one"},
            },
        )


if __name__ == "__main__":
    unittest.main()
