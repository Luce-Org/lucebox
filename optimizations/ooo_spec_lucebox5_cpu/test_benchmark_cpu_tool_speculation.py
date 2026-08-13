from __future__ import annotations

import argparse
import unittest

from benchmark_cpu_tool_speculation import (
    TOOL_NAME,
    expected_arguments,
    parse_cpu_list,
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


if __name__ == "__main__":
    unittest.main()
