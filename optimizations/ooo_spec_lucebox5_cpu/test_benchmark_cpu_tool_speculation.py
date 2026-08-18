from __future__ import annotations

import argparse
import signal
import subprocess
import unittest
from unittest.mock import Mock, patch

from benchmark_cpu_tool_speculation import (
    TOOL_NAME,
    expected_arguments,
    normalize_tool_call,
    parse_cpu_list,
    percentile,
    props_url,
    require_qualified_cpu_tool_props,
    request_body,
    stop_executor,
)


class CpuToolSpeculationBenchmarkTest(unittest.TestCase):
    def test_stop_executor_signals_group_after_leader_exit(self) -> None:
        process = Mock(pid=4321)
        process.poll.return_value = 0

        def kill_group(_pgid: int, sent_signal: int) -> None:
            if sent_signal == 0:
                raise ProcessLookupError

        with patch(
            "benchmark_cpu_tool_speculation.os.killpg",
            side_effect=kill_group,
        ) as killpg:
            stop_executor({"process": process, "pgid": 4321})

        killpg.assert_any_call(4321, signal.SIGTERM)

    def test_stop_executor_converts_final_wait_timeout(self) -> None:
        process = Mock(pid=4321)
        process.poll.return_value = None
        process.wait.side_effect = subprocess.TimeoutExpired("executor", 5.0)

        def kill_group(_pgid: int, sent_signal: int) -> None:
            if sent_signal == 0:
                raise ProcessLookupError

        with patch(
            "benchmark_cpu_tool_speculation.os.killpg",
            side_effect=kill_group,
        ):
            with self.assertRaisesRegex(RuntimeError, "did not stop"):
                stop_executor({"process": process, "pgid": 4321})

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

    def test_automatic_gate_requires_qualified_disjoint_lane(self) -> None:
        args = argparse.Namespace(tool_cpus=[14, 15])
        tool_props = {
            "enabled": True,
            "automatic_prediction_enabled": True,
            "execution_mode": "child_process_cpu_affinity",
            "profile_status": "qualified",
            "compute_isolation": "disjoint_cpu_affinity",
            "cpu_affinity_isolated": True,
            "preserves_token_speculation": True,
            "tool_cpu_affinity": [14, 15],
            "model_cpu_affinity": [0, 1],
        }
        self.assertIs(
            require_qualified_cpu_tool_props(
                {"tool_speculation": tool_props}, args, automatic=True
            ),
            tool_props,
        )
        with self.assertRaisesRegex(SystemExit, "profile_status"):
            require_qualified_cpu_tool_props(
                {
                    "tool_speculation": {
                        **tool_props,
                        "profile_status": "unqualified",
                    }
                },
                args,
                automatic=True,
            )


if __name__ == "__main__":
    unittest.main()
