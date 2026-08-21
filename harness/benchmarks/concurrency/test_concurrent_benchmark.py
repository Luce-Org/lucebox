#!/usr/bin/env python3
"""Focused tests for concurrent_benchmark.py."""

from __future__ import annotations

import argparse
import importlib.util
import time
import unittest
from pathlib import Path
from unittest import mock

SCRIPT = Path(__file__).with_name("concurrent_benchmark.py")
SPEC = importlib.util.spec_from_file_location("concurrent_benchmark", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


class BenchmarkTests(unittest.TestCase):
    def test_sse_parser_handles_events_and_done(self) -> None:
        lines = [
            b'data: {"choices":[{"delta":{"content":"hi"}}]}\n', b"\n",
            b"data: [DONE]\n", b"\n",
        ]
        self.assertEqual(
            list(benchmark.iter_sse_data(lines)),
            ['{"choices":[{"delta":{"content":"hi"}}]}', "[DONE]"],
        )

    def test_default_matrix_starts_at_c2(self) -> None:
        self.assertEqual(benchmark.DEFAULT_CLIENT_LEVELS, (2, 4, 8, 16))

    def test_run_rejects_duplicate_client_levels(self) -> None:
        args = argparse.Namespace(client_levels=[2, 4, 2])
        with self.assertRaisesRegex(ValueError, "distinct"):
            benchmark.run(args)

    def test_prompt_selection_never_wraps(self) -> None:
        self.assertEqual(benchmark.request_prompts(["a", "b", "c"], 2, 1), ["b", "c"])
        with self.assertRaisesRegex(ValueError, "refusing to reuse"):
            benchmark.request_prompts(["a", "b"], 2, 1)

    def test_prompt_messages_rejects_non_array(self) -> None:
        with self.assertRaisesRegex(ValueError, "messages must contain"):
            benchmark.prompt_messages({"role": "user"})

    def test_level_uses_exact_usage_and_first_token_window(self) -> None:
        prompt_counts = iter((10, 30))

        def fake_request(_args: argparse.Namespace, prompt: str) -> dict:
            prompt_count = next(prompt_counts)
            start = time.perf_counter()
            return {
                "t_start": start, "t_first": start + 0.5, "t_end": start + 1.0,
                "duration_s": 1.0, "ttft_s": 0.5, "decode_duration_s": 0.5,
                "completion_tokens": 8, "prompt_tokens": prompt_count,
                "server_native_prefilled_tokens": prompt_count,
                "server_native_prefill_ms": 100.0 if prompt_count == 10 else 200.0,
                "finish_reason": "length", "error": None,
                "content_sha256": benchmark.sha256_text(prompt + " output"),
                "reasoning_content_sha256": benchmark.sha256_text(""),
                "content_chars": 6, "reasoning_content_chars": 0,
                "request_output_tok_s": 8.0, "request_decode_tok_s": 14.0,
            }

        args = argparse.Namespace(max_tokens=8, ignore_eos=True, timeout=2.0)
        with mock.patch.object(benchmark, "stream_request", side_effect=fake_request):
            level = benchmark.run_level(2, args, ["first", "second"], 0)
        self.assertEqual(level["completion_tokens_total"], 16)
        self.assertEqual(level["prompt_tokens_total"], 40)
        self.assertTrue(level["fixed_token_workload_valid"])
        self.assertAlmostEqual(
            level["output_window_tok_s"],
            16 / level["output_window_s"],
        )
        self.assertEqual(level["request_decode_tok_s_median"], 14.0)
        self.assertAlmostEqual(
            level["prompt_tokens_per_s_to_first_token"],
            40 / level["prompt_to_first_token_s"],
        )
        self.assertEqual(level["server_native_prefilled_tokens_total"], 40)
        self.assertEqual(level["server_native_prefill_ms_max"], 200.0)
        expected_window_ms = max(
            record["start_offset_s"] * 1000.0 + record["server_native_prefill_ms"]
            for record in level["requests_detail"]
        )
        self.assertAlmostEqual(
            level["server_native_prefill_window_ms"], expected_window_ms
        )
        self.assertAlmostEqual(
            level["server_native_prefill_tokens_per_s"],
            40_000.0 / expected_window_ms,
        )
        self.assertTrue(level["server_native_prefill_token_count_complete"])
        self.assertTrue(level["server_native_prefill_timing_complete"])

    def test_stream_request_keeps_usage_separate_from_sse_chunks(self) -> None:
        class Response:
            def __enter__(self): return self
            def __exit__(self, *_args): return None
            def __iter__(self):
                return iter([
                    b'data: {"choices":[{"delta":{"content":"one chunk"}}]}\n', b"\n",
                    b'data: {"choices":[{"delta":{},"finish_reason":"length"}]}\n', b"\n",
                    b'data: {"choices":[],"usage":{"prompt_tokens":12,"completion_tokens":64,"timings":{"prefilled_tokens":9,"prefill_ms":30.0}}}\n', b"\n",
                    b"data: [DONE]\n", b"\n",
                ])

        args = argparse.Namespace(
            model="m", max_tokens=64, temperature=0.0, seed=1, ignore_eos=True,
            api_key="", base_url="http://localhost/v1", timeout=2.0,
        )
        with mock.patch.object(benchmark.urllib.request, "urlopen", return_value=Response()):
            record = benchmark.stream_request(args, "prompt")
        self.assertEqual(record["completion_tokens"], 64)
        self.assertEqual(record["prompt_tokens"], 12)
        self.assertTrue(record["done_received"])
        self.assertIsNone(record["error"])
        self.assertIsNotNone(record["request_decode_tok_s"])
        self.assertEqual(record["content_sha256"], benchmark.sha256_text("one chunk"))
        self.assertEqual(record["server_native_prefilled_tokens"], 9)
        self.assertEqual(record["server_native_prefill_ms"], 30.0)
        self.assertEqual(record["server_native_prefill_tokens_per_s"], 300.0)

    def test_missing_prompt_usage_invalidates_fixed_workload(self) -> None:
        def fake_request(_args: argparse.Namespace, prompt: str) -> dict:
            start = time.perf_counter()
            return {
                "t_start": start, "t_first": start + 0.5, "t_end": start + 1.0,
                "duration_s": 1.0, "ttft_s": 0.5, "decode_duration_s": 0.5,
                "completion_tokens": 8, "prompt_tokens": None,
                "finish_reason": "length", "error": None,
                "content_sha256": benchmark.sha256_text(prompt + " output"),
                "reasoning_content_sha256": benchmark.sha256_text(""),
                "content_chars": 6, "reasoning_content_chars": 0,
                "request_output_tok_s": 8.0, "request_decode_tok_s": 14.0,
            }

        args = argparse.Namespace(max_tokens=8, ignore_eos=True, timeout=2.0)
        with mock.patch.object(benchmark, "stream_request", side_effect=fake_request):
            level = benchmark.run_level(1, args, ["prompt"], 0)
        self.assertFalse(level["prompt_token_count_complete"])
        self.assertFalse(level["fixed_token_workload_valid"])

    def test_stream_request_preserves_canonical_message_roles(self) -> None:
        captured = {}

        class Response:
            def __enter__(self): return self
            def __exit__(self, *_args): return None
            def __iter__(self):
                return iter([
                    b'data: {"choices":[{"delta":{"content":"ok"},"finish_reason":"length"}],"usage":{"prompt_tokens":4,"completion_tokens":1}}\n',
                    b"\n", b"data: [DONE]\n", b"\n",
                ])

        def fake_open(request, timeout):
            captured["payload"] = __import__("json").loads(request.data)
            return Response()

        args = argparse.Namespace(
            model="m", max_tokens=1, temperature=0.0, seed=1, ignore_eos=True,
            api_key="", base_url="http://localhost/v1", timeout=2.0,
        )
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "user"},
        ]
        with mock.patch.object(benchmark.urllib.request, "urlopen", side_effect=fake_open):
            record = benchmark.stream_request(args, messages)
        self.assertEqual(captured["payload"]["messages"], messages)
        self.assertIsNone(record["server_native_prefilled_tokens"])
        self.assertIsNone(record["server_native_prefill_ms"])
        self.assertIsNone(record["server_native_prefill_tokens_per_s"])

    def test_native_prefill_values_rejects_partial_or_invalid_shapes(self) -> None:
        self.assertEqual(benchmark.native_prefill_values({}), (None, None))
        self.assertEqual(
            benchmark.native_prefill_values({
                "timings": {"prefilled_tokens": True, "prefill_ms": "10"},
            }),
            (None, None),
        )
        self.assertEqual(
            benchmark.native_prefill_values({
                "timings": {"prefilled_tokens": 4, "prefill_ms": None},
            }),
            (4, None),
        )

    def test_stream_request_rejects_clean_eof_without_done(self) -> None:
        class Response:
            def __enter__(self): return self
            def __exit__(self, *_args): return None
            def __iter__(self):
                return iter([
                    b'data: {"choices":[{"delta":{"content":"partial"}}]}\n', b"\n",
                    b'data: {"choices":[{"delta":{},"finish_reason":"length"}]}\n', b"\n",
                    b'data: {"choices":[],"usage":{"prompt_tokens":12,"completion_tokens":64}}\n', b"\n",
                ])

        args = argparse.Namespace(
            model="m", max_tokens=64, temperature=0.0, seed=1, ignore_eos=True,
            api_key="", base_url="http://localhost/v1", timeout=2.0,
        )
        with mock.patch.object(benchmark.urllib.request, "urlopen", return_value=Response()):
            record = benchmark.stream_request(args, "prompt")
        self.assertFalse(record["done_received"])
        self.assertIn("before [DONE]", record["error"])

    def test_missing_prompt_usage_fails_level(self) -> None:
        level = {
            "failures": 0,
            "token_count_complete": True,
            "prompt_token_count_complete": False,
            "fixed_token_workload_valid": True,
        }
        self.assertTrue(benchmark.level_failed(level, ignore_eos=True))


if __name__ == "__main__":
    unittest.main()
