#!/usr/bin/env python3
"""Tests for canonical_concurrent_benchmark.py and blog prompt parity."""

from __future__ import annotations

import argparse
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "canonical_concurrent_benchmark", HERE / "canonical_concurrent_benchmark.py"
)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(benchmark)


class CanonicalBenchmarkTests(unittest.TestCase):
    def test_loads_raw_and_multi_message_cases(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cases.jsonl"
            path.write_text(
                json.dumps({"id": "raw", "prompt": "code"}) + "\n"
                + json.dumps({"id": "chat", "messages": [
                    {"role": "system", "content": "s"},
                    {"role": "user", "content": "u"},
                ]}) + "\n",
                encoding="utf-8",
            )
            cases = benchmark.load_cases(path)
        self.assertEqual(cases[0]["prompt"], "code")
        self.assertEqual(cases[1]["prompt"][0]["role"], "system")

    def test_rejects_non_array_messages(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "invalid.jsonl"
            path.write_text(json.dumps({"id": "bad", "messages": {"role": "user"}}) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "messages.*array"):
                benchmark.load_cases(path)

    def test_full_suite_waves_have_no_tail_or_reuse(self) -> None:
        cases = [
            json.dumps({"id": f"p{i}", "prompt": f"prompt {i}"})
            for i in range(10)
        ]
        seen = []

        def fake_level(clients, args, prompts, offset):
            self.assertEqual(clients, 5)
            self.assertEqual(offset, 0)
            seen.extend(prompts)
            details = [{
                "error": None, "completion_tokens": 8, "prompt_tokens": 4,
                "request_decode_tok_s": 2.0, "ttft_s": 0.1,
            } for _ in prompts]
            return {
                "requests_detail": details, "failures": 0, "wall_s": 4.0,
                "output_window_s": 3.0, "fixed_token_workload_valid": True,
            }

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompt_file = root / "cases.jsonl"
            prompt_file.write_text("\n".join(cases) + "\n", encoding="utf-8")
            args = argparse.Namespace(
                clients=5, prompt_file=prompt_file, suite="he", ignore_eos=True,
                server_metadata_json=None, label="test", base_url="x", model="m",
                max_tokens=8, temperature=0.0, seed=1, out=root / "report.json",
                retire_log=None, case_limit=None,
            )
            with mock.patch.object(benchmark.base, "run_level", side_effect=fake_level):
                self.assertEqual(benchmark.run(args), 0)
            report = json.loads(args.out.read_text(encoding="utf-8"))
        self.assertEqual(seen, [f"prompt {i}" for i in range(10)])
        self.assertEqual(report["levels"][0]["waves"], 2)
        self.assertEqual(report["levels"][0]["requests"], 10)

    def test_aggregate_waves_sums_native_prefill_windows(self) -> None:
        def wave(tokens: int, window_ms: float) -> dict:
            return {
                "requests_detail": [{
                    "error": None, "completion_tokens": 8, "prompt_tokens": 4,
                    "request_decode_tok_s": 2.0, "ttft_s": 0.1,
                }],
                "failures": 0, "wall_s": 1.0, "output_window_s": 0.8,
                "prompt_to_first_token_s": 0.2,
                "server_native_prefilled_tokens_total": tokens,
                "server_native_prefill_window_ms": window_ms,
                "server_native_prefill_token_count_complete": True,
                "server_native_prefill_timing_complete": True,
            }

        level = benchmark.aggregate_waves(1, [wave(40, 100.0), wave(60, 300.0)])
        self.assertEqual(level["server_native_prefilled_tokens_total"], 100)
        self.assertEqual(level["server_native_prefill_window_ms"], 400.0)
        self.assertEqual(level["server_native_prefill_tokens_per_s"], 250.0)
        self.assertTrue(level["server_native_prefill_token_count_complete"])
        self.assertTrue(level["server_native_prefill_timing_complete"])

        incomplete = wave(60, 300.0)
        incomplete.update({
            "server_native_prefilled_tokens_total": None,
            "server_native_prefill_window_ms": None,
            "server_native_prefill_token_count_complete": False,
            "server_native_prefill_timing_complete": False,
        })
        level = benchmark.aggregate_waves(1, [wave(40, 100.0), incomplete])
        self.assertIsNone(level["server_native_prefilled_tokens_total"])
        self.assertIsNone(level["server_native_prefill_window_ms"])
        self.assertIsNone(level["server_native_prefill_tokens_per_s"])
        self.assertFalse(level["server_native_prefill_token_count_complete"])
        self.assertFalse(level["server_native_prefill_timing_complete"])

    def test_rejects_partial_tail_wave(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompt_file = root / "cases.jsonl"
            prompt_file.write_text("".join(
                json.dumps({"id": f"p{i}", "prompt": "x"}) + "\n"
                for i in range(10)
            ), encoding="utf-8")
            args = argparse.Namespace(clients=4, prompt_file=prompt_file, case_limit=None)
            with self.assertRaisesRegex(ValueError, "lower-concurrency tail"):
                benchmark.run(args)

    def test_case_limit_supports_three_full_c3_waves(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompt_file = root / "cases.jsonl"
            prompt_file.write_text("".join(
                json.dumps({"id": f"p{i}", "prompt": f"x{i}"}) + "\n"
                for i in range(10)
            ), encoding="utf-8")
            args = argparse.Namespace(
                clients=3, case_limit=9, prompt_file=prompt_file, suite="he-raw",
                ignore_eos=True, server_metadata_json=None, label="c3", base_url="x",
                model="m", max_tokens=8, temperature=0.0, seed=1,
                out=root / "report.json", retire_log=None,
            )
            seen = []

            def fake_level(clients, _args, prompts, offset):
                self.assertEqual((clients, offset, len(prompts)), (3, 0, 3))
                seen.extend(prompts)
                return {
                    "requests_detail": [{
                        "error": None, "completion_tokens": 8, "prompt_tokens": 4,
                        "request_decode_tok_s": 2.0, "ttft_s": 0.1,
                    } for _ in prompts],
                    "failures": 0, "wall_s": 2.0, "output_window_s": 1.5,
                    "fixed_token_workload_valid": True,
                }

            with mock.patch.object(benchmark.base, "run_level", side_effect=fake_level):
                self.assertEqual(benchmark.run(args), 0)
            report = json.loads(args.out.read_text(encoding="utf-8"))
        self.assertEqual(seen, [f"x{i}" for i in range(9)])
        self.assertEqual(report["case_limit"], 9)
        self.assertEqual(report["levels"][0]["waves"], 3)

    def test_retirement_wait_matches_every_response(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / "server.log"
            log.write_text(
                '[concurrency-metrics] {"ddtree_steps": 1, "request_id": "a"}\n'
                '[server] chat DONE b ok=true\n',
                encoding="utf-8",
            )
            elapsed = benchmark.wait_for_retirement(log, ["a", "b"], 0.1)
            self.assertGreaterEqual(elapsed, 0)
            with self.assertRaisesRegex(TimeoutError, "did not retire"):
                benchmark.wait_for_retirement(log, ["missing"], 0.01)

    def test_attaches_ddtree_proof_from_matched_requests(self) -> None:
        report = {"levels": [{"wave_results": [{"requests_detail": [
            {"response_id": "a", "error": None},
            {"response_id": "b", "error": None},
        ]}]}]}
        metrics = {
            "a": {"ddtree_steps": 2, "ddtree_accepted_tokens": 8, "target_forwards": 4},
            "b": {"ddtree_steps": 3, "ddtree_accepted_tokens": 12, "target_forwards": 6},
        }
        benchmark.attach_ddtree_proof(report, metrics)
        self.assertEqual(report["ddtree_proof"]["ddtree_steps"], 5)
        self.assertEqual(report["ddtree_proof"]["mean_accepted_length"], 5.0)
        self.assertEqual(report["ddtree_proof"]["acceptance_rate"], 5 / 16)

    def test_missing_or_zero_step_ddtree_proof_fails_closed(self) -> None:
        report = {"levels": [{"wave_results": [{"requests_detail": [
            {"response_id": "a", "error": None},
        ]}]}]}
        with self.assertRaisesRegex(ValueError, "missing concurrency metric"):
            benchmark.attach_ddtree_proof(report, {})
        metrics = {"a": {
            "ddtree_steps": 0, "ddtree_accepted_tokens": 0, "target_forwards": 1,
        }}
        with self.assertRaisesRegex(ValueError, "ddtree_steps must be positive"):
            benchmark.attach_ddtree_proof(report, metrics)

    def test_boolean_ddtree_counters_are_not_integers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "server.log"
            path.write_text("[concurrency-metrics] " + json.dumps({
                "response_id": "a", "ddtree_steps": True,
                "ddtree_accepted_tokens": 1, "target_forwards": 1,
            }) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "invalid ddtree_steps"):
                benchmark.load_ddtree_metrics(path)

    def test_blog_generator_matches_bench_he_source(self) -> None:
        generator_spec = importlib.util.spec_from_file_location(
            "generate_prompts", HERE / "generate_prompts.py"
        )
        assert generator_spec is not None and generator_spec.loader is not None
        generator = importlib.util.module_from_spec(generator_spec)
        generator_spec.loader.exec_module(generator)
        self.assertEqual(len(generator.PROMPTS), 10)
        source_spec = importlib.util.spec_from_file_location(
            "bench_he_source", HERE.parents[2] / "server" / "scripts" / "bench_he.py"
        )
        assert source_spec is not None and source_spec.loader is not None
        source = importlib.util.module_from_spec(source_spec)
        source_spec.loader.exec_module(source)
        self.assertEqual(generator.PROMPTS, source.PROMPTS)
        self.assertEqual(
            (HERE / "raw_prompt_identity.jinja").read_text(encoding="utf-8"),
            "{%- for message in messages -%}{{ message.content }}{%- endfor -%}\n",
        )

    def test_summary_keeps_suites_separate_and_reports_acceptance(self) -> None:
        summary_spec = importlib.util.spec_from_file_location(
            "summarize_concurrency", HERE / "summarize_concurrency.py"
        )
        assert summary_spec is not None and summary_spec.loader is not None
        summary = importlib.util.module_from_spec(summary_spec)
        summary_spec.loader.exec_module(summary)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for suite, variant, acceptance in (("he-raw", "blog-ddtree", 0.35), ("gsm", "ar", None)):
                path = root / suite / "c1" / "r1" / variant / "bench.json"
                path.parent.mkdir(parents=True)
                report = {
                    "suite": suite, "case_limit": None,
                    "prompt_file_sha256": f"{suite}-prompts",
                    "server_metadata": {"variant": variant, "repeat": 1},
                    "levels": [{
                        "clients": 1, "failures": 0, "fixed_token_workload_valid": True,
                        "token_count_complete": True, "prompt_token_count_complete": True,
                        "requests": 1,
                        "aggregate_tok_s": 10.0, "output_window_tok_s": 11.0,
                        "prompt_tokens_per_s_to_first_token": 13.0,
                        "request_decode_tok_s_median": 12.0,
                        "ttft_median_s": 0.1, "ttft_max_s": 0.2,
                    }],
                }
                if suite == "he-raw":
                    report["levels"][0].update({
                        "server_native_prefill_window_ms": 400.0,
                        "server_native_prefill_tokens_per_s": 250.0,
                        "server_native_prefill_token_count_complete": True,
                        "server_native_prefill_timing_complete": True,
                    })
                if acceptance is not None:
                    report["ddtree_proof"] = {
                        "ddtree_steps": 1, "requests_proven": 1,
                        "mean_accepted_length": 5.6, "acceptance_rate": acceptance,
                    }
                path.write_text(json.dumps(report), encoding="utf-8")
            text = summary.summarize_canonical(root)
        self.assertIn("| he-raw | 1 | 1 | blog-ddtree", text)
        self.assertIn("5.60 | 35.0%", text)
        self.assertIn("| gsm | 1 | 1 | ar", text)
        self.assertIn("Native prefill tok/s | Native prefill window ms", text)
        he_row = next(line for line in text.splitlines() if "| he-raw |" in line)
        gsm_row = next(line for line in text.splitlines() if "| gsm |" in line)
        self.assertEqual(he_row.split("|")[-3].strip(), "250.00")
        self.assertEqual(he_row.split("|")[-2].strip(), "400.0")
        self.assertEqual(gsm_row.split("|")[-3].strip(), "n/a")
        self.assertEqual(gsm_row.split("|")[-2].strip(), "n/a")

    def test_canonical_summary_rejects_mismatched_native_completeness(self) -> None:
        summary_spec = importlib.util.spec_from_file_location(
            "summarize_concurrency_native_flags", HERE / "summarize_concurrency.py"
        )
        assert summary_spec is not None and summary_spec.loader is not None
        summary = importlib.util.module_from_spec(summary_spec)
        summary_spec.loader.exec_module(summary)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "gsm" / "c1" / "r1" / "ar" / "bench.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({
                "suite": "gsm", "case_limit": None,
                "prompt_file_sha256": "gsm-prompts",
                "server_metadata": {"variant": "ar", "repeat": 1},
                "levels": [{
                    "clients": 1, "requests": 1, "failures": 0,
                    "fixed_token_workload_valid": True,
                    "token_count_complete": True,
                    "prompt_token_count_complete": True,
                    "aggregate_tok_s": 10.0,
                    "output_window_tok_s": 11.0,
                    "prompt_tokens_per_s_to_first_token": 13.0,
                    "request_decode_tok_s_median": 12.0,
                    "ttft_median_s": 0.1,
                    "ttft_max_s": 0.2,
                    "server_native_prefill_token_count_complete": True,
                    "server_native_prefill_timing_complete": False,
                }],
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "mismatched native prefill"):
                summary.summarize_canonical(root)

    def test_summary_reports_output_stability_by_case_id(self) -> None:
        summary_spec = importlib.util.spec_from_file_location(
            "summarize_concurrency_stability", HERE / "summarize_concurrency.py"
        )
        assert summary_spec is not None and summary_spec.loader is not None
        summary = importlib.util.module_from_spec(summary_spec)
        summary_spec.loader.exec_module(summary)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for repeat, content_hash in ((1, "same"), (2, "changed")):
                path = root / "gsm" / "c1" / f"r{repeat}" / "ar" / "bench.json"
                path.parent.mkdir(parents=True)
                path.write_text(json.dumps({
                    "suite": "gsm", "case_limit": None,
                    "prompt_file_sha256": "gsm-prompts",
                    "server_metadata": {"variant": "ar", "repeat": repeat},
                    "levels": [{
                        "clients": 1, "requests": 1, "failures": 0,
                        "fixed_token_workload_valid": True, "aggregate_tok_s": 10.0,
                        "token_count_complete": True, "prompt_token_count_complete": True,
                        "output_window_tok_s": 11.0,
                        "prompt_tokens_per_s_to_first_token": 13.0,
                        "request_decode_tok_s_median": 12.0,
                        "ttft_median_s": 0.1, "ttft_max_s": 0.2,
                        "wave_results": [{"requests_detail": [{
                            "case_id": "gsm_01", "content_sha256": content_hash,
                            "reasoning_content_sha256": "reasoning",
                        }]}],
                    }],
                }), encoding="utf-8")
            text = summary.summarize_canonical(root)
        self.assertIn("| n/a | n/a | NO |", text)

    def test_ddtree_proof_rejects_failed_request_before_attachment(self) -> None:
        request = {"response_id": "a", "error": "request failed"}
        report = {"levels": [{
            "failures": 1,
            "wave_results": [{"requests_detail": [request]}],
        }]}
        metrics = {
            "a": {"ddtree_steps": 2, "ddtree_accepted_tokens": 8, "target_forwards": 4},
        }
        with self.assertRaisesRegex(ValueError, "failed requests"):
            benchmark.attach_ddtree_proof(report, metrics)
        self.assertNotIn("ddtree_metrics", request)
        self.assertNotIn("ddtree_proof", report)

    def test_ddtree_run_writes_failed_report_without_proof(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prompt_file = root / "cases.jsonl"
            prompt_file.write_text(json.dumps({"id": "p0", "prompt": "prompt"}) + "\n",
                                   encoding="utf-8")
            args = argparse.Namespace(
                clients=1, prompt_file=prompt_file, case_limit=None, suite="gsm",
                ignore_eos=True, server_metadata_json=None, label="ddtree",
                base_url="x", model="m", max_tokens=8, temperature=0.0, seed=1,
                out=root / "report.json", retire_log=root / "server.log",
                ddtree_proof=True,
            )
            wave = {
                "requests_detail": [{
                    "error": "request failed", "completion_tokens": None,
                    "prompt_tokens": None, "request_decode_tok_s": None,
                    "ttft_s": None,
                }],
                "failures": 1, "wall_s": 1.0, "output_window_s": None,
                "fixed_token_workload_valid": False,
            }
            with mock.patch.object(benchmark.base, "run_level", return_value=wave):
                self.assertEqual(benchmark.run(args), 1)
            report = json.loads(args.out.read_text(encoding="utf-8"))
        self.assertNotIn("ddtree_proof", report)
        self.assertEqual(report["levels"][0]["failures"], 1)

    def test_canonical_rejects_multiple_client_levels(self) -> None:
        summary_spec = importlib.util.spec_from_file_location(
            "summarize_concurrency_levels", HERE / "summarize_concurrency.py"
        )
        assert summary_spec is not None and summary_spec.loader is not None
        summary = importlib.util.module_from_spec(summary_spec)
        summary_spec.loader.exec_module(summary)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "gsm" / "c1" / "r1" / "ar" / "bench.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({
                "suite": "gsm", "case_limit": None,
                "server_metadata": {"variant": "ar", "repeat": 1},
                "levels": [{"clients": 1}, {"clients": 2}],
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exactly one client level"):
                summary.summarize_canonical(root)

    def test_canonical_rejects_incomplete_token_accounting(self) -> None:
        summary_spec = importlib.util.spec_from_file_location(
            "summarize_concurrency_tokens", HERE / "summarize_concurrency.py"
        )
        assert summary_spec is not None and summary_spec.loader is not None
        summary = importlib.util.module_from_spec(summary_spec)
        summary_spec.loader.exec_module(summary)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "gsm" / "c1" / "r1" / "ar" / "bench.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({
                "suite": "gsm", "case_limit": None,
                "server_metadata": {"variant": "ar", "repeat": 1},
                "levels": [{
                    "clients": 1, "requests": 1, "failures": 0,
                    "fixed_token_workload_valid": True,
                    "token_count_complete": True,
                    "prompt_token_count_complete": False,
                }],
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "incomplete token accounting"):
                summary.summarize_canonical(root)

    def test_canonical_rejects_missing_or_mismatched_prompt_file_hash(self) -> None:
        summary_spec = importlib.util.spec_from_file_location(
            "summarize_concurrency_prompt_hash", HERE / "summarize_concurrency.py"
        )
        assert summary_spec is not None and summary_spec.loader is not None
        summary = importlib.util.module_from_spec(summary_spec)
        summary_spec.loader.exec_module(summary)

        def write_report(root: Path, repeat: int, prompt_hash: str | None) -> None:
            path = root / "gsm" / "c1" / f"r{repeat}" / "ar" / "bench.json"
            path.parent.mkdir(parents=True)
            report = {
                "suite": "gsm", "case_limit": None,
                "server_metadata": {"variant": "ar", "repeat": repeat},
                "levels": [{
                    "clients": 1, "requests": 1, "failures": 0,
                    "fixed_token_workload_valid": True,
                    "token_count_complete": True,
                    "prompt_token_count_complete": True,
                    "aggregate_tok_s": 1.0, "output_window_tok_s": 1.0,
                    "prompt_tokens_per_s_to_first_token": 1.0,
                    "request_decode_tok_s_median": 1.0,
                    "ttft_median_s": 1.0, "ttft_max_s": 1.0,
                }],
            }
            if prompt_hash is not None:
                report["prompt_file_sha256"] = prompt_hash
            path.write_text(json.dumps(report), encoding="utf-8")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_report(root, 1, None)
            with self.assertRaisesRegex(ValueError, "missing prompt_file_sha256"):
                summary.summarize_canonical(root)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_report(root, 1, "prompt-a")
            write_report(root, 2, "prompt-b")
            with self.assertRaisesRegex(ValueError, "prompt files differ"):
                summary.summarize_canonical(root)


if __name__ == "__main__":
    unittest.main()
