#!/usr/bin/env python3
"""Unit tests for the deterministic prompt generator and compact summarizer."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).parent


def load(name: str):
    path = HERE / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


generator = load("generate_prompts")
summarizer = load("summarize_concurrency")


class PromptGeneratorTests(unittest.TestCase):
    def test_cohorts_are_disjoint_ragged_and_mean_matched(self) -> None:
        records = generator.build_records("short")
        self.assertEqual(len(records), 30)
        self.assertEqual(
            [row["cohort"] for row in records],
            ["c2"] * 2 + ["c4"] * 4 + ["c8"] * 8 + ["c16"] * 16,
        )
        self.assertEqual(len({row["prompt"] for row in records}), 30)
        by_cohort = {
            cohort: [row for row in records if row["cohort"] == cohort]
            for cohort in ("c2", "c4", "c8", "c16")
        }
        means = {
            cohort: sum(row["target_words"] for row in rows) / len(rows)
            for cohort, rows in by_cohort.items()
        }
        self.assertEqual(len(set(means.values())), 1)
        self.assertEqual(
            {cohort: rows[0]["cohort_offset"] for cohort, rows in by_cohort.items()},
            {"c2": 0, "c4": 2, "c8": 6, "c16": 14},
        )
        self.assertEqual(
            {row["target_words"] for row in by_cohort["c2"]}, {250, 550}
        )
        for cohort in ("c4", "c8", "c16"):
            self.assertEqual(len({row["target_words"] for row in by_cohort[cohort]}), 4)
        for row in records:
            self.assertEqual(len(row["prompt"].split()), row["target_words"])

    def test_extended_matrix_is_deterministic_without_changing_existing_cohorts(self) -> None:
        base = generator.build_records("long", (2, 4, 8, 16))
        extended = generator.build_records("long", (2, 4, 8, 16, 32, 33))
        self.assertEqual(base, extended[:len(base)])
        self.assertEqual(len(extended), 95)
        self.assertEqual(len({row["id"] for row in extended}), len(extended))
        self.assertEqual(len({row["prompt"] for row in extended}), len(extended))
        offset = 0
        for clients in (2, 4, 8, 16, 32, 33):
            cohort = [
                row for row in extended if row["cohort_clients"] == clients
            ]
            self.assertEqual(len(cohort), clients)
            self.assertTrue(all(row["cohort_offset"] == offset for row in cohort))
            self.assertEqual(
                [row["cohort_index"] for row in cohort], list(range(clients))
            )
            self.assertEqual(
                sum(row["target_words"] for row in cohort), clients * 3000
            )
            offset += clients

    def test_client_level_parser_rejects_reuse(self) -> None:
        self.assertEqual(generator.parse_client_levels("2,4,8,16,32"), (2, 4, 8, 16, 32))
        with self.assertRaisesRegex(ValueError, "distinct"):
            generator.parse_client_levels("2,4,2")
        with self.assertRaisesRegex(ValueError, "positive"):
            generator.parse_client_levels("2,0")


class RunnerTests(unittest.TestCase):
    def test_runners_isolate_and_record_selected_gpu(self) -> None:
        for script_name in (
            "run_qwen36_concurrency.sh",
            "run_qwen36_canonical_concurrency.sh",
        ):
            text = (HERE / script_name).read_text(encoding="utf-8")
            self.assertIn('GPU_DEVICE="${GPU_DEVICE:-0}"', text)
            self.assertIn('ROCR_VISIBLE_DEVICES="$GPU_DEVICE"', text)
            self.assertIn('"rocr_visible_devices"', text)

    def test_ragged_runner_derives_offsets_from_requested_matrix(self) -> None:
        text = (HERE / "run_qwen36_concurrency.sh").read_text(encoding="utf-8")
        self.assertIn('CLIENTS="${CLIENTS:-2,4,8,16}"', text)
        self.assertIn('prompt_offsets[$c]="$next_prompt_offset"', text)
        self.assertIn('--clients "$CLIENTS"', text)
        self.assertNotIn("prompt_offsets=([", text)

    def test_ragged_runner_records_prefill_first_policy(self) -> None:
        text = (HERE / "run_qwen36_concurrency.sh").read_text(encoding="utf-8")
        self.assertIn('PREFILL_FIRST_BURST_STEPS="${PREFILL_FIRST_BURST_STEPS:-0}"', text)
        self.assertIn('DFLASH_PREFILL_FIRST_BURST_STEPS="$PREFILL_FIRST_BURST_STEPS"', text)
        self.assertIn('"prefill_first_burst_steps"', text)
        self.assertIn('(( 10#$PREFILL_FIRST_BURST_STEPS > 1024 ))', text)
        launch_start = text.index("launch_command=(env", text.index("else\n    launch_command="))
        burst_assignment = text.index(
            'DFLASH_PREFILL_FIRST_BURST_STEPS="$PREFILL_FIRST_BURST_STEPS"',
            launch_start,
        )
        launch_end = text.index('"${command[@]}")', burst_assignment)
        self.assertLess(launch_start, burst_assignment)
        self.assertLess(burst_assignment, launch_end)

    def test_runners_record_bounded_idle_prefill_budget(self) -> None:
        ragged = (HERE / "run_qwen36_concurrency.sh").read_text(encoding="utf-8")
        canonical = (
            HERE / "run_qwen36_canonical_concurrency.sh"
        ).read_text(encoding="utf-8")
        for text in (ragged, canonical):
            self.assertIn(
                'IDLE_PREFILL_TOKENS="${IDLE_PREFILL_TOKENS:-4096}"', text
            )
            self.assertIn('^[0-9]{1,4}$ ]]', text)
            self.assertIn('^[1-9][0-9]{0,4}$ ]]', text)
            self.assertIn('(( 10#$IDLE_PREFILL_TOKENS > 16384 ))', text)
            self.assertIn(
                "IDLE_PREFILL_TOKENS must be an integer in range 1..16384", text
            )
            self.assertIn('"idle_prefill_tokens"', text)
        assignment = 'DFLASH_IDLE_PREFILL_TOKENS="$IDLE_PREFILL_TOKENS"'
        self.assertEqual(ragged.count(assignment), 1)
        self.assertEqual(canonical.count(assignment), 2)
        self.assertIn(
            '"idle_prefill_tokens":int(idle_prefill_tokens) '
            'if variant != "llama" else None',
            ragged,
        )
        self.assertIn(
            '"idle_prefill_tokens":int(idle_prefill_tokens)', canonical
        )

    def test_ragged_runner_rejects_oversized_tuning_integers(self) -> None:
        runner = HERE / "run_qwen36_concurrency.sh"
        oversized = "18446744073709551616"
        cases = (
            (
                "PREFILL_FIRST_BURST_STEPS",
                "PREFILL_FIRST_BURST_STEPS must be an integer in range 0..1024",
            ),
            (
                "IDLE_PREFILL_TOKENS",
                "IDLE_PREFILL_TOKENS must be an integer in range 1..16384",
            ),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for variable, message in cases:
                env = {
                    "PATH": os.environ.get("PATH", ""),
                    "MODEL": "/dev/null",
                    "LUCE_SERVER_BIN": "/bin/true",
                    "LLAMA_SERVER_BIN": "/bin/true",
                    "OUT": str(root / variable),
                    variable: oversized,
                }
                result = subprocess.run(
                    [str(runner)], env=env, text=True, capture_output=True, check=False
                )
                with self.subTest(variable=variable):
                    self.assertEqual(result.returncode, 2)
                    self.assertIn(message, result.stderr)

    def test_ragged_runner_can_fail_closed_on_gpu_arch(self) -> None:
        text = (HERE / "run_qwen36_concurrency.sh").read_text(encoding="utf-8")
        self.assertIn('EXPECTED_GPU_ARCH="${EXPECTED_GPU_ARCH:-}"', text)
        self.assertIn('rocminfo 2>/dev/null | grep -F -- "$EXPECTED_GPU_ARCH"', text)
        self.assertIn('"$case_dir/gpu-identity.txt"', text)
        self.assertIn('"$binary" --list-devices', text)
        self.assertIn('"$case_dir/llama-list-devices.txt"', text)
        self.assertIn('-ngl all -lv 4 --reasoning off --reasoning-format none', text)
        self.assertIn('offloaded ([1-9][0-9]*)/([1-9][0-9]*) layers to GPU', text)
        self.assertIn('"$case_dir/server-gpu-proof.txt"', text)
        self.assertIn('"expected_gpu_arch"', text)


class SummarizerTests(unittest.TestCase):
    @staticmethod
    def item(
        variant: str,
        goodput: float,
        output_window: float | None = None,
        *,
        repeat: int = 1,
        output_hash: str = "same-outputs",
    ) -> dict:
        return {
            "meta": {"workload": "short", "variant": variant, "repeat": repeat},
            "level": {
                "clients": 8,
                "aggregate_tok_s": goodput,
                "output_window_tok_s": output_window if output_window is not None else goodput,
                "request_decode_tok_s_median": goodput / 8,
                "prompt_tokens_per_s_to_first_token": 100.0,
                "ttft_median_s": 1.0,
                "ttft_max_s": 2.0,
                "selected_prompt_set_sha256": "same-prompts",
                "selected_output_set_sha256": output_hash,
            },
        }

    def test_summary_reports_product_and_packing_deltas(self) -> None:
        text = summarizer.summarize([
            self.item("luce-k8", 20.0),
            self.item("luce-k1", 10.0),
            self.item("llama", 8.0),
        ])
        self.assertIn("+150.0%", text)
        self.assertIn("+100.0%", text)
        self.assertIn("Decode vs llama", text)

    def test_summary_uses_same_repeat_ratios(self) -> None:
        reports = []
        for repeat, luce, llama in (
            (1, 10.0, 1.0),
            (2, 20.0, 90.0),
            (3, 100.0, 50.0),
        ):
            reports.extend([
                self.item("luce-k8", luce, repeat=repeat),
                self.item("llama", llama, repeat=repeat),
            ])
        text = summarizer.summarize(reports)
        luce_row = next(line for line in text.splitlines() if "| luce-k8 |" in line)
        self.assertIn("+100.0%", luce_row)
        self.assertNotIn("-60.0%", luce_row)

    def test_summary_rejects_mismatched_repeat_sets(self) -> None:
        reports = [
            self.item("luce-k8", 20.0, repeat=1),
            self.item("luce-k8", 22.0, repeat=2),
            self.item("llama", 10.0, repeat=1),
        ]
        with self.assertRaisesRegex(ValueError, "repeat sets differ"):
            summarizer.summarize(reports)

    def test_single_repeat_does_not_claim_stability(self) -> None:
        text = summarizer.summarize([self.item("llama", 8.0)])
        row = next(line for line in text.splitlines() if "| llama |" in line)
        self.assertEqual(row.split("|")[11].strip(), "n/a")

    def test_multiple_repeats_report_output_stability(self) -> None:
        stable = summarizer.summarize([
            self.item("llama", 8.0, repeat=1),
            self.item("llama", 9.0, repeat=2),
        ])
        stable_row = next(line for line in stable.splitlines() if "| llama |" in line)
        self.assertEqual(stable_row.split("|")[11].strip(), "yes")

        unstable = summarizer.summarize([
            self.item("llama", 8.0, repeat=1, output_hash="first"),
            self.item("llama", 9.0, repeat=2, output_hash="second"),
        ])
        unstable_row = next(line for line in unstable.splitlines() if "| llama |" in line)
        self.assertEqual(unstable_row.split("|")[11].strip(), "NO")

    def test_unstable_output_suppresses_comparison_deltas(self) -> None:
        reports = [
            self.item("luce-k8", 20.0, repeat=1, output_hash="luce-a"),
            self.item("luce-k8", 22.0, repeat=2, output_hash="luce-b"),
            self.item("llama", 10.0, repeat=1, output_hash="llama"),
            self.item("llama", 11.0, repeat=2, output_hash="llama"),
        ]
        row = next(
            line for line in summarizer.summarize(reports).splitlines()
            if "| luce-k8 |" in line
        )
        self.assertEqual(row.split("|")[12].strip(), "n/a")
        self.assertEqual(row.split("|")[13].strip(), "n/a")

        peer_unstable = [
            self.item("luce-k8", 20.0, repeat=1, output_hash="luce"),
            self.item("luce-k8", 22.0, repeat=2, output_hash="luce"),
            self.item("llama", 10.0, repeat=1, output_hash="llama-a"),
            self.item("llama", 11.0, repeat=2, output_hash="llama-b"),
        ]
        peer_row = next(
            line for line in summarizer.summarize(peer_unstable).splitlines()
            if "| luce-k8 |" in line
        )
        self.assertEqual(peer_row.split("|")[12].strip(), "n/a")
        self.assertEqual(peer_row.split("|")[13].strip(), "n/a")

    def test_summary_reports_ttft_median_and_max(self) -> None:
        text = summarizer.summarize([self.item("llama", 8.0)])
        self.assertIn("TTFT median s | TTFT max s", text)
        row = next(line for line in text.splitlines() if "| llama |" in line)
        self.assertEqual(row.split("|")[9].strip(), "1.000")
        self.assertEqual(row.split("|")[10].strip(), "2.000")

    def test_summary_reports_optional_native_prefill_metrics(self) -> None:
        luce = self.item("luce-k8", 20.0)
        luce["level"].update({
            "server_native_prefilled_tokens_total": 4096,
            "server_native_prefill_window_ms": 1234.5,
            "server_native_prefill_token_count_complete": True,
            "server_native_prefill_timing_complete": True,
            "server_native_prefill_tokens_per_s": 321.25,
        })
        text = summarizer.summarize([luce, self.item("llama", 8.0)])
        self.assertIn("Native prefill tok/s | Native prefill window ms", text)
        luce_row = next(
            line for line in text.splitlines() if "| luce-k8 |" in line
        )
        llama_row = next(
            line for line in text.splitlines() if "| llama |" in line
        )
        self.assertEqual(luce_row.split("|")[-3].strip(), "321.25")
        self.assertEqual(luce_row.split("|")[-2].strip(), "1234.5")
        self.assertEqual(llama_row.split("|")[-3].strip(), "n/a")
        self.assertEqual(llama_row.split("|")[-2].strip(), "n/a")

    def test_summary_rejects_mismatched_native_completeness_flags(self) -> None:
        item = self.item("luce-k8", 20.0)
        item["level"].update({
            "server_native_prefill_token_count_complete": True,
            "server_native_prefill_timing_complete": False,
        })
        with self.assertRaisesRegex(ValueError, "mismatched native prefill"):
            summarizer.summarize([item])

    def test_summary_rejects_partial_native_repeats(self) -> None:
        complete = self.item("luce-k8", 20.0, repeat=1)
        complete["level"].update({
            "server_native_prefill_window_ms": 100.0,
            "server_native_prefill_tokens_per_s": 200.0,
            "server_native_prefill_token_count_complete": True,
            "server_native_prefill_timing_complete": True,
        })
        missing = self.item("luce-k8", 21.0, repeat=2)
        with self.assertRaisesRegex(ValueError, "partial native prefill telemetry"):
            summarizer.summarize([complete, missing])

    def test_load_reports_rejects_missing_prompt_usage(self) -> None:
        report = {
            "ignore_eos": True,
            "server_metadata": {"workload": "short", "variant": "llama", "repeat": 1},
            "levels": [{
                "failures": 0,
                "token_count_complete": True,
                "prompt_token_count_complete": False,
                "fixed_token_workload_valid": True,
            }],
        }
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "bench.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "incomplete token accounting"):
                summarizer.load_reports(Path(root))

    def test_load_reports_requires_fixed_token_protocol(self) -> None:
        report = {
            "ignore_eos": False,
            "server_metadata": {"workload": "short", "variant": "llama", "repeat": 1},
            "levels": [{
                "failures": 0,
                "token_count_complete": True,
                "prompt_token_count_complete": True,
                "fixed_token_workload_valid": None,
            }],
        }
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "bench.json"
            path.write_text(json.dumps(report), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "fixed-token protocol"):
                summarizer.load_reports(Path(root))


if __name__ == "__main__":
    unittest.main()
