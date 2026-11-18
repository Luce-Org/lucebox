"""Regression tests for generation benchmark metrics and report comparisons."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "harness" / "benchmarks"))

import generation_benchmark  # noqa: E402

CASE = {"id": "case-1", "prompt": "Say hello."}
TEXT = "one two three four"


def _response(usage: Any) -> dict[str, Any]:
    response: dict[str, Any] = {
        "choices": [{"message": {"role": "assistant", "content": TEXT}}]
    }
    if usage is not None:
        response["usage"] = usage
    return response


def _run(usage: Any) -> dict[str, Any]:
    with mock.patch.object(
        generation_benchmark, "post_chat", return_value=_response(usage)
    ):
        result = generation_benchmark.run_case(
            CASE,
            base_url="http://127.0.0.1:1/v1",
            api_key="",
            model="test",
            max_tokens=16,
            temperature=0.0,
            timeout=1.0,
            repeats=1,
        )
    return result["runs"][0]


class RunCaseUsageTests(unittest.TestCase):
    def test_valid_usage_is_reported(self) -> None:
        run = _run({"completion_tokens": 12, "prompt_tokens": 5})
        self.assertEqual(run["completion_tokens"], 12)
        self.assertEqual(run["prompt_tokens"], 5)
        self.assertEqual(run["token_count_source"], "usage")
        # A zero prompt count is a valid count, not a bad value.
        self.assertEqual(_run({"completion_tokens": 12, "prompt_tokens": 0})["prompt_tokens"], 0)

    def test_bad_completion_tokens_fall_back_to_word_count(self) -> None:
        # bool is an int subclass; negative is invalid; zero carries no count.
        for value in (True, -3, 0):
            with self.subTest(value=value):
                run = _run({"completion_tokens": value, "prompt_tokens": 5})
                self.assertEqual(run["completion_tokens"], 4)
                self.assertEqual(run["token_count_source"], "approx_words")

    def test_invalid_prompt_tokens_are_dropped(self) -> None:
        for value in (True, -1, "5", 5.0):
            with self.subTest(value=value):
                run = _run({"completion_tokens": 12, "prompt_tokens": value})
                self.assertIsNone(run["prompt_tokens"])

    def test_non_dict_usage_does_not_crash(self) -> None:
        for usage in (None, [], "oops", 7):
            with self.subTest(usage=usage):
                run = _run(usage)
                self.assertEqual(run["completion_tokens"], 4)
                self.assertIsNone(run["prompt_tokens"])
                self.assertEqual(run["token_count_source"], "approx_words")


class CompareReportTests(unittest.TestCase):
    def setUp(self) -> None:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name)
        self.json_out = self.directory / "compare.json"
        self.md_out = self.directory / "report.md"

    def compare(self, baseline: list[dict], candidate: list[dict]) -> subprocess.CompletedProcess:
        for name, cases in (("baseline", baseline), ("candidate", candidate)):
            (self.directory / f"{name}.json").write_text(
                json.dumps({"name": name, "cases": cases}), encoding="utf-8"
            )
        return subprocess.run(
            [
                sys.executable, str(Path(generation_benchmark.__file__)), "compare",
                "--baseline", str(self.directory / "baseline.json"),
                "--candidate", str(self.directory / "candidate.json"),
                "--json-out", str(self.json_out), "--md-out", str(self.md_out),
            ],
            capture_output=True, text=True, check=False,
        )

    @staticmethod
    def case(case_id: str, passed: bool = True) -> dict:
        return {"id": case_id, "mean_tok_s": 10.0, "expected_pass": passed, "text": "ok"}

    def assert_invalid(self, result: subprocess.CompletedProcess, message: str) -> None:
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn(message, result.stderr)
        self.assertFalse(self.json_out.exists())
        self.assertFalse(self.md_out.exists())

    def test_rejects_mismatched_case_sets(self) -> None:
        for baseline, candidate in (
            (["a", "b"], ["a"]),
            (["a"], ["a", "b"]),
            (["a"], ["b"]),
        ):
            with self.subTest(baseline=baseline, candidate=candidate):
                result = self.compare(
                    [self.case(key) for key in baseline],
                    [self.case(key) for key in candidate],
                )
                self.assert_invalid(result, "case IDs differ")
                self.assertIn(
                    f"missing from candidate: {sorted(set(baseline) - set(candidate))}",
                    result.stderr,
                )
                self.assertIn(
                    f"missing from baseline: {sorted(set(candidate) - set(baseline))}",
                    result.stderr,
                )

    def test_rejects_empty_reports(self) -> None:
        self.assert_invalid(self.compare([], []), "at least one case")

    def test_rejects_duplicate_ids_in_either_report(self) -> None:
        case = self.case("a")
        for baseline, candidate in (([case, case], [case]), ([case], [case, case])):
            with self.subTest(baseline=baseline, candidate=candidate):
                self.assert_invalid(self.compare(baseline, candidate), "duplicate case id: 'a'")

    def test_compares_complete_reports_regardless_of_order(self) -> None:
        result = self.compare(
            [self.case("b"), self.case("a")], [self.case("a"), self.case("b")]
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(self.json_out.read_text(encoding="utf-8"))
        self.assertEqual([case["id"] for case in report["cases"]], ["a", "b"])
        self.assertEqual(report["summary"]["candidate_expected_pass"], 2)
        self.assertEqual(report["summary"]["mean_speedup"], 1.0)
        self.assertIn("2/2", self.md_out.read_text(encoding="utf-8"))

    def test_preserves_failed_expected_checks(self) -> None:
        result = self.compare([self.case("a")], [self.case("a", passed=False)])
        self.assertEqual(result.returncode, 1, result.stderr)
        report = json.loads(self.json_out.read_text(encoding="utf-8"))
        self.assertEqual(report["summary"]["candidate_expected_pass"], 0)


if __name__ == "__main__":
    unittest.main()
