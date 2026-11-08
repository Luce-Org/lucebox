"""Regression tests for usage token counts in generation_benchmark.run_case."""

from __future__ import annotations

import sys
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


class TokenCountTests(unittest.TestCase):
    def test_token_count_accepts_non_negative_ints(self) -> None:
        self.assertEqual(generation_benchmark._token_count(0), 0)
        self.assertEqual(generation_benchmark._token_count(42), 42)

    def test_token_count_rejects_bad_values(self) -> None:
        for value in (True, False, -1, 3.0, "7", None, [], {}):
            with self.subTest(value=value):
                self.assertIsNone(generation_benchmark._token_count(value))


class RunCaseUsageTests(unittest.TestCase):
    def test_valid_usage_is_reported(self) -> None:
        run = _run({"completion_tokens": 12, "prompt_tokens": 5})
        self.assertEqual(run["completion_tokens"], 12)
        self.assertEqual(run["prompt_tokens"], 5)
        self.assertEqual(run["token_count_source"], "usage")

    def test_bool_completion_tokens_fall_back_to_word_count(self) -> None:
        run = _run({"completion_tokens": True, "prompt_tokens": 5})
        self.assertEqual(run["completion_tokens"], 4)
        self.assertEqual(run["token_count_source"], "approx_words")

    def test_negative_completion_tokens_fall_back_to_word_count(self) -> None:
        run = _run({"completion_tokens": -3})
        self.assertEqual(run["completion_tokens"], 4)
        self.assertEqual(run["token_count_source"], "approx_words")

    def test_zero_completion_tokens_fall_back_to_word_count(self) -> None:
        run = _run({"completion_tokens": 0})
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


if __name__ == "__main__":
    unittest.main()
