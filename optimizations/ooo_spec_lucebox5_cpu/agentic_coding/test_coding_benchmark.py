from __future__ import annotations

import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import coding_bench
import coding_summary


class CodingBenchmarkTests(unittest.TestCase):
    def test_bundled_task_files_are_valid(self) -> None:
        directory = Path(__file__).resolve().parent
        for name in ("tasks.json", "tasks_lucebox.json"):
            with (directory / name).open(encoding="utf-8") as source:
                tasks = coding_bench.validate_tasks(json.load(source))
            self.assertTrue(tasks)

    def test_run_task_isolates_only_agent_turn_cache(self) -> None:
        task = {
            "id": "demo",
            "prompt": "Find the answer.",
            "answer_terms": ["answer"],
            "evidence_terms": ["needle"],
        }
        args = SimpleNamespace(
            tag="unit",
            run_id="unit-run",
            model="model",
            max_turns=2,
            max_tokens=32,
            timeout=5,
            url="http://unused",
            workspace=Path.cwd(),
        )

        def run(arm: str) -> tuple[dict, list[dict]]:
            bodies: list[dict] = []

            def fake_post(url: str, body: dict, timeout: int):
                del url, timeout
                bodies.append(body)
                turn = len(bodies) - 1
                hit = arm == "cache" and turn == 1
                message = (
                    {
                        "role": "assistant",
                        "content": "",
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "read_file",
                                    "arguments": '{"path":"README.md"}',
                                },
                            }
                        ],
                    }
                    if turn == 0
                    else {"role": "assistant", "content": "The answer."}
                )
                return {
                    "choices": [{"message": message}],
                    "usage": {
                        "timings": {
                            "cache_hit": turn == 1,
                            "agent_turn_cache_hit": hit,
                            "prefill_ms": 2.0 if hit else 10.0,
                            "prefilled_tokens": 4 if hit else 100,
                            "effective_prompt_tokens": 100,
                        }
                    },
                }, 12.0 if hit else 20.0

            with (
                patch.object(coding_bench, "post", side_effect=fake_post),
                patch.object(
                    coding_bench,
                    "execute_tool",
                    return_value={"ok": True, "value": "needle"},
                ),
            ):
                return coding_bench.run_task(args, task, arm, 0), bodies

        control, control_bodies = run("control")
        cached, cache_bodies = run("cache")

        self.assertTrue(control["correct"])
        self.assertTrue(cached["correct"])
        self.assertEqual(control["trace_sha256"], cached["trace_sha256"])
        self.assertEqual(
            control["assistant_trace_sha256"], cached["assistant_trace_sha256"]
        )
        self.assertEqual(control["eligible_followup_turns"], 1)
        self.assertEqual(control["eligible_backend_cache_hits"], 1)
        self.assertEqual(cached["eligible_backend_cache_hits"], 1)
        self.assertEqual(control["agent_turn_cache_hits"], 0)
        self.assertEqual(cached["agent_turn_cache_hits"], 1)
        self.assertGreater(
            control["eligible_prefilled_tokens"],
            cached["eligible_prefilled_tokens"],
        )
        self.assertTrue(all(not body["automatic_tool_speculation"] for body in control_bodies))
        self.assertTrue(all(body["agent_turn_cache"] is False for body in control_bodies))
        self.assertTrue(all(body["agent_turn_cache"] is True for body in cache_bodies))
        self.assertEqual([body["tool_choice"] for body in control_bodies], ["required", "auto"])
        self.assertEqual([body["tool_choice"] for body in cache_bodies], ["required", "auto"])
        self.assertEqual(control["pair_id"], cached["pair_id"])
        for control_body, cache_body in zip(control_bodies, cache_bodies, strict=True):
            self.assertEqual(
                {key: value for key, value in control_body.items() if key != "agent_turn_cache"},
                {key: value for key, value in cache_body.items() if key != "agent_turn_cache"},
            )

    def test_pair_identifier_is_stable_and_isolates_repetitions(self) -> None:
        self.assertEqual(
            coding_bench.pair_identifier("run", "task", 0),
            coding_bench.pair_identifier("run", "task", 0),
        )
        self.assertNotEqual(
            coding_bench.pair_identifier("run", "task", 0),
            coding_bench.pair_identifier("run", "task", 1),
        )
        self.assertNotEqual(
            coding_bench.pair_identifier("run", "task", 0),
            coding_bench.pair_identifier("run", "other", 0),
        )
        self.assertNotEqual(
            coding_bench.pair_identifier("run-one", "task", 0),
            coding_bench.pair_identifier("run-two", "task", 0),
        )

    def test_server_props_url_uses_request_origin(self) -> None:
        response = MagicMock()
        response.read.return_value = b'{"prefix_cache":{"capacity":32}}'
        response.__enter__.return_value = response
        with patch.object(coding_bench.urllib.request, "urlopen", return_value=response) as open_url:
            props_url, props = coding_bench.fetch_server_props(
                "http://127.0.0.1:18145/v1/chat/completions", 5
            )
        self.assertEqual(props_url, "http://127.0.0.1:18145/props")
        self.assertEqual(props["prefix_cache"]["capacity"], 32)
        open_url.assert_called_once_with(props_url, timeout=5)

    def test_validation_allows_a_recovered_tool_error(self) -> None:
        task = {
            "answer_terms": ["64"],
            "evidence_terms": ["DEFAULT_MAX_ENTRIES = 64"],
        }
        validation = coding_bench.validate(
            task,
            "The value is 64.",
            [
                {"ok": False, "error": "bad exploratory glob"},
                {"ok": True, "content": "DEFAULT_MAX_ENTRIES = 64"},
            ],
        )

        self.assertTrue(validation["ok"])
        self.assertEqual(validation["successful_tool_calls"], 1)
        self.assertEqual(validation["failed_tool_calls"], 1)
        self.assertFalse(validation["all_tools_succeeded"])

    def test_validation_requires_evidence_from_a_successful_tool(self) -> None:
        task = {
            "answer_terms": ["64"],
            "evidence_terms": ["DEFAULT_MAX_ENTRIES = 64"],
        }
        validation = coding_bench.validate(
            task,
            "The value is 64.",
            [
                {"ok": False, "error": "DEFAULT_MAX_ENTRIES = 64"},
                {"ok": True, "content": "unrelated evidence"},
            ],
        )

        self.assertFalse(validation["ok"])
        self.assertEqual(
            validation["missing_evidence_terms"], ["DEFAULT_MAX_ENTRIES = 64"]
        )

    def test_eligible_prompt_shape_uses_only_followups(self) -> None:
        result = {
            "turn_log": [
                {
                    "cache_eligible": False,
                    "timings": {"effective_prompt_tokens": 10},
                },
                {
                    "cache_eligible": True,
                    "timings": {"effective_prompt_tokens": 25},
                },
            ]
        }
        self.assertEqual(coding_summary.eligible_prompt_shape(result), [25])

    def test_backend_cache_hits_are_distinct_from_agent_turn_hits(self) -> None:
        result = {
            "eligible_backend_cache_hits": 2,
            "agent_turn_cache_hits": 0,
            "turn_log": [],
        }
        self.assertEqual(coding_summary.eligible_backend_cache_hit_count(result), 2)

        legacy_result = {
            "turn_log": [
                {"cache_eligible": False, "timings": {"cache_hit": False}},
                {"cache_eligible": True, "timings": {"cache_hit": True}},
                {"cache_eligible": True, "timings": {"cache_hit": False}},
            ]
        }
        self.assertEqual(
            coding_summary.eligible_backend_cache_hit_count(legacy_result), 1
        )

    def test_tool_trace_ignores_equivalent_final_answer_wording(self) -> None:
        call = {
            "name": "read_file",
            "arguments": {"path": "README.md"},
            "result": "evidence",
        }
        left = [
            {"assistant_content": "I'll inspect it.", "tool_calls": [call]},
            {"assistant_content": "The value is 64.", "tool_calls": []},
        ]
        right = [
            {"assistant_content": "Checking.", "tool_calls": [call]},
            {"assistant_content": "It is **64**.", "tool_calls": []},
        ]
        self.assertEqual(
            coding_bench._tool_trace_sha256(left),
            coding_bench._tool_trace_sha256(right),
        )
        self.assertNotEqual(
            coding_bench._trace_sha256(left), coding_bench._trace_sha256(right)
        )

    def test_full_transcript_hash_includes_reasoning(self) -> None:
        left = [{
            "assistant_reasoning": "Inspect cache.cpp.",
            "assistant_content": "",
            "tool_calls": [],
        }]
        right = [{
            "assistant_reasoning": "Inspect runner.cpp.",
            "assistant_content": "",
            "tool_calls": [],
        }]
        self.assertNotEqual(
            coding_bench._trace_sha256(left), coding_bench._trace_sha256(right)
        )

    def test_bootstrap_uses_requested_followup_metric(self) -> None:
        pairs = [
            {
                "control": {"eligible_prefill_ms": 20.0},
                "cache": {"eligible_prefill_ms": 5.0},
            }
        ]
        self.assertEqual(
            coding_summary.bootstrap_aggregate_ci(
                pairs, 20, "eligible_prefill_ms"
            ),
            (4.0, 4.0),
        )


if __name__ == "__main__":
    unittest.main()
