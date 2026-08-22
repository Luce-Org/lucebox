from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from coding_tools import execute_tool, format_tool_result


class CodingToolsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        (self.root / "src").mkdir()
        (self.root / "src" / "cache.py").write_text(
            "MAX_ENTRIES = 64\n\ndef should_refresh(age):\n    return age > 300\n",
            encoding="utf-8",
        )
        (self.root / "deps").mkdir()
        (self.root / "deps" / "vendored.py").write_text(
            "SHOULD_NOT_BE_SCANNED = True\n", encoding="utf-8"
        )
        (self.root / "README.md").write_text("fixture repository\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_read_search_and_list_are_result_bearing(self) -> None:
        read = execute_tool("read_file", {"path": "src/cache.py", "start_line": 1, "end_line": 2}, self.root)
        self.assertTrue(read["ok"])
        self.assertIn("MAX_ENTRIES = 64", read["value"]["content"])

        search = execute_tool("search_code", {"query": "should_refresh", "path": "src"}, self.root)
        self.assertTrue(search["ok"])
        self.assertEqual(search["value"]["matches"][0]["path"], "src/cache.py")

        listed = execute_tool("list_files", {"path": ".", "glob": "*.py"}, self.root)
        self.assertTrue(listed["ok"])
        self.assertEqual(listed["value"]["files"], ["src/cache.py"])

        default_listing = execute_tool("list_files", {}, self.root)
        self.assertTrue(default_listing["ok"])
        self.assertEqual(
            default_listing["value"]["files"], ["README.md", "src/cache.py"]
        )

        empty_optional_glob = execute_tool("list_files", {"glob": ""}, self.root)
        self.assertTrue(empty_optional_glob["ok"])
        self.assertEqual(
            empty_optional_glob["value"]["files"], ["README.md", "src/cache.py"]
        )

        invalid_glob = execute_tool("list_files", {"glob": False}, self.root)
        self.assertFalse(invalid_glob["ok"])

    def test_paths_cannot_escape_workspace(self) -> None:
        outside = self.root.parent / "outside-coding-tool.txt"
        outside.write_text("secret\n", encoding="utf-8")
        try:
            escaped = execute_tool("read_file", {"path": "../outside-coding-tool.txt"}, self.root)
            self.assertFalse(escaped["ok"])
            if hasattr(os, "symlink"):
                (self.root / "escape").symlink_to(outside)
                linked = execute_tool("read_file", {"path": "escape"}, self.root)
                self.assertFalse(linked["ok"])
        finally:
            outside.unlink(missing_ok=True)

    def test_write_and_shell_tools_are_not_available(self) -> None:
        for name in ("write_file", "apply_patch", "exec_command"):
            self.assertFalse(execute_tool(name, {}, self.root)["ok"])
        self.assertFalse(execute_tool(["read_file"], {}, self.root)["ok"])
        unknown = execute_tool(
            "read_file", {"path": "README.md", "command": "ignored?"}, self.root
        )
        self.assertFalse(unknown["ok"])
        self.assertIn("unknown arguments", unknown["error"])

    def test_tool_content_matches_server_string_semantics(self) -> None:
        self.assertEqual(format_tool_result({"ok": True, "value": "plain"}), "plain")
        self.assertEqual(
            format_tool_result({"ok": True, "value": {"matches": []}}),
            '{"matches":[]}',
        )

    def test_executor_rejects_an_unisolated_request(self) -> None:
        script = Path(__file__).with_name("coding_tool_executor.py")
        request = {
            "protocol": "dflash.tool-speculation.v1",
            "request_id": "test-unisolated",
            "cpu_affinity": [],
            "call": {"name": "read_file", "arguments": {"path": "README.md"}},
        }
        environment = os.environ.copy()
        environment["DFLASH_TOOL_WORKSPACE"] = str(self.root)
        completed = subprocess.run(
            [sys.executable, str(script), "--dflash-tool-spec-v1"],
            input=json.dumps(request) + "\n",
            text=True,
            capture_output=True,
            env=environment,
            check=False,
            timeout=5,
        )
        self.assertEqual(completed.returncode, 2)
        self.assertEqual(completed.stdout, "")
        self.assertIn("non-empty isolated CPU affinity", completed.stderr)

    @unittest.skipUnless(
        hasattr(os, "sched_getaffinity"), "CPU affinity verification requires Linux"
    )
    def test_executor_protocol_round_trip(self) -> None:
        script = Path(__file__).with_name("coding_tool_executor.py")
        request = {
            "protocol": "dflash.tool-speculation.v1",
            "request_id": "test",
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "call": {"name": "search_code", "arguments": {"query": "MAX_ENTRIES"}},
        }
        environment = os.environ.copy()
        environment["DFLASH_TOOL_WORKSPACE"] = str(self.root)
        completed = subprocess.run(
            [sys.executable, str(script), "--dflash-tool-spec-v1"],
            input=json.dumps(request) + "\n",
            text=True,
            capture_output=True,
            env=environment,
            check=False,
            timeout=5,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        envelope = json.loads(completed.stdout)
        self.assertTrue(envelope["ok"])
        self.assertEqual(envelope["result"]["tool_name"], "search_code")

    def test_summary_refuses_an_unproven_performance_claim(self) -> None:
        artifact_path = self.root / "benchmark.json"
        base = {
            "task": "one",
            "repetition": 0,
            "correct": True,
            "calls": 1,
            "wall_ms": 100.0,
            "eligible_followup_turns": 1,
            "eligible_backend_cache_hits": 0,
            "agent_turn_cache_hits": 0,
            "unexpected_agent_turn_cache_hits": 0,
            "eligible_prefill_ms": 10.0,
            "eligible_model_wall_ms": 20.0,
            "eligible_prefilled_tokens": 100,
            "eligible_effective_prompt_tokens": 100,
            "timing_records_complete": True,
            "trace_sha256": "a" * 64,
            "assistant_trace_sha256": "b" * 64,
            "turn_log": [
                {
                    "cache_eligible": True,
                    "timings": {
                        "cache_hit": False,
                        "effective_prompt_tokens": 100,
                    },
                }
            ],
        }
        artifact_path.write_text(
            json.dumps(
                {
                    "schema": "lucebox.agent-turn-cache-benchmark.v1",
                    "task_count": 1,
                    "task_ids": ["one"],
                    "repetitions": 1,
                    "results": [
                        {**base, "arm": "control"},
                        {
                            **base,
                            "arm": "cache",
                            "wall_ms": 80.0,
                            "eligible_prefill_ms": 8.0,
                            "eligible_model_wall_ms": 16.0,
                            "eligible_prefilled_tokens": 80,
                        },
                    ],
                }
            ),
            encoding="utf-8",
        )
        summary = Path(__file__).with_name("coding_summary.py")
        completed = subprocess.run(
            [
                sys.executable,
                str(summary),
                str(artifact_path),
                "--bootstrap-samples",
                "20",
                "--minimum-pairs",
                "1",
            ],
            text=True,
            capture_output=True,
            check=False,
            timeout=5,
        )
        self.assertEqual(completed.returncode, 1, completed.stderr)
        self.assertIn("publication gate failed", completed.stdout)
        self.assertIn("all_eligible_cache_turns_hit: FAIL", completed.stdout)


if __name__ == "__main__":
    unittest.main()
