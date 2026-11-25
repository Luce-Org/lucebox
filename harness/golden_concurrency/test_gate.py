#!/usr/bin/env python3
"""Fast (< 3s) CPU-only unittest suite for the Golden Concurrency Gate."""

from __future__ import annotations

import http.server
import json
import re
import socketserver
import tempfile
import threading
import time
import unittest
from pathlib import Path
from typing import Any
from unittest import mock

import sys

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import gate  # noqa: E402


CANARY_RE = re.compile(r"REQUIRED_BUILD_TOKEN\s*=\s*(CANARY-L\d{2}-[0-9A-F]{8})")


class MockFaultConfig:
    def __init__(self) -> None:
        self.inject_sibling_canary: str | None = None
        self.inject_sibling_in_reasoning: bool = False
        self.force_cached_tokens: int | None = None
        self.oversized_status: int = 400
        self.fail_http_500_on_burst: bool = False
        self.corrupt_tool_call: bool = False
        self.leak_slots_busy: int = 0
        self.leak_kv_tokens: int = 0
        self.spec_mean_accepted: float | None = None
        self.p1_delay_s: float = 0.004
        self.p2_delay_s: float = 0.0005
        self.decode_chunk_delay_s: float = 0.0
        self.oversize_threshold_chars: int = 50000
        self.log_path: Path | None = None
        self.log_spec_steps: int = 0
        self.log_spec_accepted: int = 0
        self.omit_log_metrics: bool = False


class MockOpenAIServer:
    """In-process threaded OpenAI-compatible SSE server with fault injection."""

    def __init__(self, faults: MockFaultConfig | None = None) -> None:
        self.faults = faults or MockFaultConfig()
        self._lock = threading.Lock()
        self._req_counter = 0
        self._active_slots = 0
        self._seen_canaries: list[str] = []
        self.aborts_observed = 0
        self._expected_tools = self._load_expected_tools()
        self._httpd = self._build_server()
        self._thread = threading.Thread(
            target=self._httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True
        )

    @staticmethod
    def _load_expected_tools() -> dict[str, tuple[str, dict[str, Any]]]:
        mapping: dict[str, tuple[str, dict[str, Any]]] = {}
        wl_dir = HERE / "v1" / "workloads"
        for rec in gate.load_jsonl(wl_dir / "agentic_2turn_smoke.jsonl"):
            key = rec["turn2_followup"]["content"]
            mapping[key] = (rec["expected_tool_name"], rec["expected_arg_values"])
        for rec in gate.load_jsonl(wl_dir / "tool_calls_golden.jsonl"):
            key = rec["messages"][-1]["content"]
            args = dict(rec["expected_arg_values"])
            # Fill any additional required keys needed by strict schema checks
            if rec["expected_tool_name"] == "apply_structured_patch":
                args["metadata"] = {"author": "ci-bot", "priority": 2}
                args["hunks"] = [{"line": 42, "old_text": "return false;", "new_text": "return true;"}]
            elif rec["expected_tool_name"] == "run_check":
                args["env"] = {"CTEST_OUTPUT_ON_FAILURE": "1"}
            elif rec["expected_tool_name"] == "find_regex":
                args["regex"] = '"status":\\s*500'
                args["max_matches"] = 10
            elif rec["expected_tool_name"] == "configure_kv_pool":
                args["headroom_ratio"] = 0.15
            elif rec["expected_tool_name"] == "register_wave_plan":
                args["tags"] = ["isolation", "prefix-cache"]
            mapping[key] = (rec["expected_tool_name"], args)
        return mapping

    def _next_id(self) -> str:
        with self._lock:
            self._req_counter += 1
            return f"chatcmpl-mock-{self._req_counter:04d}"

    def _build_server(self) -> socketserver.ThreadingTCPServer:
        parent = self

        class ThreadingServer(socketserver.ThreadingMixIn, http.server.HTTPServer):
            daemon_threads = True
            allow_reuse_address = True

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: Any) -> None:
                pass

            def _send_json(self, code: int, obj: dict[str, Any]) -> None:
                raw = (json.dumps(obj) + "\n").encode("utf-8")
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(raw)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(raw)

            def do_GET(self) -> None:
                if self.path == "/health":
                    self._send_json(200, {"status": "ok"})
                    return
                if self.path == "/props":
                    with parent._lock:
                        busy = parent._active_slots + parent.faults.leak_slots_busy
                    kv_total = 32768
                    kv_free = max(0, kv_total - parent.faults.leak_kv_tokens)
                    self._send_json(
                        200,
                        {
                            "n_ctx": 8192,
                            "load": {
                                "slots": 10,
                                "slots_busy": busy,
                                "waiting": 0,
                                "queue_limit": 16,
                                "kv_tokens": kv_total,
                                "kv_free_tokens": kv_free,
                            },
                        },
                    )
                    return
                self._send_json(404, {"error": "not found"})

            def do_POST(self) -> None:
                if not self.path.endswith("/chat/completions"):
                    self._send_json(404, {"error": "unknown endpoint"})
                    return
                length = int(self.headers.get("Content-Length", "0"))
                try:
                    raw_body = self.rfile.read(length)
                    req = json.loads(raw_body.decode("utf-8"))
                except Exception:
                    self._send_json(400, {"error": "bad json"})
                    return

                messages = req.get("messages") or []
                total_chars = sum(len(str(m.get("content", ""))) for m in messages)
                if total_chars >= parent.faults.oversize_threshold_chars:
                    if parent.faults.oversized_status != 200:
                        self._send_json(parent.faults.oversized_status, {"error": "prompt exceeds context"})
                        return

                with parent._lock:
                    parent._active_slots += 1
                    concurrent_now = parent._active_slots

                try:
                    if parent.faults.fail_http_500_on_burst and concurrent_now > 3:
                        self._send_json(500, {"error": "simulated OOM 500"})
                        return

                    resp_id = parent._next_id()
                    last_user = str(messages[-1].get("content", "")) if messages else ""
                    is_turn2 = len(messages) >= 4
                    prompt_tokens = max(32, total_chars // 4)

                    if is_turn2:
                        time.sleep(parent.faults.p2_delay_s)
                        if parent.faults.force_cached_tokens is not None:
                            cached_tokens = parent.faults.force_cached_tokens
                        else:
                            # Simulate ~94% warm prefix cache hit relative to Turn 1
                            p1_chars = sum(len(str(m.get("content", ""))) for m in messages[:2])
                            cached_tokens = int((p1_chars // 4) * 0.96)
                    else:
                        time.sleep(parent.faults.p1_delay_s)
                        cached_tokens = 0

                    # Build response tokens (canary, tool call, or Turn-1 summary)
                    tool_call_obj: dict[str, Any] | None = None
                    content_chunks: list[str] = []
                    reasoning_chunks: list[str] = []

                    m_canary = CANARY_RE.search(last_user)
                    if last_user in parent._expected_tools:
                        tname, targs = parent._expected_tools[last_user]
                        if parent.faults.corrupt_tool_call:
                            # Drop -i or corrupt JSON
                            raw_args = '{"broken_json": '
                        else:
                            raw_args = json.dumps(targs)
                        tool_call_obj = {
                            "index": 0,
                            "id": f"call_{resp_id}",
                            "type": "function",
                            "function": {"name": tname, "arguments": raw_args},
                        }
                    elif m_canary:
                        own = m_canary.group(1)
                        content_chunks = ["Token ", "is ", own, " ", "verified", ".", " Done", "."]
                        if parent.faults.inject_sibling_canary:
                            sib = parent.faults.inject_sibling_canary
                            if sib != own:
                                if parent.faults.inject_sibling_in_reasoning:
                                    reasoning_chunks = [f"thinking leaked {sib}"]
                                else:
                                    content_chunks.append(f" leaked={sib}")
                    else:
                        content_chunks = ["Verified ", "concurrency ", "invariant ", "for ", "target ", "module."]

                    self.send_response(200)
                    self.send_header("Content-Type", "text/event-stream")
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Connection", "close")
                    self.end_headers()

                    def send_event(obj: Any) -> None:
                        data = obj if isinstance(obj, str) else json.dumps(obj)
                        self.wfile.write(f"data: {data}\n\n".encode("utf-8"))
                        self.wfile.flush()

                    for rc in reasoning_chunks:
                        send_event({"id": resp_id, "choices": [{"index": 0, "delta": {"reasoning_content": rc}, "finish_reason": None}]})
                        time.sleep(parent.faults.decode_chunk_delay_s)

                    if tool_call_obj is not None:
                        send_event({"id": resp_id, "choices": [{"index": 0, "delta": {"tool_calls": [tool_call_obj]}, "finish_reason": None}]})
                        finish_reason = "tool_calls"
                        comp_tokens = 12
                    else:
                        for chunk in content_chunks:
                            send_event({"id": resp_id, "choices": [{"index": 0, "delta": {"content": chunk}, "finish_reason": None}]})
                            time.sleep(parent.faults.decode_chunk_delay_s)
                        finish_reason = "stop"
                        comp_tokens = max(1, len(content_chunks))

                    send_event({"id": resp_id, "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}]})
                    timings: dict[str, Any] = {
                        "prefill_ms": round(parent.faults.p2_delay_s * 1000.0 if is_turn2 else parent.faults.p1_delay_s * 1000.0, 1),
                        "decode_ms": 10.0,
                        "cached_prefix_tokens": cached_tokens,
                        "prefilled_tokens": max(1, prompt_tokens - cached_tokens),
                    }
                    if parent.faults.spec_mean_accepted is not None:
                        timings["mean_accepted_len"] = parent.faults.spec_mean_accepted
                    usage = {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": comp_tokens,
                        "total_tokens": prompt_tokens + comp_tokens,
                        "prompt_tokens_details": {"cached_tokens": cached_tokens},
                        "timings": timings,
                    }
                    send_event({"id": resp_id, "choices": [], "usage": usage})
                    send_event("[DONE]")

                    if parent.faults.log_path is not None and not parent.faults.omit_log_metrics:
                        rec = {
                            "response_id": resp_id,
                            "ddtree_steps": parent.faults.log_spec_steps,
                            "ddtree_accepted_tokens": parent.faults.log_spec_accepted,
                            "target_forwards": parent.faults.log_spec_steps,
                        }
                        with parent._lock:
                            with parent.faults.log_path.open("a", encoding="utf-8") as lfh:
                                lfh.write(f"[concurrency-metrics] {json.dumps(rec)}\n")
                except (BrokenPipeError, ConnectionResetError, OSError):
                    with parent._lock:
                        parent.aborts_observed += 1
                finally:
                    with parent._lock:
                        parent._active_slots = max(0, parent._active_slots - 1)

        return ThreadingServer(("127.0.0.1", 0), Handler)

    @property
    def base_url(self) -> str:
        host, port = self._httpd.server_address
        return f"http://{host}:{port}/v1"

    def __enter__(self) -> "MockOpenAIServer":
        self._thread.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()


class GoldenConcurrencyGateTests(unittest.TestCase):
    def _run(
        self,
        base_url: str,
        tmpdir: Path,
        extra_args: list[str] | None = None,
        concurrency: str = "1,3",
    ) -> tuple[int, dict[str, Any]]:
        argv = [
            "--pack", str(HERE / "v1" / "pack.json"),
            "--tier", "tier1_pr_smoke",
            "--model-profile", "qwen3.8-27b",
            "--base-url", base_url,
            "--concurrency-override", concurrency,
            "--oversized-chars", "55000",
            "--output-dir", str(tmpdir),
        ]
        if extra_args:
            argv.extend(extra_args)
        args = gate.build_arg_parser().parse_args(argv)
        with mock.patch.object(gate.MemorySampler, "read_amd_gpu_mem_mib", return_value=(None, None)):
            return gate.run_gate(args)

    def test_pack_schema_and_workloads_integrity(self) -> None:
        pack_path = HERE / "v1" / "pack.json"
        pack = json.loads(pack_path.read_text(encoding="utf-8"))
        self.assertEqual(pack["schema_version"], 1)
        self.assertEqual(pack["pack_id"], "lucebox-concurrency-golden-v1")
        self.assertTrue(pack["public_oss_trajectory_contract"]["public_safe_only"])
        self.assertFalse(pack["public_oss_trajectory_contract"]["private_customer_data_allowed"])

        for hw in ("r9700_32g", "strix_halo_98g"):
            self.assertIn(hw, pack["hardware_profiles"])
        for model in ("qwen3.8-27b", "qwen3.8-flash-next", "inkling-2-mini"):
            self.assertIn(model, pack["models"])
            self.assertGreaterEqual(pack["models"][model]["max_supported_concurrency"], 1)
        for tier in ("tier1_pr_smoke", "tier2_nightly_agentic", "tier3_release_soak"):
            self.assertIn(tier, pack["tiers"])

        # Canary fixture: 12 distinct lanes, all unique canaries present in prompts
        canaries = gate.load_jsonl(HERE / "v1" / "workloads" / "canary_niah_4k.jsonl")
        self.assertEqual(len(canaries), 12)
        seen_tokens = set()
        for idx, rec in enumerate(canaries):
            tok = rec["expected_canary"]
            self.assertRegex(tok, r"^CANARY-L\d{2}-[0-9A-F]{8}$")
            self.assertNotIn(tok, seen_tokens)
            seen_tokens.add(tok)
            self.assertIn(tok, rec["messages"][-1]["content"])
            self.assertEqual(rec["lane_index"], idx)

        # Agentic 2-turn fixture: 12 distinct sessions, >= 90% shared prefix
        sessions = gate.load_jsonl(HERE / "v1" / "workloads" / "agentic_2turn_smoke.jsonl")
        self.assertEqual(len(sessions), 12)
        self.assertEqual(len({s["id"] for s in sessions}), 12)
        for s in sessions:
            self.assertGreaterEqual(s["shared_prefix_ratio_estimate"], 0.90)
            self.assertTrue(s["expected_tool_name"])
            self.assertTrue(s["required_args"])

        # Tool calls golden fixture: includes -i param and nested JSON schema
        tools = gate.load_jsonl(HERE / "v1" / "workloads" / "tool_calls_golden.jsonl")
        self.assertGreaterEqual(len(tools), 6)
        has_dash_i = any("-i" in t["required_args"] for t in tools)
        has_nested = any("object" in t.get("arg_types", {}).values() for t in tools)
        self.assertTrue(has_dash_i)
        self.assertTrue(has_nested)

    def test_pass_end_to_end_and_scorecard_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as td, MockOpenAIServer() as srv:
            out = Path(td) / "out"
            code, sc = self._run(srv.base_url, out, concurrency="1,3,4")
            self.assertEqual(code, gate.EXIT_PASS, sc.get("failures"))
            self.assertEqual(sc["verdict"], "PASS")
            for inv_name, inv in sc["invariants"].items():
                self.assertTrue(inv["pass"], f"{inv_name} failed: {inv}")
            # CPU-only host mock reports null VRAM/UMA gracefully
            self.assertIsNone(sc["memory_hwm"]["r9700_vram_peak_mib"])
            self.assertIsNone(sc["memory_hwm"]["strix_uma_peak_mib"])
            self.assertTrue((out / "scorecard.json").is_file())
            self.assertTrue((out / "scorecard.md").is_file())
            parsed = json.loads((out / "scorecard.json").read_text(encoding="utf-8"))
            self.assertEqual(parsed["exit_code"], 0)
            self.assertGreaterEqual(sc["invariants"]["I6_cancellation"]["aborts_injected"], 2)

    def test_i1_sibling_canary_crosstalk_content_and_reasoning_fail(self) -> None:
        canaries = gate.load_jsonl(HERE / "v1" / "workloads" / "canary_niah_4k.jsonl")
        leaked_sibling = canaries[1]["expected_canary"]

        for in_reasoning in (False, True):
            with self.subTest(in_reasoning=in_reasoning):
                faults = MockFaultConfig()
                faults.inject_sibling_canary = leaked_sibling
                faults.inject_sibling_in_reasoning = in_reasoning
                with tempfile.TemporaryDirectory() as td, MockOpenAIServer(faults) as srv:
                    code, sc = self._run(srv.base_url, Path(td))
                    self.assertEqual(code, gate.EXIT_INVARIANT_FAIL)
                    self.assertFalse(sc["invariants"]["I1_isolation"]["pass"])
                    self.assertGreater(sc["invariants"]["I1_isolation"]["sibling_leaks"], 0)

    def test_i3_invalid_tool_call_fails(self) -> None:
        faults = MockFaultConfig()
        faults.corrupt_tool_call = True
        with tempfile.TemporaryDirectory() as td, MockOpenAIServer(faults) as srv:
            code, sc = self._run(srv.base_url, Path(td))
            self.assertEqual(code, gate.EXIT_INVARIANT_FAIL)
            self.assertFalse(sc["invariants"]["I3_quality_floor"]["pass"])
            self.assertEqual(sc["invariants"]["I3_quality_floor"]["tool_call_valid"], 0.0)

    def test_i4_prefix_cache_miss_fails(self) -> None:
        faults = MockFaultConfig()
        faults.force_cached_tokens = 0
        with tempfile.TemporaryDirectory() as td, MockOpenAIServer(faults) as srv:
            code, sc = self._run(srv.base_url, Path(td))
            self.assertEqual(code, gate.EXIT_SLO_OR_MEM_FAIL)
            self.assertFalse(sc["invariants"]["I4_prefix_cache"]["pass"])
            self.assertEqual(sc["invariants"]["I4_prefix_cache"]["p2_ram_hit_rate"], 0.0)

    def test_i5_oversized_non_400_and_http_500_fail(self) -> None:
        with self.subTest("oversized_returns_200_instead_of_400"):
            faults = MockFaultConfig()
            faults.oversized_status = 200
            with tempfile.TemporaryDirectory() as td, MockOpenAIServer(faults) as srv:
                code, sc = self._run(srv.base_url, Path(td))
                self.assertEqual(code, gate.EXIT_INVARIANT_FAIL)
                self.assertFalse(sc["invariants"]["I5_backpressure"]["pass"])
                self.assertEqual(sc["invariants"]["I5_backpressure"]["oversized_status"], 200)

        with self.subTest("oversubscription_returns_500"):
            faults = MockFaultConfig()
            faults.fail_http_500_on_burst = True
            with tempfile.TemporaryDirectory() as td, MockOpenAIServer(faults) as srv:
                code, sc = self._run(srv.base_url, Path(td))
                self.assertEqual(code, gate.EXIT_INVARIANT_FAIL)
                self.assertFalse(sc["invariants"]["I5_backpressure"]["pass"])
                self.assertGreater(sc["invariants"]["I5_backpressure"]["http_5xx"], 0)

    def test_i6_cancellation_socket_reset_and_slot_leak_detection(self) -> None:
        # Healthy cancellation is verified in test_pass_end_to_end_and_scorecard_artifacts;
        # here we verify that a dangling busy slot after socket resets fails I6 with exit 1.
        faults = MockFaultConfig()
        faults.leak_slots_busy = 1
        with tempfile.TemporaryDirectory() as td, MockOpenAIServer(faults) as srv:
            with mock.patch.object(gate, "DRAIN_TIMEOUT_S", 0.15):
                code, sc = self._run(srv.base_url, Path(td) / "out")
            self.assertEqual(code, gate.EXIT_INVARIANT_FAIL)
            self.assertFalse(sc["invariants"]["I6_cancellation"]["pass"])
            self.assertEqual(sc["invariants"]["I6_cancellation"]["slot_leak"], 1)

    def test_i7_spec_decode_metrics_log_pass_and_low_acceptance_fail(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "server.log"
            faults = MockFaultConfig()
            faults.log_path = log_path
            faults.log_spec_steps = 10
            faults.log_spec_accepted = 20  # mean_accepted_len = (20+10)/10 = 3.0 >= 2.2
            with MockOpenAIServer(faults) as srv:
                code, sc = self._run(srv.base_url, Path(td) / "pass_out", ["--server-log", str(log_path)])
                self.assertEqual(code, gate.EXIT_PASS, sc.get("failures"))
                self.assertTrue(sc["invariants"]["I7_spec_kvflash"]["pass"])
                self.assertAlmostEqual(sc["invariants"]["I7_spec_kvflash"]["mean_accepted_len"], 3.0, places=2)
                self.assertEqual(sc["invariants"]["I7_spec_kvflash"]["metrics_coverage"], 1.0)

        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "server_low.log"
            faults = MockFaultConfig()
            faults.log_path = log_path
            faults.log_spec_steps = 10
            faults.log_spec_accepted = 5  # mean_accepted_len = 1.5 < 2.2
            with MockOpenAIServer(faults) as srv:
                code, sc = self._run(srv.base_url, Path(td) / "fail_out", ["--server-log", str(log_path)])
                self.assertEqual(code, gate.EXIT_SLO_OR_MEM_FAIL)
                self.assertFalse(sc["invariants"]["I7_spec_kvflash"]["pass"])


if __name__ == "__main__":
    unittest.main()
