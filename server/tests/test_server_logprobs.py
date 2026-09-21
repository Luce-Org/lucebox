#!/usr/bin/env python3
"""Integration test for opt-in `logprobs`/`top_logprobs` on /v1/chat/completions.

Exercises the non-streaming AR-path logprob response against a real
dflash_server: response shape, value bounds, top-K ordering, the
concatenation invariants (join(entry.token) == message field), the
byte-identical no-regression check against a plain request, and the
validation 400s.

Usage:
  # Against a running single-slot server:
  python3 tests/test_server_logprobs.py --base-url http://localhost:9099

  # Or launch one (qwen35 example):
  python3 tests/test_server_logprobs.py --launch ~/models/Qwen3.8-27B-UD-IQ4_XS.gguf \
      --draft ~/models/dflash2/model.safetensors --draft-block-size 16

  # Extra server flags can be appended after --server-args:
  python3 tests/test_server_logprobs.py --launch model.gguf --server-args "--paged-attention"
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request

PROMPT = "What is 17 + 25? Reply with just the number."
NO_THINKING = {"enable_thinking": False}


class LogprobsTest:
    def __init__(self, base_url: str, allow_spec_divergence: bool = False):
        self.base = base_url.rstrip("/")
        self.allow_spec_divergence = allow_spec_divergence
        self.passed = 0
        self.failed = 0
        self.warnings = 0

    def _warn(self, name: str, detail: str) -> None:
        self.warnings += 1
        print(f"  ⚠️  {name}: {detail}")

    def _req(self, path: str, body: dict, timeout: float = 300.0) -> dict:
        req = urllib.request.Request(
            self.base + path,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        return json.loads(urllib.request.urlopen(req, timeout=timeout).read())

    def _report(self, name: str, ok: bool, detail: str = "") -> None:
        if ok:
            self.passed += 1
            print(f"  ✅ {name}")
        else:
            self.failed += 1
            print(f"  ❌ {name}: {detail}")

    def _chat(self, **overrides) -> dict:
        body = {
            "model": "dflash",
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": 48,
            "temperature": 0.0,
            "stream": False,
            "chat_template_kwargs": dict(NO_THINKING),
        }
        body.update(overrides)
        return self._req("/v1/chat/completions", body)

    @staticmethod
    def _entries(resp: dict) -> list:
        lp = resp["choices"][0]["logprobs"]
        return lp.get("content", []) + lp.get("reasoning_content", [])

    def test_shape_and_absent_when_unrequested(self):
        print("\n[1] Response shape; logprobs key absent when not requested")
        plain = self._chat()
        self._report(
            "logprobs key absent when not requested",
            "logprobs" not in plain["choices"][0],
            f"keys: {sorted(plain['choices'][0])}",
        )
        r = self._chat(logprobs=True)
        self._report("got response", "choices" in r, f"got: {str(r)[:300]}")
        lp = r["choices"][0].get("logprobs")
        self._report("choices[0].logprobs present", lp is not None)
        if lp is None:
            return
        self._report(
            "logprobs has content+reasoning_content lists",
            isinstance(lp.get("content"), list)
            and isinstance(lp.get("reasoning_content"), list),
            f"keys: {sorted(lp)}",
        )
        entries = self._entries(r)
        self._report("at least one entry", len(entries) > 0)
        if not entries:
            return
        e = entries[0]
        for key in ("token", "logprob", "bytes", "top_logprobs"):
            self._report(f"entry has '{key}'", key in e, f"keys: {sorted(e)}")
        self._report(
            "top_logprobs empty when top_logprobs unset",
            e["top_logprobs"] == [],
            f"got: {e['top_logprobs']}",
        )

    def test_value_bounds(self):
        print("\n[2] logprob <= 0 and exp(logprob) in (0, 1]")
        r = self._chat(logprobs=True, top_logprobs=4)
        entries = self._entries(r)
        ok = bool(entries)
        bad = ""
        for e in entries:
            vals = [e["logprob"]] + [t["logprob"] for t in e["top_logprobs"]]
            for v in vals:
                if not (math.isfinite(v) and v <= 0.0 and 0.0 < math.exp(v) <= 1.0):
                    ok = False
                    bad = f"bad logprob {v}"
                    break
            if not ok:
                break
        self._report("all logprobs <= 0 and exp() in (0,1]", ok, bad)

    def test_top_logprobs_ordering(self):
        print("\n[3] top_logprobs=4: 4 entries, sorted, argmax matches chosen")
        r = self._chat(logprobs=True, top_logprobs=4)
        entries = self._entries(r)
        ok = bool(entries)
        detail = ""
        for i, e in enumerate(entries):
            top = e["top_logprobs"]
            if len(top) != 4:
                ok, detail = False, f"entry {i}: {len(top)} top entries"
                break
            lps = [t["logprob"] for t in top]
            if lps != sorted(lps, reverse=True):
                ok, detail = False, f"entry {i}: not descending {lps}"
                break
            # Greedy: the committed token is the argmax of the raw row.
            if top[0].get("token_id") != e.get("token_id"):
                ok, detail = False, (
                    f"entry {i}: chosen token_id {e.get('token_id')} != "
                    f"top0 {top[0].get('token_id')}"
                )
                break
            if abs(top[0]["logprob"] - e["logprob"]) > 1e-6:
                ok, detail = False, (
                    f"entry {i}: chosen logprob {e['logprob']} != "
                    f"top0 {top[0]['logprob']}"
                )
                break
        self._report("top-4 sorted desc; chosen == top0 (id + logprob)", ok, detail)

    def test_concat_invariants(self):
        print("\n[4] join(entry.token) == message.content / reasoning_content")

        def check(resp: dict, label: str) -> None:
            lp = resp["choices"][0].get("logprobs") or {}
            msg = resp["choices"][0]["message"]
            joined_c = "".join(e["token"] for e in lp.get("content", []))
            joined_r = "".join(e["token"] for e in lp.get("reasoning_content", []))
            self._report(
                f"{label}: content invariant",
                joined_c == (msg.get("content") or ""),
                f"joined={joined_c!r} content={msg.get('content')!r}",
            )
            self._report(
                f"{label}: reasoning invariant",
                joined_r == (msg.get("reasoning_content") or ""),
                f"joined={joined_r!r} reasoning={msg.get('reasoning_content')!r}",
            )

        check(self._chat(logprobs=True, top_logprobs=2), "no-thinking")

        # Thinking on: exercise the reasoning_content bucket.
        thinking = self._chat(
            logprobs=True,
            top_logprobs=2,
            max_tokens=256,
            chat_template_kwargs={"enable_thinking": True},
        )
        msg = thinking["choices"][0]["message"]
        self._report(
            "thinking request produced reasoning_content",
            bool(msg.get("reasoning_content")),
            f"keys: {sorted(msg)}",
        )
        check(thinking, "thinking")

    def test_no_regression_byte_identity(self):
        print("\n[5] content byte-identical with and without logprobs")
        off = self._chat(logprobs=False)
        on = self._chat(logprobs=True)
        c_off = off["choices"][0]["message"].get("content")
        c_on = on["choices"][0]["message"].get("content")
        identical = c_off == c_on
        if not identical and self.allow_spec_divergence and \
                off["usage"].get("spec_decode_ran"):
            # Spec verify is not bit-identical to the AR path on every
            # backend (e.g. ds4 hybrid prefill/verify numerics), so the
            # forced-AR logprobs request may legitimately diverge.
            self._warn(
                "message.content identical (spec-decode vs forced AR)",
                f"spec/AR divergence allowed: off={c_off!r} on={c_on!r}")
        else:
            self._report(
                "message.content identical (spec-decode vs forced AR)",
                identical,
                f"off={c_off!r} on={c_on!r}",
            )
        self._report(
            "finish_reason identical",
            off["choices"][0]["finish_reason"] == on["choices"][0]["finish_reason"],
            f"off={off['choices'][0]['finish_reason']} "
            f"on={on['choices'][0]['finish_reason']}",
        )

    def test_validation_400s(self):
        print("\n[6] Validation 400s")
        cases = [
            ("top_logprobs=21", {"logprobs": True, "top_logprobs": 21}),
            ("top_logprobs without logprobs", {"top_logprobs": 4}),
            ("top_logprobs with logprobs:false",
             {"logprobs": False, "top_logprobs": 4}),
            ("stream + logprobs",
             {"logprobs": True, "stream": True}),
            ("logprobs wrong type", {"logprobs": "yes"}),
            ("top_logprobs wrong type",
             {"logprobs": True, "top_logprobs": "4"}),
        ]
        for name, extra in cases:
            body = {
                "model": "dflash",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 4,
            }
            body.update(extra)
            req = urllib.request.Request(
                self.base + "/v1/chat/completions",
                data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                urllib.request.urlopen(req, timeout=30)
                self._report(f"400 for {name}", False, "got 200")
            except urllib.error.HTTPError as e:
                detail = ""
                try:
                    payload = json.loads(e.read())
                    detail = payload.get("error", {}).get("message", "")
                except Exception:
                    pass
                self._report(
                    f"400 for {name}", e.code == 400,
                    f"got HTTP {e.code} ({detail})",
                )
            except Exception as e:
                self._report(f"400 for {name}", False, str(e))

    def test_stop_sequence_invariant(self):
        print("\n[7] stop sequence: concatenation invariant still holds")
        # Greedy answer to PROMPT starts with "42"; stop mid-answer.
        r = self._chat(logprobs=True, top_logprobs=2, stop=["4"])
        msg = r["choices"][0]["message"]
        lp = r["choices"][0]["logprobs"]
        joined = "".join(e["token"] for e in lp["content"])
        self._report(
            "stop sequence delivered text",
            isinstance(msg.get("content"), str),
            f"content={msg.get('content')!r}",
        )
        self._report(
            "content invariant under stop truncation",
            joined == (msg.get("content") or ""),
            f"joined={joined!r} content={msg.get('content')!r}",
        )

    def run_all(self) -> bool:
        print(f"Logprobs test: {self.base}")
        self.test_shape_and_absent_when_unrequested()
        self.test_value_bounds()
        self.test_top_logprobs_ordering()
        self.test_concat_invariants()
        self.test_no_regression_byte_identity()
        self.test_validation_400s()
        self.test_stop_sequence_invariant()
        total = self.passed + self.failed
        print(f"\n{'=' * 50}")
        print(f"Results: {self.passed}/{total} passed, {self.failed} failed"
              + (f" ({self.warnings} warnings)" if self.warnings else ""))
        return self.failed == 0


def wait_for_server(base_url: str, timeout: float = 600.0) -> bool:
    start = time.time()
    while time.time() - start < timeout:
        try:
            if urllib.request.urlopen(f"{base_url}/health", timeout=5).status == 200:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def main() -> None:
    parser = argparse.ArgumentParser(description="dflash_server logprobs test")
    parser.add_argument("--base-url", default="http://localhost:8080")
    parser.add_argument("--launch", metavar="MODEL_PATH",
                        help="Launch dflash_server with this model")
    parser.add_argument("--draft", metavar="DRAFT_PATH",
                        help="Spec-decode drafter for --launch")
    parser.add_argument("--draft-block-size", type=int, default=None)
    parser.add_argument("--max-concurrency", type=int, default=None)
    parser.add_argument("--port", type=int, default=9099)
    parser.add_argument("--server-bin", default=None)
    parser.add_argument("--server-args", default="",
                        help="Extra flags for --launch, e.g. '--paged-attention'")
    parser.add_argument("--allow-spec-divergence", action="store_true",
                        help="Downgrade the spec-decode-vs-AR byte-identity "
                             "check to a warning when the baseline request "
                             "ran speculative decode (e.g. ds4 hybrid, where "
                             "spec verify is not bit-identical to AR)")
    args = parser.parse_args()

    server_proc = None
    try:
        if args.launch:
            base_url = f"http://localhost:{args.port}"
            bin_path = args.server_bin
            if not bin_path:
                for c in ("build/dflash_server", "dflash_server"):
                    if os.path.isfile(c):
                        bin_path = c
                        break
            if not bin_path:
                print("ERROR: Could not find dflash_server binary")
                sys.exit(1)
            cmd = [bin_path, args.launch, "--port", str(args.port)]
            if args.draft:
                cmd += ["--draft", args.draft]
            if args.draft_block_size:
                cmd += ["--draft-block-size", str(args.draft_block_size)]
            if args.max_concurrency:
                cmd += ["--max-concurrency", str(args.max_concurrency)]
            cmd += shlex.split(args.server_args)
            print(f"Launching: {' '.join(cmd)}")
            server_proc = subprocess.Popen(cmd)
            print("Waiting for server to start...")
            if not wait_for_server(base_url):
                print("ERROR: Server did not start within timeout")
                server_proc.terminate()
                sys.exit(1)
            print("Server ready!")
            args.base_url = base_url

        ok = LogprobsTest(
            args.base_url,
            allow_spec_divergence=args.allow_spec_divergence).run_all()
        sys.exit(0 if ok else 1)
    finally:
        if server_proc:
            print("\nShutting down server...")
            server_proc.send_signal(signal.SIGINT)
            try:
                server_proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server_proc.kill()
                server_proc.wait()
            print("Server stopped.")


if __name__ == "__main__":
    main()
