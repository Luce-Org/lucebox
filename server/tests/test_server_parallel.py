#!/usr/bin/env python3
"""End-to-end integration tests for concurrent serving (--max-concurrency N).

Exercises the qwen35 paged-attention slot engine: true decode overlap across
streams, per-request state isolation, sequential-vs-concurrent consistency,
SSE interleaving, over-subscription (queueing beyond slot count), concurrent
non-streaming completions, and non-pausing admission.

Usage:
    # Start server first with concurrent serving enabled:
    ./server/build/dflash_server <qwen36-model.gguf> --port 9099 \
        --paged-attention --max-concurrency 3

    # Then run tests:
    python3 server/tests/test_server_parallel.py \
        --base-url http://127.0.0.1:9099 --max-concurrency 3

The non-power-of-two default is intentional: three physical slots produce a
four-row compact decode bucket, with the final row mapped to padding. This
guards the distinction between graph bucket width and physical slot count.

Note: batched decode may legally differ from single-request decode at the
token level (GEMM reduction order can flip near-tie tokens), so all answer
checks are content-level (the expected number appears), never exact-match.
"""

import argparse
import json
import re
import sys
import threading
import time
import urllib.request
import urllib.error


def make_math_prompts(count: int):
    """Deterministic arithmetic prompts for up to 64 isolated streams.

    Answers occupy the 3-digit range 100..163 while operands occupy the
    disjoint 2-digit range 50..82.  This matters because the isolation test
    inspects reasoning as well as final content: a model may echo an operand
    while working, and that must not look like another stream's answer.
    """
    if not 0 <= count <= 64:
        raise ValueError(f"prompt count must be in [0, 64], got {count}")

    prompts = []
    for i in range(count):
        s = 100 + i
        a = 50 + i // 2
        b = 50 + (i + 1) // 2
        prompts.append((f"What is {a}+{b}? Answer with just the number.", str(s)))
    return prompts


def make_long_prompts(count: int):
    """Prompts that force long, multi-chunk generations, so overlap and
    interleave are observable. Each stream counts a distinct range; the
    expected marker is the range start (always emitted early)."""
    prompts = []
    for i in range(count):
        start = 100 + 50 * i
        end = start + 40
        prompts.append((
            f"Count from {start} to {end}, separated by commas, "
            f"no other text.", str(start)))
    return prompts


def contains_number(text: str, number: str) -> bool:
    """True if `number` appears in `text` as a standalone number
    (not embedded in a longer digit run)."""
    return re.search(rf"(?<![0-9]){re.escape(number)}(?![0-9])", text) is not None


class ParallelTestSuite:
    def __init__(self, base_url: str, parallel: int):
        self.base = base_url.rstrip("/")
        self.parallel = parallel
        self.passed = 0
        self.failed = 0
        self.skipped = 0

    def _req(self, method: str, path: str, body: dict | None = None,
             stream: bool = False, timeout: float = 300.0):
        url = self.base + path
        data = json.dumps(body).encode() if body else None
        headers = {"Content-Type": "application/json"} if body else {}
        req = urllib.request.Request(url, data=data, headers=headers, method=method)
        resp = urllib.request.urlopen(req, timeout=timeout)
        if stream:
            return resp  # caller reads lines
        return json.loads(resp.read().decode())

    def _check(self, name: str, ok: bool, detail: str = ""):
        if ok:
            self.passed += 1
            print(f"  ✅ {name}")
        else:
            self.failed += 1
            print(f"  ❌ {name}: {detail}")

    def _skip(self, name: str, reason: str):
        self.skipped += 1
        print(f"  ⏭️  {name}: {reason}")

    # ── Request workers ──────────────────────────────────────────────────

    def _chat_body(self, prompt: str, max_tokens: int, stream: bool) -> dict:
        return {
            "model": "dflash",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": stream,
        }

    def _stream_worker(self, idx: int, prompt: str, max_tokens: int,
                       results: list, barrier: threading.Barrier | None = None,
                       timeline: list | None = None,
                       timeline_lock=None,
                       timeout: float = 600.0):
        """Run one streaming chat completion; record content and chunk timing."""
        r = {"ok": False, "error": None, "content": "", "reasoning": "",
             "first_chunk_t": None, "finish_t": None, "n_chunks": 0}
        results[idx] = r
        try:
            if barrier is not None:
                barrier.wait(timeout=60)
            resp = self._req("POST", "/v1/chat/completions",
                             self._chat_body(prompt, max_tokens, stream=True),
                             stream=True, timeout=timeout)
            for line in resp:
                line = line.decode().strip()
                if not line:
                    continue
                if line == "data: [DONE]":
                    break
                if not line.startswith("data: "):
                    continue
                chunk = json.loads(line[6:])
                choices = chunk.get("choices") or [{}]
                delta = choices[0].get("delta", {})
                got_piece = False
                if delta.get("reasoning_content"):
                    r["reasoning"] += delta["reasoning_content"]
                    got_piece = True
                if delta.get("content"):
                    r["content"] += delta["content"]
                    got_piece = True
                if got_piece:
                    now = time.monotonic()
                    r["n_chunks"] += 1
                    if r["first_chunk_t"] is None:
                        r["first_chunk_t"] = now
                    if timeline is not None:
                        with timeline_lock:
                            timeline.append((now, idx))
            r["finish_t"] = time.monotonic()
            resp.close()
            r["ok"] = True
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            r["error"] = f"HTTP {e.code}: {body[:300]}"
        except Exception as e:
            r["error"] = f"{type(e).__name__}: {e}"

    def _nonstream_worker(self, idx: int, prompt: str, max_tokens: int,
                          results: list, barrier: threading.Barrier | None = None,
                          start_delay: float = 0.0,
                          timeout: float = 600.0):
        """Run one non-streaming chat completion."""
        r = {"ok": False, "error": None, "content": "", "reasoning": "",
             "usage": {}, "started_t": None, "finish_t": None}
        results[idx] = r
        try:
            if barrier is not None:
                barrier.wait(timeout=60)
            if start_delay > 0:
                time.sleep(start_delay)
            r["started_t"] = time.monotonic()
            resp = self._req("POST", "/v1/chat/completions",
                             self._chat_body(prompt, max_tokens, stream=False),
                             timeout=timeout)
            msg = resp["choices"][0]["message"]
            r["content"] = msg.get("content") or ""
            r["reasoning"] = msg.get("reasoning_content") or ""
            r["usage"] = resp.get("usage", {})
            r["finish_t"] = time.monotonic()
            r["ok"] = True
        except urllib.error.HTTPError as e:
            body = e.read().decode(errors="replace")
            r["error"] = f"HTTP {e.code}: {body[:300]}"
        except Exception as e:
            r["error"] = f"{type(e).__name__}: {e}"

    def _run_workers(self, worker, count: int, per_request_kwargs: list,
                     join_timeout: float = 900.0):
        """Launch `count` worker threads through a start barrier; return results."""
        results: list = [None] * count
        barrier = threading.Barrier(count)
        threads = []
        for i in range(count):
            t = threading.Thread(target=worker,
                                 args=(i,), kwargs={**per_request_kwargs[i],
                                                    "results": results,
                                                    "barrier": barrier},
                                 daemon=True)
            threads.append(t)
            t.start()
        for t in threads:
            t.join(timeout=join_timeout)
        return results

    def _launch_streams(self, prompts, max_tokens: int,
                        timeline: list | None = None,
                        join_timeout: float = 900.0):
        """Fire len(prompts) streaming requests simultaneously."""
        timeline_lock = threading.Lock() if timeline is not None else None
        kwargs = [{"prompt": p, "max_tokens": max_tokens,
                   "timeline": timeline, "timeline_lock": timeline_lock}
                  for p, _ in prompts]
        return self._run_workers(self._stream_worker, len(prompts), kwargs,
                                 join_timeout=join_timeout)

    def _launch_nonstream(self, prompts, max_tokens: int,
                          join_timeout: float = 900.0):
        """Fire len(prompts) non-streaming requests simultaneously."""
        kwargs = [{"prompt": p, "max_tokens": max_tokens} for p, _ in prompts]
        return self._run_workers(self._nonstream_worker, len(prompts), kwargs,
                                 join_timeout=join_timeout)

    def _all_completed(self, results, label: str) -> bool:
        """Check every worker result; report per-request failures."""
        all_ok = True
        for i, r in enumerate(results):
            if r is None:
                self._check(f"{label} request {i+1} completes", False,
                            "worker did not finish (timeout)")
                all_ok = False
            elif not r["ok"]:
                self._check(f"{label} request {i+1} completes", False, r["error"])
                all_ok = False
        if all_ok:
            self._check(f"all {len(results)} {label} requests complete "
                        "with 200", True)
        return all_ok

    @staticmethod
    def _combined(r) -> str:
        return r["reasoning"] + "\n" + r["content"]

    # ── Tests ────────────────────────────────────────────────────────────

    def test_parallel_streaming(self):
        """N simultaneous streams must overlap and their chunks interleave."""
        n = self.parallel
        print(f"\n[PAR-1] Streaming overlap + interleave — "
              f"{n} simultaneous requests")
        if n == 1:
            self._skip("streaming overlap + interleave",
                       "--max-concurrency 1: concurrency is not expected; "
                       "run with --max-concurrency > 1")
            return
        prompts = make_long_prompts(n)
        timeline: list = []
        t0 = time.monotonic()
        results = self._launch_streams(prompts, max_tokens=192,
                                       timeline=timeline)
        elapsed = time.monotonic() - t0
        if not self._all_completed(results, "streaming"):
            return

        missing = [i + 1 for i, r in enumerate(results)
                   if r["first_chunk_t"] is None]
        self._check("every stream produced at least one chunk",
                    not missing, f"streams with no chunks: {missing}")
        if missing:
            return

        latest_first = max(r["first_chunk_t"] for r in results)
        earliest_finish = min(r["finish_t"] for r in results)
        rel = [(f"s{i+1}: first={r['first_chunk_t']-t0:.2f}s "
                f"finish={r['finish_t']-t0:.2f}s")
               for i, r in enumerate(results)]
        self._check("all first chunks arrive before earliest finish "
                    "(true overlap, not serialization)",
                    latest_first < earliest_finish,
                    f"latest first chunk at {latest_first-t0:.2f}s, earliest "
                    f"finish at {earliest_finish-t0:.2f}s; {'; '.join(rel)}")
        print(f"    → {n} streams completed in {elapsed:.1f}s; "
              f"latest first chunk {latest_first-t0:.2f}s, "
              f"earliest finish {earliest_finish-t0:.2f}s")

        seq = [idx for _, idx in sorted(timeline)]
        distinct = len(set(seq))
        runs = 1 + sum(1 for a, b in zip(seq, seq[1:]) if a != b) if seq else 0
        self._check("chunks arrived from at least two streams", distinct >= 2,
                    f"streams seen: {sorted(set(seq))}")
        # If every stream's chunks formed one contiguous block, runs would
        # equal distinct; interleaving means some stream owns >= 2 runs.
        self._check("chunks from different streams interleave",
                    runs > distinct,
                    f"{len(seq)} chunks arrived in {runs} contiguous runs "
                    f"across {distinct} streams (fully serialized order)")
        print(f"    → {len(seq)} chunks, {distinct} streams, {runs} runs")

    def test_parallel_isolation(self):
        """Concurrent streams with distinct prompts: each answer belongs to
        its own prompt, none leaks into another stream (state isolation)."""
        n = self.parallel
        print(f"\n[PAR-2] Isolation — {n} concurrent distinct prompts")
        prompts = make_math_prompts(n)
        results = self._launch_streams(prompts, max_tokens=512)
        if not self._all_completed(results, "streaming"):
            return

        answers = [ans for _, ans in prompts]
        for i, r in enumerate(results):
            combined = self._combined(r)
            # Positive: own answer somewhere in reasoning+content.
            self._check(f"stream {i+1} contains its own answer {answers[i]}",
                        contains_number(combined, answers[i]),
                        f"content={r['content']!r} "
                        f"reasoning={r['reasoning'][:200]!r}")
            for j, answer in enumerate(answers):
                if i == j:
                    continue
                self._check(
                    f"stream {i+1} excludes stream {j+1}'s answer {answer}",
                    not contains_number(combined, answer),
                    f"content={r['content']!r} "
                    f"reasoning={r['reasoning'][:200]!r}")

    def test_parallel_nonstream(self):
        """A prompt answered correctly alone must still be answered correctly
        when decoded inside an N-way concurrent non-streaming batch. Also
        validates every concurrent answer and its completion-token usage.
        Content-level match only — batched GEMM reduction order may legally
        flip near-tie tokens, so exact token equality is NOT required."""
        n = self.parallel
        print(f"\n[PAR-3] Sequential + concurrent non-streaming consistency")
        prompts = make_math_prompts(n)
        probe_prompt, probe_answer = prompts[0]

        # Solo run.
        solo = [None]
        self._nonstream_worker(0, probe_prompt, 512, solo)
        if not solo[0]["ok"]:
            self._check("solo request completes", False, solo[0]["error"])
            return
        self._check("solo request completes", True)
        solo_ok = contains_number(self._combined(solo[0]), probe_answer)
        self._check(f"solo answer contains {probe_answer}", solo_ok,
                    f"content={solo[0]['content']!r} "
                    f"reasoning={solo[0]['reasoning'][:200]!r}")
        print(f"    → solo content: {solo[0]['content']!r}")

        # Same prompt inside an N-way concurrent batch.
        results = self._launch_nonstream(prompts, max_tokens=512)
        if not self._all_completed(results, "non-streaming"):
            return
        conc = results[0]
        print(f"    → concurrent content: {conc['content']!r}")
        if solo[0]["content"] != conc["content"]:
            print("    → note: solo/concurrent contents differ at token level "
                  "(allowed — batched reduction order)")
        for i, r in enumerate(results):
            ans = prompts[i][1]
            self._check(f"request {i+1} contains its answer {ans}",
                        contains_number(self._combined(r), ans),
                        f"content={r['content']!r} "
                        f"reasoning={r['reasoning'][:200]!r}")
            completion_tokens = r["usage"].get("completion_tokens", 0)
            self._check(f"request {i+1} usage.completion_tokens > 0",
                        completion_tokens > 0,
                        f"usage={r['usage']}")
            prompt_tokens = r["usage"].get("prompt_tokens", 0)
            timings = r["usage"].get("timings", {})
            self._check(
                f"request {i+1} timings account for the full prompt",
                prompt_tokens > 0
                and timings.get("prefilled_tokens") == prompt_tokens
                and timings.get("effective_prompt_tokens") == prompt_tokens,
                f"usage={r['usage']}")

    def test_parallel_more_than_slots(self):
        """2*N requests at once: extras must queue behind the N slots and
        all must finish with correct answers."""
        n = self.parallel
        count = 2 * n
        print(f"\n[PAR-4] Over-subscription — {count} requests on {n} slots")
        prompts = make_math_prompts(count)
        t0 = time.monotonic()
        results = self._launch_nonstream(prompts, max_tokens=512,
                                         join_timeout=1800.0)
        elapsed = time.monotonic() - t0
        if not self._all_completed(results, "non-streaming"):
            return
        for i, r in enumerate(results):
            ans = prompts[i][1]
            self._check(f"request {i+1} contains its answer {ans}",
                        contains_number(self._combined(r), ans),
                        f"content={r['content']!r} "
                        f"reasoning={r['reasoning'][:200]!r}")
        print(f"    → {count} requests completed in {elapsed:.1f}s")

    def test_unequal_packed_prefills(self):
        """A multi-chunk prefill must keep its slot-local state after an
        earlier short prefill commits and leaves the FIFO head."""
        print("\n[PAR-5] Unequal prefills retain isolated paged state")
        if self.parallel < 2:
            self._skip("unequal packed prefill",
                       "--max-concurrency 1: packed prefill is unavailable")
            return

        short_prompt = "What is 55+56? Answer with just the number."
        filler = "\n".join(
            f"record {i}: alpha beta gamma delta epsilon zeta eta theta"
            for i in range(360))
        long_prompt = (
            f"Read and then ignore these padding records:\n{filler}\n"
            "What is 4500+4501? Answer with just the number.")

        # Bias the short request toward the FIFO head, then assert the observed
        # completion order below so this test cannot silently cover the reverse
        # scenario when host scheduling delays either client thread.
        kwargs = [
            {"prompt": short_prompt, "max_tokens": 512,
             "start_delay": 0.0},
            {"prompt": long_prompt, "max_tokens": 512,
             "start_delay": 0.001},
        ]
        results = self._run_workers(
            self._nonstream_worker, 2, kwargs, join_timeout=1800.0)
        if not self._all_completed(results, "unequal-prefill"):
            return
        self._check(
            "short prefill commits before the long prefill",
            results[0]["finish_t"] < results[1]["finish_t"],
            f"short_finish={results[0]['finish_t']} "
            f"long_finish={results[1]['finish_t']}")

        self._check(
            "short prefill answers 111",
            contains_number(self._combined(results[0]), "111"),
            f"content={results[0]['content']!r}")
        self._check(
            "long prefill answers 9001 after the short prefill commits",
            contains_number(self._combined(results[1]), "9001"),
            f"content={results[1]['content']!r} "
            f"reasoning={results[1]['reasoning'][:200]!r}")
        long_prompt_tokens = results[1]["usage"].get("prompt_tokens", 0)
        self._check(
            "long prompt crosses the 2048-token initial prefill chunk",
            long_prompt_tokens > 2048,
            f"usage={results[1]['usage']}")

    def test_parallel_prefill_no_pause(self):
        """A long admission must not pause a stream that is already decoding."""
        n = self.parallel
        print("\n[PAR-7] Prefill/decode fusion — decode continues during "
              "a long prefill")
        if n == 1:
            self._skip("prefill no-pause",
                       "--max-concurrency 1: fusion not applicable, skipping")
            return

        # Stream A starts alone and reaches steady decode before B arrives.
        a_prompt, _ = make_long_prompts(1)[0]
        timeline: list = []
        timeline_lock = threading.Lock()
        a_res: list = [None]
        a_thread = threading.Thread(
            target=self._stream_worker, args=(0, a_prompt, 320, a_res),
            kwargs={"timeline": timeline, "timeline_lock": timeline_lock},
            daemon=True)
        a_thread.start()
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline:
            ra = a_res[0]
            if ra is not None and (ra["n_chunks"] >= 3 or ra["error"]):
                break
            time.sleep(0.05)
        ra = a_res[0]
        if ra is None or ra["error"] or ra["n_chunks"] < 3:
            self._check("stream A reaches steady decode", False,
                        "no chunks" if ra is None else
                        (ra["error"] or f"only {ra['n_chunks']} chunks"))
            return
        self._check("stream A reaches steady decode", True)

        # Stream B has several 512-token prefill chunks. Answer 167 cannot
        # collide with the filler item indices (0..139).
        filler = "\n".join(
            f"item {i}: the quick brown fox jumps over the lazy dog"
            for i in range(140))
        b_prompt = (f"Here is a list:\n{filler}\n"
                    "Ignore the list entirely. What is 83+84? "
                    "Answer with just the number.")
        b_res: list = [None]
        b_started = time.monotonic()
        b_thread = threading.Thread(
            target=self._stream_worker, args=(0, b_prompt, 512, b_res),
            daemon=True)
        b_thread.start()
        b_thread.join(timeout=900)
        a_thread.join(timeout=900)

        if b_thread.is_alive():
            self._check("stream B completes", False, "worker timed out")
            return
        ra = a_res[0]
        if a_thread.is_alive() or ra is None or not ra["ok"]:
            self._check(
                "stream A completes", False,
                "worker timed out" if a_thread.is_alive() else
                ("no result" if ra is None else str(ra["error"])))
            return
        self._check("stream A completes", True)

        rb = b_res[0]
        if rb is None or not rb["ok"] or rb["first_chunk_t"] is None:
            self._check("stream B completes", False,
                        "no result" if rb is None else str(rb["error"]))
            return
        self._check("stream B completes", True)
        self._check("stream B answers 167",
                    contains_number(self._combined(rb), "167"),
                    f"content={rb['content']!r}")

        # Blocking admission leaves the prefill-dominated first 70% of B's
        # TTFT window empty. Fused steps keep producing A outputs there.
        window = rb["first_chunk_t"] - b_started
        early_end = b_started + 0.7 * window
        with timeline_lock:
            a_early = [t for t, idx in timeline
                       if idx == 0 and b_started <= t <= early_end]
        if ra["finish_t"] is not None and ra["finish_t"] < early_end:
            self._skip("A kept emitting during B's prefill",
                       "stream A finished before B's window closed")
            return
        self._check("stream A kept emitting during B's prefill window",
                    len(a_early) >= 2,
                    f"A emitted {len(a_early)} chunks in the first "
                    f"{0.7 * window:.2f}s of B's {window:.2f}s "
                    "prefill window")

    # ── Run all ──────────────────────────────────────────────────────────

    def run_all(self):
        print("=" * 60)
        print("Parallel Serving Tests")
        print(f"Target: {self.base} (parallel={self.parallel})")
        print("=" * 60)

        self.test_parallel_streaming()
        self.test_parallel_isolation()
        self.test_parallel_nonstream()
        self.test_parallel_more_than_slots()
        self.test_unequal_packed_prefills()
        self.test_parallel_prefill_no_pause()

        print("\n" + "=" * 60)
        total = self.passed + self.failed + self.skipped
        print(f"Results: {self.passed}/{total} passed, {self.failed} failed"
              + (f", {self.skipped} skipped" if self.skipped else ""))
        return 0 if self.failed == 0 else 1


# ─── Main ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Concurrent serving tests for dflash_server "
                    "(run against a server started with "
                    "--paged-attention --max-concurrency N)")
    parser.add_argument("--base-url", default="http://127.0.0.1:9099",
                        help="Server base URL")
    parser.add_argument("--max-concurrency", type=int, default=3,
                        help="N: --max-concurrency value used to start the server")
    args = parser.parse_args()

    if not (1 <= args.max_concurrency <= 64):
        print("ERROR: --max-concurrency must be in [1, 64], "
              f"got {args.max_concurrency}")
        sys.exit(2)

    # Preflight: server must already be running.
    base = args.base_url.rstrip("/")
    try:
        r = urllib.request.urlopen(base + "/health", timeout=10)
        health = json.loads(r.read().decode())
        if health.get("status") != "ok":
            print(f"ERROR: server unhealthy: {health}")
            sys.exit(2)
    except Exception as e:
        print(f"ERROR: server not reachable at {base}: {e}")
        print("Start it first, e.g.:")
        print(f"  ./server/build/dflash_server <model.gguf> --port 9099 "
              f"--paged-attention --max-concurrency {args.max_concurrency}")
        sys.exit(2)

    suite = ParallelTestSuite(base, args.max_concurrency)
    sys.exit(suite.run_all())


if __name__ == "__main__":
    main()
