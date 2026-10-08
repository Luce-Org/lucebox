#!/usr/bin/env python3
"""Run four long prompts alone, then together; also check planted-fact recall.

Usage: python3 server/test/qwen4exp_concurrency_http.py [http://127.0.0.1:8080]
Run against `luce_server MODEL --max-concurrency 4 --max-ctx 32768
--prefix-cache-slots 0 --disk-prefix-cache off`. No environment settings.
Solo means one active request on this same server.
HTTP checks text/counts; the GPU C++ probes check token IDs and indexer state.
"""
import concurrent.futures
import argparse
import json
import sys
import threading
import time
import urllib.request


def post(base, path, body):
    request = urllib.request.Request(
        base + path, json.dumps(body).encode(), {"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=7200) as response:
        return json.load(response)


def payload(slot, lines):
    secret = ["VIOLET-7319", "AMBER-4826", "COBALT-9153", "JADE-2648"][slot]
    filler = "The archive entry describes ordinary weather and routine maintenance.\n"
    split = lines * (slot + 1) // 5
    prompt = (
        f"Archive {slot}. Read the records and answer the question at the end.\n"
        + filler * split
        + f"The vault access code is {secret}.\n"
        + filler * (lines - split)
        + "What is the vault access code? Reply with only the code."
    )
    return {
        "model": "luce", "messages": [{"role": "user", "content": prompt}],
        "temperature": 0, "seed": 42, "max_tokens": 96,
        "reasoning_effort": "none", "stream": False,
    }, secret


def timing_summary(responses, parallel):
    timings = [r["usage"]["timings"] for r in responses]
    # Parallel requests start together: use the longest phase, not the sum
    # of their overlapping durations. Solo requests run sequentially.
    duration = max if parallel else sum
    prefill_ms = duration(t["prefill_ms"] for t in timings)
    decode_ms = duration(t["decode_ms"] for t in timings)
    if prefill_ms <= 0 or decode_ms <= 0:
        raise ValueError("missing or nonpositive HTTP phase timings")
    return {
        "prefill_tok_s": sum(t["prefilled_tokens"] for t in timings) * 1000 / prefill_ms,
        "decode_tok_s": sum(r["usage"]["completion_tokens"] for r in responses) * 1000 / decode_ms,
        "decode_per_stream_tok_s": [t["decode_tokens_per_sec"] for t in timings],
    }


def run(base, target):
    requests = []
    for slot in range(4):
        # Count with the server tokenizer, rather than assuming words == tokens.
        lo, hi = 0, target
        while lo < hi:
            mid = (lo + hi) // 2
            body, _ = payload(slot, mid)
            count = post(base, "/v1/messages/count_tokens", body)["input_tokens"]
            if count < target:
                lo = mid + 1
            else:
                hi = mid
        requests.append(payload(slot, lo))

    def generate(item):
        return post(base, "/v1/chat/completions", item[0])

    start = time.monotonic()
    solo = [generate(item) for item in requests]
    solo_wall_s = time.monotonic() - start
    barrier = threading.Barrier(4)

    def generate_parallel(item):
        barrier.wait()
        return generate(item)

    start = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        parallel = list(pool.map(generate_parallel, requests))
    parallel_wall_s = time.monotonic() - start
    ok = True
    passed_count = 0
    for slot, ((_, secret), a, b) in enumerate(zip(requests, solo, parallel)):
        text_a = a["choices"][0]["message"].get("content") or ""
        text_b = b["choices"][0]["message"].get("content") or ""
        counts = [r["usage"]["prompt_tokens"] for r in (a, b)]
        passed = (
            all(target <= n < target + 64 for n in counts)
            and a["choices"] == b["choices"]
            and a["usage"]["completion_tokens"] == b["usage"]["completion_tokens"]
            and text_a.strip() == secret and text_b.strip() == secret
            and all(not r["usage"]["timings"]["cache_hit"]
                    and r["usage"]["timings"]["cached_prefix_tokens"] == 0
                    and r["usage"]["timings"]["prefilled_tokens"] == n
                    for r, n in zip((a, b), counts))
        )
        print(json.dumps({"target": target, "slot": slot, "pass": passed,
                          "solo": a, "parallel": b}), flush=True)
        ok = passed and ok
        passed_count += passed
    summary = {
        "type": "timings", "target": target, "passed": passed_count, "total": 4,
        "solo": timing_summary(solo, False), "parallel": timing_summary(parallel, True),
        "solo_wall_s": solo_wall_s, "parallel_wall_s": parallel_wall_s,
    }
    if target == 16384:
        summary["session68"] = {"solo_prefill_tok_s": 1140, "parallel_prefill_tok_s": 1137,
                                "solo_decode_tok_s": 24, "decode_per_stream_tok_s": [6.1, 6.4],
                                "gtt_peak_gb": 87.9}
        summary["change_percent"] = {
            "solo_prefill": 100 * (summary["solo"]["prefill_tok_s"] / 1140 - 1),
            "parallel_prefill": 100 * (summary["parallel"]["prefill_tok_s"] / 1137 - 1),
            "solo_decode": 100 * (summary["solo"]["decode_tok_s"] / 24 - 1),
        }
    print(json.dumps(summary), flush=True)
    return ok


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", nargs="?", default="http://127.0.0.1:8080")
    parser.add_argument("--targets", type=int, nargs="+", default=[2304, 16384])
    args = parser.parse_args()
    if any(target <= 0 for target in args.targets):
        parser.error("targets must be positive")
    results = [run(args.base.rstrip("/"), target) for target in args.targets]
    print(json.dumps({"type": "result", "pass": all(results), "targets": args.targets}), flush=True)
    sys.exit(0 if all(results) else 1)
