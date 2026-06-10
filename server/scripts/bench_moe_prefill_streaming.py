#!/usr/bin/env python3
"""
bench_moe_prefill_streaming.py — End-to-end benchmark comparing prefill
throughput with and without MoE streaming.

Usage:
    # Against a running server (streaming enabled):
    python bench_moe_prefill_streaming.py --url http://localhost:8080

    # Compare two servers (streaming vs baseline):
    python bench_moe_prefill_streaming.py \
        --url-streaming http://localhost:8080 \
        --url-baseline http://localhost:8081

Measures prompt eval tok/s at varying prompt lengths.
"""

import argparse
import json
import time
import sys
from urllib.request import urlopen, Request
from urllib.error import URLError


def generate_prompt(n_tokens: int) -> str:
    """Generate a prompt that's approximately n_tokens long."""
    # Average ~1.3 tokens per word in English
    words_needed = max(1, int(n_tokens / 1.3))
    base = "The quick brown fox jumps over the lazy dog. "
    words = base.split()
    prompt_words = []
    for i in range(words_needed):
        prompt_words.append(words[i % len(words)])
    return " ".join(prompt_words)


def bench_prefill(url: str, prompt: str, max_tokens: int = 1) -> dict:
    """Send a completion request and measure prefill time."""
    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }

    req = Request(
        f"{url}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )

    t0 = time.perf_counter()
    try:
        with urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read())
    except URLError as e:
        return {"error": str(e)}
    t1 = time.perf_counter()

    usage = body.get("usage", {})
    prompt_tokens = usage.get("prompt_tokens", 0)
    # The server reports timings in the response if available
    timings = body.get("timings", {})
    prefill_ms = timings.get("prompt_ms", (t1 - t0) * 1000)

    return {
        "prompt_tokens": prompt_tokens,
        "wall_ms": (t1 - t0) * 1000,
        "prefill_ms": prefill_ms,
        "tok_per_s": prompt_tokens / (prefill_ms / 1000) if prefill_ms > 0 else 0,
    }


def run_sweep(url: str, prompt_lengths: list, n_warmup: int = 1, n_iter: int = 3) -> list:
    """Run prefill benchmark at various prompt lengths."""
    results = []

    for n_tok in prompt_lengths:
        prompt = generate_prompt(n_tok)

        # Warmup
        for _ in range(n_warmup):
            bench_prefill(url, prompt)

        # Measure
        measurements = []
        for _ in range(n_iter):
            r = bench_prefill(url, prompt)
            if "error" in r:
                print(f"  ERROR at T={n_tok}: {r['error']}", file=sys.stderr)
                break
            measurements.append(r)

        if measurements:
            avg_prefill_ms = sum(m["prefill_ms"] for m in measurements) / len(measurements)
            avg_tok_s = sum(m["tok_per_s"] for m in measurements) / len(measurements)
            actual_tokens = measurements[0]["prompt_tokens"]
            results.append({
                "target_tokens": n_tok,
                "actual_tokens": actual_tokens,
                "avg_prefill_ms": avg_prefill_ms,
                "avg_tok_per_s": avg_tok_s,
            })
            print(f"  T={n_tok:5d}  actual={actual_tokens:5d}  "
                  f"prefill={avg_prefill_ms:8.1f} ms  "
                  f"tok/s={avg_tok_s:8.1f}")

    return results


def main():
    parser = argparse.ArgumentParser(description="MoE prefill streaming benchmark")
    parser.add_argument("--url", type=str, help="Server URL to benchmark")
    parser.add_argument("--url-streaming", type=str, help="Server URL with streaming enabled")
    parser.add_argument("--url-baseline", type=str, help="Server URL with baseline (CPU cold)")
    parser.add_argument("--prompt-lengths", type=str, default="128,256,512,1024,2048,4096,8192",
                        help="Comma-separated prompt lengths to test")
    parser.add_argument("--n-iter", type=int, default=3, help="Iterations per measurement")
    parser.add_argument("--n-warmup", type=int, default=1, help="Warmup iterations")
    args = parser.parse_args()

    prompt_lengths = [int(x) for x in args.prompt_lengths.split(",")]

    if args.url:
        print(f"\n=== Benchmarking: {args.url} ===")
        results = run_sweep(args.url, prompt_lengths, args.n_warmup, args.n_iter)
        print(f"\n{'T':>8}  {'Prefill ms':>12}  {'tok/s':>10}")
        print("-" * 35)
        for r in results:
            print(f"{r['actual_tokens']:>8}  {r['avg_prefill_ms']:>12.1f}  {r['avg_tok_per_s']:>10.1f}")

    elif args.url_streaming and args.url_baseline:
        print(f"\n=== Baseline: {args.url_baseline} ===")
        baseline = run_sweep(args.url_baseline, prompt_lengths, args.n_warmup, args.n_iter)

        print(f"\n=== Streaming: {args.url_streaming} ===")
        streaming = run_sweep(args.url_streaming, prompt_lengths, args.n_warmup, args.n_iter)

        print(f"\n{'T':>8}  {'Baseline ms':>12}  {'Stream ms':>12}  {'Speedup':>8}")
        print("-" * 50)
        stream_map = {s["target_tokens"]: s for s in streaming}
        for b in baseline:
            s = stream_map.get(b["target_tokens"])
            if not s:
                continue
            speedup = b["avg_prefill_ms"] / s["avg_prefill_ms"] if s["avg_prefill_ms"] > 0 else 0
            print(f"{b['actual_tokens']:>8}  {b['avg_prefill_ms']:>12.1f}  "
                  f"{s['avg_prefill_ms']:>12.1f}  {speedup:>7.2f}x")
    else:
        parser.error("Provide --url or both --url-streaming and --url-baseline")


if __name__ == "__main__":
    main()
