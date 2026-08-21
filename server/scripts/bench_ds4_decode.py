#!/usr/bin/env python3
"""Run the controlled DeepSeek4 speculative-decode throughput benchmark."""

import argparse
import hashlib
import json
import sys
import time
import urllib.error
import urllib.request

PROMPT = (
    "Continue this exact sequence indefinitely. Output only the word BETA "
    "separated by single spaces and never stop before the token limit: "
    "BETA BETA BETA BETA BETA BETA BETA BETA"
)


def is_beta_sequence(content: str) -> bool:
    words = content.split()
    return bool(words) and all(word == "BETA" for word in words)


def run_request(url: str, model: str, max_tokens: int, timeout: float) -> dict:
    body = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": PROMPT}],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": False,
        },
        separators=(",", ":"),
    ).encode()
    request = urllib.request.Request(
        f"{url.rstrip('/')}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )

    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        result = json.load(response)
    wall_seconds = time.monotonic() - started

    choice = result["choices"][0]
    content = choice["message"]["content"]
    usage = result["usage"]
    timings = usage.get("timings", {})
    return {
        "completion_tokens": usage["completion_tokens"],
        "decode_seconds": timings.get("decode_ms", 0.0) / 1000.0,
        "decode_tokens_per_second": timings.get("decode_tokens_per_sec"),
        "accept_rate": usage.get("accept_rate"),
        "spec_decode_ran": usage.get("spec_decode_ran"),
        "cache_hit": timings.get("cache_hit"),
        "cached_prefix_tokens": timings.get("cached_prefix_tokens"),
        "finish_reason": choice.get("finish_reason"),
        "output_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "output_matches_prompt": is_beta_sequence(content),
        "wall_seconds": wall_seconds,
    }


def validate_run(run: dict, max_tokens: int) -> None:
    if run["completion_tokens"] != max_tokens:
        raise RuntimeError(
            f"expected {max_tokens} completion tokens, got "
            f"{run['completion_tokens']}"
        )
    if not run["output_matches_prompt"]:
        raise RuntimeError("model output did not contain only the requested BETA sequence")
    if run["cache_hit"] or run["cached_prefix_tokens"] not in (None, 0):
        raise RuntimeError("benchmark request reused a cached prefix")
    if run["spec_decode_ran"] is not True:
        raise RuntimeError("speculative decode did not run")
    throughput = run["decode_tokens_per_second"]
    if throughput is None or throughput <= 0:
        raise RuntimeError("server did not report positive decode throughput")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run one warm-up and three fresh 512-token BETA requests against "
            "an OpenAI-compatible dflash_server."
        )
    )
    parser.add_argument("--url", default="http://127.0.0.1:8016")
    parser.add_argument("--model", default="dflash")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=900.0)
    args = parser.parse_args()

    if args.max_tokens <= 0 or args.warmups < 0 or args.runs <= 0:
        parser.error("max-tokens and runs must be positive; warmups cannot be negative")

    try:
        warmups = [
            run_request(args.url, args.model, args.max_tokens, args.timeout)
            for _ in range(args.warmups)
        ]
        runs = []
        for _ in range(args.runs):
            run = run_request(args.url, args.model, args.max_tokens, args.timeout)
            validate_run(run, args.max_tokens)
            runs.append(run)
    except (KeyError, TypeError, ValueError, urllib.error.URLError, RuntimeError) as error:
        print(f"benchmark failed: {error}", file=sys.stderr)
        return 1

    output_digests = {run["output_sha256"] for run in runs}
    if len(output_digests) != 1:
        print("benchmark failed: measured outputs were not byte-identical", file=sys.stderr)
        return 1

    print(
        json.dumps(
            {
                "request": {
                    "model": args.model,
                    "prompt": PROMPT,
                    "temperature": 0,
                    "max_tokens": args.max_tokens,
                    "stream": False,
                },
                "warmups": warmups,
                "runs": runs,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
