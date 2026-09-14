#!/usr/bin/env python3
"""Verify the Lucebox Gemma 4 RTX 4090 backend.

The verifier checks two things:
  1. OpenAI-compatible chat returns non-empty text from the requested GGUF.
  2. llama.cpp request timings meet the single-stream decode floor.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from typing import Any


PROMPTS = [
    "Explain speculative decoding for local LLM inference in one paragraph.",
    "Give three practical RTX 4090 tuning tips for a GGUF model.",
    "Describe why flash attention matters for single-stream generation.",
]


def post_json(base_url: str, path: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def get_json(base_url: str, path: str, timeout: float) -> dict[str, Any]:
    with urllib.request.urlopen(base_url.rstrip("/") + path, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def wait_health(base_url: str, timeout_s: float) -> None:
    start = time.time()
    last_error = ""
    while time.time() - start < timeout_s:
        try:
            get_json(base_url, "/health", timeout=2)
            return
        except (OSError, urllib.error.URLError, json.JSONDecodeError) as exc:
            last_error = str(exc)
            time.sleep(1)
    raise RuntimeError(f"server did not become healthy within {timeout_s:.0f}s: {last_error}")


def verify_chat(base_url: str, request_timeout: float) -> str:
    data = post_json(
        base_url,
        "/v1/chat/completions",
        {
            "model": "lucebox-gemma4-31b-4090",
            "messages": [
                {
                    "role": "user",
                    "content": "Reply in one sentence: what GPU is this Lucebox Gemma backend tuned for?",
                }
            ],
            "max_tokens": 80,
            "temperature": 0,
            "stream": False,
        },
        timeout=request_timeout,
    )
    content = data["choices"][0]["message"].get("content") or ""
    content = content.strip()
    if not content:
        raise RuntimeError("chat completion returned empty content")
    return content


def run_decode_probe(base_url: str, prompt: str, n_predict: int, request_timeout: float) -> dict[str, Any]:
    data = post_json(
        base_url,
        "/completion",
        {
            "prompt": prompt,
            "n_predict": n_predict,
            "temperature": 0,
            "cache_prompt": False,
            "stream": False,
        },
        timeout=request_timeout,
    )
    content = (data.get("content") or "").strip()
    timings = data.get("timings") or {}
    predicted_per_second = timings.get("predicted_per_second")
    if not content:
        raise RuntimeError("completion returned empty content")
    if not isinstance(predicted_per_second, (int, float)):
        raise RuntimeError(f"completion response lacks timings.predicted_per_second: {data}")
    return {
        "content_prefix": content[:120],
        "predicted_n": timings.get("predicted_n"),
        "predicted_ms": timings.get("predicted_ms"),
        "predicted_per_second": float(predicted_per_second),
        "prompt_per_second": timings.get("prompt_per_second"),
        "draft_n": timings.get("draft_n"),
        "draft_n_accepted": timings.get("draft_n_accepted"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:18191")
    parser.add_argument("--threshold", type=float, default=60.0)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--n-predict", type=int, default=256)
    parser.add_argument("--wait", type=float, default=300.0)
    parser.add_argument("--request-timeout", type=float, default=180.0)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    wait_health(args.base_url, args.wait)
    chat_reply = verify_chat(args.base_url, args.request_timeout)

    results = []
    for i in range(args.runs):
        results.append(
            run_decode_probe(
                args.base_url,
                PROMPTS[i % len(PROMPTS)],
                args.n_predict,
                args.request_timeout,
            )
        )

    rates = [r["predicted_per_second"] for r in results]
    summary = {
        "base_url": args.base_url,
        "chat_reply": chat_reply,
        "threshold": args.threshold,
        "all_ge_threshold": all(rate >= args.threshold for rate in rates),
        "min_predicted_per_second": min(rates),
        "avg_predicted_per_second": sum(rates) / len(rates),
        "results": results,
    }

    text = json.dumps(summary, indent=2)
    print(text)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            f.write(text)
            f.write("\n")

    if not summary["all_ge_threshold"]:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
