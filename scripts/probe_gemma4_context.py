#!/usr/bin/env python3
"""Probe long-context stability for the Lucebox Gemma 4 backend."""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from typing import Any


def marker_unit(cache_type_k: str, cache_type_v: str) -> str:
    return (
        f"Lucebox {cache_type_k} key cache and {cache_type_v} value cache "
        "stability marker for RTX 4090 CUDA flash attention and Gemma 4 MTP decoding. "
    )


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


def tokenize(base_url: str, content: str, timeout: float) -> int:
    data = post_json(
        base_url,
        "/tokenize",
        {"content": content, "add_special": False},
        timeout=timeout,
    )
    tokens = data.get("tokens")
    if not isinstance(tokens, list):
        raise RuntimeError(f"unexpected /tokenize response: {data}")
    return len(tokens)


def build_prompt(base_url: str, target_tokens: int, timeout: float, unit: str) -> tuple[str, int]:
    unit_tokens = max(1, tokenize(base_url, unit, timeout))
    repetitions = max(1, target_tokens // unit_tokens)
    prompt = unit * repetitions
    count = tokenize(base_url, prompt, timeout)

    while count < target_tokens:
        prompt += unit * max(1, (target_tokens - count) // unit_tokens)
        count = tokenize(base_url, prompt, timeout)

    while repetitions > 1 and count > target_tokens + unit_tokens:
        repetitions -= max(1, (count - target_tokens) // unit_tokens)
        prompt = unit * repetitions
        count = tokenize(base_url, prompt, timeout)

    return prompt, count


def run_target(
    base_url: str,
    target_tokens: int,
    max_tokens: int,
    timeout: float,
    cache_type_k: str,
    cache_type_v: str,
) -> dict[str, Any]:
    prompt, raw_user_tokens = build_prompt(
        base_url,
        target_tokens,
        timeout,
        marker_unit(cache_type_k, cache_type_v),
    )
    data = post_json(
        base_url,
        "/v1/chat/completions",
        {
            "model": "lucebox-gemma4-31b-4090",
            "messages": [
                {
                    "role": "user",
                    "content": (
                        prompt
                        + (
                            "\nReturn five concise numbered observations confirming whether the "
                            f"{cache_type_k}/{cache_type_v} K/V context remained stable."
                        )
                    ),
                }
            ],
            "max_tokens": max_tokens,
            "temperature": 0,
            "stream": False,
        },
        timeout=timeout,
    )
    content = (data["choices"][0]["message"].get("content") or "").strip()
    timings = data.get("timings") or {}
    usage = data.get("usage") or {}
    return {
        "target_raw_user_tokens": target_tokens,
        "raw_user_tokens": raw_user_tokens,
        "ok": bool(content),
        "usage_prompt_tokens": usage.get("prompt_tokens"),
        "usage_completion_tokens": usage.get("completion_tokens"),
        "prompt_n": timings.get("prompt_n"),
        "prompt_per_second": timings.get("prompt_per_second"),
        "predicted_n": timings.get("predicted_n"),
        "predicted_per_second": timings.get("predicted_per_second"),
        "draft_n": timings.get("draft_n"),
        "draft_n_accepted": timings.get("draft_n_accepted"),
        "content_prefix": content[:180],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:18191")
    parser.add_argument("--ctx", type=int, default=40960)
    parser.add_argument("--targets", default="8192,16384,32768,38912")
    parser.add_argument("--cache-type-k", default="q8_0")
    parser.add_argument("--cache-type-v", default="q8_0")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--wait", type=float, default=300.0)
    parser.add_argument("--request-timeout", type=float, default=240.0)
    parser.add_argument("--json-out", default=None)
    args = parser.parse_args()

    targets = [int(item.strip()) for item in args.targets.split(",") if item.strip()]
    wait_health(args.base_url, args.wait)
    results = [
        run_target(
            args.base_url,
            target,
            args.max_tokens,
            args.request_timeout,
            args.cache_type_k,
            args.cache_type_v,
        )
        for target in targets
    ]
    rates = [
        float(r["predicted_per_second"])
        for r in results
        if isinstance(r.get("predicted_per_second"), (int, float))
    ]
    summary = {
        "base_url": args.base_url,
        "ctx": args.ctx,
        "cache_type_k": args.cache_type_k,
        "cache_type_v": args.cache_type_v,
        "threshold": args.threshold,
        "all_ok": all(r["ok"] for r in results),
        "all_ge_threshold": (all(rate >= args.threshold for rate in rates) if args.threshold > 0 else None),
        "min_predicted_per_second": min(rates) if rates else None,
        "avg_predicted_per_second": (sum(rates) / len(rates) if rates else None),
        "results": results,
    }

    text = json.dumps(summary, indent=2)
    print(text)
    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            f.write(text)
            f.write("\n")

    if not summary["all_ok"]:
        return 1
    if args.threshold > 0 and not summary["all_ge_threshold"]:
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
