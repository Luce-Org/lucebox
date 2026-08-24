#!/usr/bin/env python3
"""Reproducible Ling 3.0 Flash checks against a running dflash_server.

The script uses only the Python standard library. It records server-reported
prefill/decode timings, a short context-depth sweep, request-level concurrency,
and a real OpenAI-style tool-call round trip.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path


DECODE_PROMPT = (
    "Continue this exact sequence until the output limit. Output only the word "
    "BETA separated by single spaces: BETA BETA BETA BETA BETA BETA BETA BETA"
)


def post_json(url: str, payload: dict, timeout: float) -> dict:
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, separators=(",", ":")).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.load(response)


def chat(base_url: str, model: str, messages: list[dict], max_tokens: int,
         timeout: float, *, tools: list[dict] | None = None) -> dict:
    payload: dict = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": False,
    }
    if tools is not None:
        payload["tools"] = tools
    started = time.monotonic()
    result = post_json(
        f"{base_url.rstrip('/')}/v1/chat/completions", payload, timeout)
    result["_wall_seconds"] = time.monotonic() - started
    return result


def measurement(result: dict) -> dict:
    usage = result["usage"]
    timings = usage.get("timings", {})
    content = result["choices"][0]["message"].get("content") or ""
    return {
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "prefill_ms": timings.get("prefill_ms"),
        "prefill_tokens": timings.get(
            "effective_prompt_tokens", usage["prompt_tokens"]),
        "decode_ms": timings.get("decode_ms"),
        "decode_tokens_per_second": timings.get("decode_tokens_per_sec"),
        "cache_hit": timings.get("cache_hit"),
        "cached_prefix_tokens": timings.get("cached_prefix_tokens", 0),
        "wall_seconds": result["_wall_seconds"],
        "finish_reason": result["choices"][0].get("finish_reason"),
        "output_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "output_excerpt": content[:160],
    }


def median(values: list[float | int | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return statistics.median(present) if present else None


def run_decode(args: argparse.Namespace) -> dict:
    for _ in range(args.warmups):
        chat(args.url, args.model,
             [{"role": "user", "content": DECODE_PROMPT}],
             args.decode_tokens, args.timeout)

    runs = [
        measurement(chat(
            args.url, args.model,
            [{"role": "user", "content": DECODE_PROMPT}],
            args.decode_tokens, args.timeout))
        for _ in range(args.decode_runs)
    ]
    return {
        "prompt": DECODE_PROMPT,
        "temperature": 0,
        "max_tokens": args.decode_tokens,
        "warmups": args.warmups,
        "runs": runs,
        "median_decode_tokens_per_second": median(
            [run["decode_tokens_per_second"] for run in runs]),
        "median_prefill_ms": median([run["prefill_ms"] for run in runs]),
    }


def synthetic_context(target_tokens: int, run: int) -> str:
    # " alpha" is one token in the Ling tokenizer. The fixed chat framing and
    # instruction account for the small difference between target and actual.
    body = "alpha " * max(1, target_tokens - 32)
    return (
        "Read this synthetic context, then reply with only OK.\n"
        f"{body}\nBenchmark run {run}; reply OK."
    )


def run_context_sweep(args: argparse.Namespace) -> list[dict]:
    rows = []
    for target in args.context_tokens:
        runs = []
        for run_index in range(args.context_runs):
            result = chat(
                args.url, args.model,
                [{"role": "user",
                  "content": synthetic_context(target, run_index)}],
                1, args.timeout)
            row = measurement(result)
            if row["prefill_ms"]:
                row["prefill_tokens_per_second"] = (
                    row["prefill_tokens"] / (row["prefill_ms"] / 1000.0))
            else:
                row["prefill_tokens_per_second"] = None
            runs.append(row)
        rows.append({
            "target_tokens": target,
            "runs": runs,
            "actual_prompt_tokens": int(median(
                [run["prompt_tokens"] for run in runs]) or 0),
            "median_prefill_ms": median([run["prefill_ms"] for run in runs]),
            "median_prefill_tokens_per_second": median(
                [run["prefill_tokens_per_second"] for run in runs]),
        })
    return rows


def run_concurrency(args: argparse.Namespace) -> list[dict]:
    rows = []
    for level in args.concurrency_levels:
        started = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=level) as pool:
            futures = [
                pool.submit(
                    chat, args.url, args.model,
                    [{"role": "user", "content":
                      f"{DECODE_PROMPT}\nConcurrent request {index}."}],
                    args.concurrency_tokens, args.timeout)
                for index in range(level)
            ]
            runs = [measurement(future.result()) for future in futures]
        wall_seconds = time.monotonic() - started
        completion_tokens = sum(run["completion_tokens"] for run in runs)
        rows.append({
            "concurrency": level,
            "max_tokens_per_request": args.concurrency_tokens,
            "wall_seconds": wall_seconds,
            "completion_tokens": completion_tokens,
            "aggregate_completion_tokens_per_second": (
                completion_tokens / wall_seconds if wall_seconds else None),
            "requests": runs,
        })
    return rows


def run_tool_round_trip(args: argparse.Namespace) -> dict:
    tools = [{
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city.",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                },
                "required": ["location"],
                "additionalProperties": False,
            },
        },
    }]
    user_message = {
        "role": "user",
        "content": "What is the current weather in Rome? Use the tool.",
    }
    first = chat(
        args.url, args.model, [user_message], 96, args.timeout, tools=tools)
    choice = first["choices"][0]
    assistant = choice["message"]
    calls = assistant.get("tool_calls") or []
    if choice.get("finish_reason") != "tool_calls" or len(calls) != 1:
        raise RuntimeError(f"expected one tool call, got: {choice}")
    function = calls[0]["function"]
    arguments = json.loads(function["arguments"])
    if function["name"] != "get_weather" or "rome" not in (
            arguments.get("location") or "").lower():
        raise RuntimeError(f"unexpected tool call: {function}")

    tool_message = {
        "role": "tool",
        "tool_call_id": calls[0]["id"],
        "content": json.dumps({
            "temperature_c": 29,
            "condition": "sunny",
            "source": "test fixture",
        }, separators=(",", ":")),
    }
    second = chat(
        args.url, args.model,
        [user_message, assistant, tool_message],
        96, args.timeout, tools=tools)
    answer = second["choices"][0]["message"].get("content") or ""
    if "29" not in answer or "sunny" not in answer.lower():
        raise RuntimeError(f"tool result was not used in the answer: {answer}")
    return {
        "call_finish_reason": choice.get("finish_reason"),
        "tool_name": function["name"],
        "arguments": arguments,
        "call_measurement": measurement(first),
        "answer": answer,
        "answer_measurement": measurement(second),
        "passed": True,
    }


def csv_ints(value: str) -> list[int]:
    values = [int(item) for item in value.split(",") if item.strip()]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18081")
    parser.add_argument("--model", default="ling-3.0-flash-lucebox")
    parser.add_argument("--weights-quant", default="Q4_K_M")
    parser.add_argument("--kv-quant", default="Q4_0")
    parser.add_argument("--server-max-context", type=int, default=32768)
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument("--decode-runs", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--context-tokens", type=csv_ints,
                        default=csv_ints("256,1024,4096,8192"))
    parser.add_argument("--context-runs", type=int, default=2)
    parser.add_argument("--concurrency-levels", type=csv_ints,
                        default=csv_ints("1,2"))
    parser.add_argument("--concurrency-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    try:
        with urllib.request.urlopen(
                f"{args.url.rstrip('/')}/health", timeout=10) as response:
            health = response.read().decode()
        report = {
            "engine": "LuceBox dflash_server",
            "model": args.model,
            "weights_quant": args.weights_quant,
            "kv_quant": args.kv_quant,
            "server_max_context": args.server_max_context,
            "health": health,
            "decode": run_decode(args),
            "context_sweep": run_context_sweep(args),
            "concurrency": run_concurrency(args),
            "tool_round_trip": run_tool_round_trip(args),
        }
    except (KeyError, TypeError, ValueError, RuntimeError,
            urllib.error.URLError) as error:
        print(f"benchmark failed: {error}")
        return 1

    rendered = json.dumps(report, indent=2, sort_keys=True)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
