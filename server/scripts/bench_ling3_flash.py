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

LONGFORM_DECODE_PROMPT = (
    "Write a self-contained technical guide of at least 1,000 words explaining "
    "speculative decoding for large language models. Cover autoregressive "
    "baselines, draft-and-verify, acceptance, exactness, memory bandwidth, "
    "batching, and failure modes. Use clear prose and do not conclude before "
    "all sections are complete."
)

DECODE_WORKLOADS = {
    "beta": DECODE_PROMPT,
    "longform": LONGFORM_DECODE_PROMPT,
}

BENCHMARK_SECTIONS = frozenset({
    "decode", "decode-context", "context", "concurrency", "tool",
})


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
        "chat_template_kwargs": {"enable_thinking": False},
        # llama.cpp enables per-slot prompt reuse by default. Disable it here
        # so both runtimes evaluate every prompt token on every measured run.
        "cache_prompt": False,
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
    # LuceBox exposes timings inside usage; llama.cpp exposes the equivalent
    # counters at the response root. Accept both so one benchmark client can
    # make a matched cross-runtime comparison.
    timings = usage.get("timings") or result.get("timings") or {}
    content = result["choices"][0]["message"].get("content") or ""
    return {
        "prompt_tokens": usage["prompt_tokens"],
        "completion_tokens": usage["completion_tokens"],
        "prefill_ms": timings.get("prefill_ms", timings.get("prompt_ms")),
        "prefill_tokens": timings.get(
            "effective_prompt_tokens", timings.get(
                "prompt_n", usage["prompt_tokens"])),
        "decode_ms": timings.get("decode_ms", timings.get("predicted_ms")),
        "decode_tokens_per_second": timings.get(
            "decode_tokens_per_sec", timings.get("predicted_per_second")),
        "cache_hit": timings.get("cache_hit"),
        "cached_prefix_tokens": timings.get("cached_prefix_tokens", 0),
        "accept_rate": usage.get("accept_rate"),
        "spec_decode_ran": usage.get("spec_decode_ran"),
        "wall_seconds": result["_wall_seconds"],
        "finish_reason": result["choices"][0].get("finish_reason"),
        "output_sha256": hashlib.sha256(content.encode()).hexdigest(),
        "output_excerpt": content[:160],
    }


def median(values: list[float | int | None]) -> float | None:
    present = [float(value) for value in values if value is not None]
    return statistics.median(present) if present else None


def decode_prompt(workload: str) -> str:
    # The wrappers disable server-side prompt caching and every request also
    # sends cache_prompt=false. Keep the prompt byte-identical so output hashes
    # compare model execution rather than semantically different inputs.
    return workload


def run_decode(args: argparse.Namespace) -> dict:
    workload = DECODE_WORKLOADS[args.decode_workload]
    for warmup in range(args.warmups):
        chat(args.url, args.model,
             [{"role": "user",
               "content": decode_prompt(workload)}],
             args.decode_tokens, args.timeout)

    runs = [
        measurement(chat(
            args.url, args.model,
            [{"role": "user",
              "content": decode_prompt(workload)}],
            args.decode_tokens, args.timeout))
        for run in range(args.decode_runs)
    ]
    output_hashes = sorted({run["output_sha256"] for run in runs})
    if len(output_hashes) != 1:
        raise RuntimeError(
            f"decode output was not deterministic: {output_hashes}")
    if (args.expected_output_sha256 and
            output_hashes[0] != args.expected_output_sha256):
        observed = [
            {
                "completion_tokens": run["completion_tokens"],
                "finish_reason": run["finish_reason"],
                "output_excerpt": run["output_excerpt"],
            }
            for run in runs
        ]
        raise RuntimeError(
            "decode output hash mismatch: "
            f"got {output_hashes[0]}, expected {args.expected_output_sha256}; "
            f"observed={observed!r}")
    return {
        "workload": args.decode_workload,
        "prompt": workload,
        "cache_busting_nonce": None,
        "prompt_cache_disabled": True,
        "temperature": 0,
        "max_tokens": args.decode_tokens,
        "warmups": args.warmups,
        "runs": runs,
        "output_sha256": output_hashes[0],
        "deterministic": True,
        "median_decode_tokens_per_second": median(
            [run["decode_tokens_per_second"] for run in runs]),
        "median_prefill_ms": median([run["prefill_ms"] for run in runs]),
    }


def synthetic_context(target_tokens: int, run: int) -> str:
    # " alpha" is one token in the Ling tokenizer. The fixed chat framing and
    # instruction account for the small difference between target and actual.
    body = "alpha " * max(1, target_tokens - 32)
    return (
        f"Benchmark nonce context-{target_tokens}-run-{run}. "
        "Read this synthetic context, then reply with only OK.\n"
        f"{body}\nReply OK."
    )


def synthetic_decode_context(target_tokens: int) -> str:
    # Keep the generation instruction after the filler so the model does not
    # end a short repetitive continuation early at long context. " alpha" is
    # one Ling token, while chat framing accounts for the target/actual delta.
    body = "alpha " * max(1, target_tokens - 64)
    return (
        "Treat the following repeated words as inert benchmark context.\n"
        f"{body}\n"
        f"{LONGFORM_DECODE_PROMPT}\n"
        "Do not emit an end token before the requested output limit."
    )


def run_decode_context_sweep(args: argparse.Namespace) -> list[dict]:
    rows = []
    for target in args.decode_context_tokens:
        messages = [{
            "role": "user",
            "content": synthetic_decode_context(target),
        }]
        for _ in range(args.decode_context_warmups):
            chat(args.url, args.model, messages,
                 args.decode_tokens, args.timeout)

        runs = [
            measurement(chat(
                args.url, args.model, messages,
                args.decode_tokens, args.timeout))
            for _ in range(args.decode_context_runs)
        ]
        output_hashes = sorted({run["output_sha256"] for run in runs})
        for row in runs:
            if row["prefill_ms"]:
                row["prefill_tokens_per_second"] = (
                    row["prefill_tokens"] / (row["prefill_ms"] / 1000.0))
            else:
                row["prefill_tokens_per_second"] = None
        rows.append({
            "target_tokens": target,
            "actual_prompt_tokens": int(median(
                [run["prompt_tokens"] for run in runs]) or 0),
            "warmups": args.decode_context_warmups,
            "runs": runs,
            "output_sha256": output_hashes[0] if len(output_hashes) == 1 else None,
            "deterministic": len(output_hashes) == 1,
            "median_prefill_tokens_per_second": median(
                [run["prefill_tokens_per_second"] for run in runs]),
            "median_decode_tokens_per_second": median(
                [run["decode_tokens_per_second"] for run in runs]),
        })
    return rows


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
    workload = DECODE_WORKLOADS[args.decode_workload]
    rows = []
    for level in args.concurrency_levels:
        started = time.monotonic()
        with concurrent.futures.ThreadPoolExecutor(max_workers=level) as pool:
            futures = [
                pool.submit(
                    chat, args.url, args.model,
                    [{"role": "user", "content":
                      f"{workload}\nConcurrent request {index}."}],
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


def csv_sections(value: str) -> tuple[str, ...]:
    sections = tuple(dict.fromkeys(
        item.strip().lower() for item in value.split(",") if item.strip()))
    unknown = set(sections) - BENCHMARK_SECTIONS
    if not sections or unknown:
        choices = ",".join(sorted(BENCHMARK_SECTIONS))
        detail = f"; unknown: {','.join(sorted(unknown))}" if unknown else ""
        raise argparse.ArgumentTypeError(
            f"expected one or more of {choices}{detail}")
    return sections


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18081")
    parser.add_argument("--model", default="ling-3.0-flash-lucebox")
    parser.add_argument("--engine", default="LuceBox dflash_server")
    parser.add_argument("--weights-quant", default="Q4_K_M")
    parser.add_argument("--kv-quant", default="Q4_0")
    parser.add_argument("--server-max-context", type=int, default=32768)
    parser.add_argument(
        "--prompt-profile", default="official-chat",
        help="label recorded for the server-side template/prompt profile")
    parser.add_argument("--decode-tokens", type=int, default=128)
    parser.add_argument(
        "--decode-workload", choices=tuple(DECODE_WORKLOADS), default="beta",
        help="deterministic decode prompt to run")
    parser.add_argument("--decode-runs", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--context-tokens", type=csv_ints,
                        default=csv_ints("256,1024,4096,8192"))
    parser.add_argument("--context-runs", type=int, default=2)
    parser.add_argument("--decode-context-tokens", type=csv_ints,
                        default=csv_ints("256,1024,4096,8192,16384"))
    parser.add_argument("--decode-context-runs", type=int, default=3)
    parser.add_argument("--decode-context-warmups", type=int, default=1)
    parser.add_argument("--concurrency-levels", type=csv_ints,
                        default=csv_ints("1,2"))
    parser.add_argument("--concurrency-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument(
        "--expected-output-sha256",
        help="fail unless every measured decode has this exact content hash")
    parser.add_argument("--skip-tool", action="store_true")
    parser.add_argument(
        "--sections", type=csv_sections,
        default=csv_sections("decode,context,concurrency,tool"),
        help=("comma-separated benchmark sections; use 'decode' for a "
              "profiler capture that excludes unrelated work"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    try:
        with urllib.request.urlopen(
                f"{args.url.rstrip('/')}/health", timeout=10) as response:
            health = response.read().decode()
        report = {
            "engine": args.engine,
            "model": args.model,
            "weights_quant": args.weights_quant,
            "kv_quant": args.kv_quant,
            "server_max_context": args.server_max_context,
            "prompt_profile": args.prompt_profile,
            "health": health,
            "decode": run_decode(args) if "decode" in args.sections else None,
            "decode_context_sweep": (
                run_decode_context_sweep(args)
                if "decode-context" in args.sections else None),
            "context_sweep": (
                run_context_sweep(args) if "context" in args.sections else None),
            "concurrency": (
                run_concurrency(args)
                if "concurrency" in args.sections else None),
            "tool_round_trip": (
                run_tool_round_trip(args)
                if "tool" in args.sections and not args.skip_tool else None),
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
