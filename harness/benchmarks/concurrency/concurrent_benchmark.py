#!/usr/bin/env python3
"""Measure end-to-end output goodput and TTFT under concurrent streaming load."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import threading
import time
import urllib.request
from collections.abc import Iterable
from pathlib import Path
from typing import Any

PromptInput = str | list[dict[str, str]]
DEFAULT_CLIENT_LEVELS = (2, 4, 8, 16)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_prompts(path: Path) -> list[str]:
    prompts = []
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        if line.startswith("{"):
            value = json.loads(line).get("prompt")
            if not isinstance(value, str) or not value:
                raise ValueError(f"{path}:{line_no}: missing string 'prompt'")
            prompts.append(value)
        else:
            prompts.append(line)
    if not prompts:
        raise ValueError(f"{path}: no prompts")
    return prompts


def request_prompts(prompts: list[str], count: int, offset: int) -> list[str]:
    if offset < 0:
        raise ValueError("--prompt-offset must be >= 0")
    if offset + count > len(prompts):
        raise ValueError(
            f"need prompts [{offset}, {offset + count}), but only "
            f"{len(prompts)} were supplied; refusing to reuse prompts"
        )
    return prompts[offset:offset + count]


def iter_sse_data(lines: Iterable[bytes]) -> Iterable[str]:
    data: list[str] = []
    for raw in lines:
        line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
        if not line:
            if data:
                yield "\n".join(data)
                data.clear()
        elif line.startswith("data:"):
            data.append(line[5:].lstrip())
    if data:
        yield "\n".join(data)


def prompt_messages(prompt: PromptInput) -> list[dict[str, str]]:
    if isinstance(prompt, str):
        return [{"role": "user", "content": prompt}]
    if not isinstance(prompt, list) or not prompt or any(
        not isinstance(message, dict)
        or not isinstance(message.get("role"), str)
        or not isinstance(message.get("content"), str)
        for message in prompt
    ):
        raise ValueError("messages must contain role/content strings")
    return prompt


def prompt_input_sha256(prompt: PromptInput) -> str:
    if isinstance(prompt, str):
        return sha256_text(prompt)
    return sha256_text(json.dumps(prompt, ensure_ascii=False, separators=(",", ":")))


def native_prefill_values(usage: Any) -> tuple[int | None, float | None]:
    """Return exact Luce usage.timings fields, without protocol fallbacks."""
    if not isinstance(usage, dict):
        return None, None
    timings = usage.get("timings")
    if not isinstance(timings, dict):
        return None, None
    tokens = timings.get("prefilled_tokens")
    prefill_ms = timings.get("prefill_ms")
    if isinstance(tokens, bool) or not isinstance(tokens, int) or tokens < 0:
        tokens = None
    if (
        isinstance(prefill_ms, bool)
        or not isinstance(prefill_ms, (int, float))
        or not math.isfinite(prefill_ms)
        or prefill_ms < 0
    ):
        prefill_ms = None
    return tokens, float(prefill_ms) if prefill_ms is not None else None


def stream_request(args: argparse.Namespace, prompt: PromptInput) -> dict[str, Any]:
    started = time.perf_counter()
    first = None
    content: list[str] = []
    reasoning: list[str] = []
    completion_tokens = None
    prompt_tokens = None
    server_native_prefilled_tokens = None
    server_native_prefill_ms = None
    finish_reason = None
    done_received = False
    response_id = None
    error = None
    payload = {
        "model": args.model,
        "messages": prompt_messages(prompt),
        "stream": True,
        "stream_options": {"include_usage": True},
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "seed": args.seed,
    }
    if args.ignore_eos:
        payload["ignore_eos"] = True
    headers = {"Content-Type": "application/json"}
    if args.api_key:
        headers["Authorization"] = f"Bearer {args.api_key}"
    request = urllib.request.Request(
        args.base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=args.timeout) as response:
            for data in iter_sse_data(response):
                if data == "[DONE]":
                    done_received = True
                    break
                event = json.loads(data)
                if isinstance(event.get("id"), str):
                    response_id = event["id"]
                usage = event.get("usage") or {}
                if isinstance(usage.get("completion_tokens"), int):
                    completion_tokens = usage["completion_tokens"]
                if isinstance(usage.get("prompt_tokens"), int):
                    prompt_tokens = usage["prompt_tokens"]
                native_tokens, native_ms = native_prefill_values(usage)
                if native_tokens is not None:
                    server_native_prefilled_tokens = native_tokens
                if native_ms is not None:
                    server_native_prefill_ms = native_ms
                for choice in event.get("choices") or []:
                    if choice.get("finish_reason") is not None:
                        finish_reason = choice["finish_reason"]
                    delta = choice.get("delta") or {}
                    piece = delta.get("content")
                    thought = delta.get("reasoning_content")
                    if isinstance(piece, str) and piece:
                        first = first or time.perf_counter()
                        content.append(piece)
                    if isinstance(thought, str) and thought:
                        first = first or time.perf_counter()
                        reasoning.append(thought)
    except Exception as exc:  # preserve partial timing/output for diagnosis
        error = f"{type(exc).__name__}: {exc}"
    if error is None and not done_received:
        error = "ProtocolError: stream ended before [DONE]"
    elif error is None and finish_reason is None:
        error = "ProtocolError: stream ended without a terminal finish_reason"
    ended = time.perf_counter()
    output = "".join(content)
    reasoning_output = "".join(reasoning)
    decode_duration = ended - first if first is not None and ended > first else None
    request_decode_tok_s = (
        (completion_tokens - 1) / decode_duration
        if isinstance(completion_tokens, int) and completion_tokens > 0
        and decode_duration is not None else None
    )
    server_native_prefill_tokens_per_s = (
        server_native_prefilled_tokens * 1000.0 / server_native_prefill_ms
        if server_native_prefilled_tokens is not None
        and server_native_prefill_ms is not None
        and server_native_prefill_ms > 0 else None
    )
    return {
        "t_start": started, "t_first": first, "t_end": ended,
        "duration_s": ended - started,
        "ttft_s": first - started if first is not None else None,
        "decode_duration_s": decode_duration,
        "completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens,
        "server_native_prefilled_tokens": server_native_prefilled_tokens,
        "server_native_prefill_ms": server_native_prefill_ms,
        "server_native_prefill_tokens_per_s": server_native_prefill_tokens_per_s,
        "finish_reason": finish_reason, "done_received": done_received, "error": error,
        "response_id": response_id,
        "content_sha256": sha256_text(output),
        "reasoning_content_sha256": sha256_text(reasoning_output),
        "content_chars": len(output), "reasoning_content_chars": len(reasoning_output),
        "request_output_tok_s": (
            completion_tokens / (ended - started)
            if completion_tokens is not None and ended > started else None
        ),
        "request_decode_tok_s": request_decode_tok_s,
    }


def run_level(
    clients: int, args: argparse.Namespace, prompts: list[str], offset: int,
) -> dict[str, Any]:
    selected = request_prompts(prompts, clients, offset)
    barrier = threading.Barrier(clients)
    records: list[dict[str, Any] | None] = [None] * clients

    def worker(index: int) -> None:
        barrier.wait()
        record = stream_request(args, selected[index])
        record["prompt_index"] = offset + index
        record["prompt_sha256"] = prompt_input_sha256(selected[index])
        records[index] = record

    threads = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(clients)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(args.timeout + 30)
    completed = [record for record in records if record is not None]
    hung = sum(thread.is_alive() for thread in threads)
    failures = hung + sum(record["error"] is not None for record in completed)
    ok = [record for record in completed if record["error"] is None]
    starts = [record["t_start"] for record in completed]
    ends = [record["t_end"] for record in completed]
    level_start = min(starts) if starts else time.perf_counter()
    wall = max(ends) - level_start if ends else 0.0
    for record in completed:
        record["start_offset_s"] = record["t_start"] - level_start

    completion_counts = [r["completion_tokens"] for r in ok]
    prompt_counts = [r["prompt_tokens"] for r in ok]
    completion_complete = bool(ok) and all(isinstance(v, int) for v in completion_counts)
    prompt_complete = bool(ok) and all(isinstance(v, int) for v in prompt_counts)
    ttfts = [r["ttft_s"] for r in ok if r["ttft_s"] is not None]
    first_window = (
        max(r["start_offset_s"] + r["ttft_s"] for r in ok)
        if len(ttfts) == len(ok) and ok else None
    )
    first_times = [r["t_first"] for r in ok if r["t_first"] is not None]
    output_window = (
        max(r["t_end"] for r in ok) - min(first_times)
        if len(first_times) == len(ok) and ok else None
    )
    request_decode_rates = [
        r["request_decode_tok_s"] for r in ok
        if r.get("request_decode_tok_s") is not None
    ]
    native_prefilled_counts = [
        r.get("server_native_prefilled_tokens") for r in ok
    ]
    native_prefill_ms_values = [r.get("server_native_prefill_ms") for r in ok]
    native_token_count_complete = bool(ok) and all(
        isinstance(value, int) and not isinstance(value, bool) and value >= 0
        for value in native_prefilled_counts
    )
    native_timing_complete = bool(ok) and all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        and math.isfinite(value) and value >= 0
        for value in native_prefill_ms_values
    )
    native_prefilled_tokens_total = (
        sum(native_prefilled_counts) if native_token_count_complete else None
    )
    native_prefill_ms_max = (
        max(native_prefill_ms_values) if native_timing_complete else None
    )
    native_prefill_window_ms = (
        max(r["start_offset_s"] * 1000.0 + r["server_native_prefill_ms"] for r in ok)
        if native_timing_complete else None
    )
    fixed_valid = (
        failures == 0 and len(ok) == clients
        and completion_complete
        and prompt_complete
        and all(v == args.max_tokens for v in completion_counts)
    ) if args.ignore_eos else None
    prompt_hashes = [r["prompt_sha256"] for r in ok]
    output_hashes = [
        [r["content_sha256"], r["reasoning_content_sha256"]] for r in ok
    ]
    digest = lambda value: sha256_text(json.dumps(value, separators=(",", ":")))
    return {
        "clients": clients, "requests": clients, "requests_ok": len(ok),
        "failures": failures, "wall_s": wall,
        "start_skew_s": max(starts) - min(starts) if starts else None,
        "completion_tokens_total": sum(completion_counts) if completion_complete else None,
        "token_count_complete": completion_complete,
        "fixed_token_workload_valid": fixed_valid,
        "aggregate_tok_s": (
            sum(completion_counts) / wall if completion_complete and wall > 0 else None
        ),
        "aggregate_metric": "completion_tokens_per_level_wall_second",
        "output_window_s": output_window,
        "output_window_tok_s": (
            sum(completion_counts) / output_window
            if completion_complete and output_window is not None and output_window > 0
            else None
        ),
        "output_window_metric": "completion_tokens_per_first_output_to_final_completion_second",
        "request_decode_tok_s_median": (
            statistics.median(request_decode_rates)
            if len(request_decode_rates) == len(ok) and ok else None
        ),
        "prompt_tokens_total": sum(prompt_counts) if prompt_complete else None,
        "prompt_tokens_min": min(prompt_counts) if prompt_complete else None,
        "prompt_tokens_max": max(prompt_counts) if prompt_complete else None,
        "prompt_tokens_distinct": len(set(prompt_counts)) if prompt_complete else None,
        "prompt_token_count_complete": prompt_complete,
        "prompt_to_first_token_s": first_window,
        "prompt_tokens_per_s_to_first_token": (
            sum(prompt_counts) / first_window
            if prompt_complete and first_window is not None and first_window > 0 else None
        ),
        "server_native_prefilled_tokens_total": native_prefilled_tokens_total,
        "server_native_prefill_ms_max": native_prefill_ms_max,
        "server_native_prefill_window_ms": native_prefill_window_ms,
        "server_native_prefill_token_count_complete": native_token_count_complete,
        "server_native_prefill_timing_complete": native_timing_complete,
        "server_native_prefill_tokens_per_s": (
            native_prefilled_tokens_total * 1000.0 / native_prefill_window_ms
            if native_prefilled_tokens_total is not None
            and native_prefill_window_ms is not None
            and native_prefill_window_ms > 0 else None
        ),
        "server_native_prefill_metric": (
            "sum_prefilled_tokens_per_common_origin_prefill_window_second"
        ),
        "ttft_median_s": statistics.median(ttfts) if ttfts else None,
        "ttft_max_s": max(ttfts) if ttfts else None,
        "selected_prompt_set_sha256": digest(prompt_hashes),
        "selected_output_set_sha256": digest(output_hashes),
        "requests_detail": completed,
    }


def fmt(value: Any, spec: str = ".2f") -> str:
    return format(value, spec) if isinstance(value, (int, float)) else "n/a"


def markdown(report: dict[str, Any]) -> str:
    lines = [
        f"# Concurrent benchmark — {report['label']}", "",
        "| C | Ok | Output goodput tok/s | Output-window tok/s | "
        "Request decode tok/s | Prompt tok/s to first | Native prefill tok/s | "
        "Native prefill window ms | Prompt range | TTFT median s | TTFT max s | Wall s |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- | "
        "---: | ---: | ---: |",
    ]
    for level in report["levels"]:
        lines.append(
            f"| {level['clients']} | {level['requests_ok']}/{level['requests']} | "
            f"{fmt(level['aggregate_tok_s'])} | "
            f"{fmt(level['output_window_tok_s'])} | "
            f"{fmt(level['request_decode_tok_s_median'])} | "
            f"{fmt(level['prompt_tokens_per_s_to_first_token'])} | "
            f"{fmt(level.get('server_native_prefill_tokens_per_s'))} | "
            f"{fmt(level.get('server_native_prefill_window_ms'), '.1f')} | "
            f"{fmt(level['prompt_tokens_min'], '.0f')}–{fmt(level['prompt_tokens_max'], '.0f')} | "
            f"{fmt(level['ttft_median_s'], '.3f')} | {fmt(level['ttft_max_s'], '.3f')} | "
            f"{fmt(level['wall_s'])} |"
        )
    return "\n".join(lines) + "\n"


def level_failed(level: dict[str, Any], ignore_eos: bool) -> bool:
    return bool(
        level["failures"]
        or not level["token_count_complete"]
        or not level["prompt_token_count_complete"]
        or (ignore_eos and level["fixed_token_workload_valid"] is not True)
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:18080/v1")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--model", default="luce-dflash")
    parser.add_argument("--clients", type=int, action="append", dest="client_levels")
    parser.add_argument("--prompt-file", type=Path, required=True)
    parser.add_argument("--prompt-offset", type=int, default=0)
    parser.add_argument("--require-distinct-prompts", action="store_true",
                        help="Compatibility flag; this client always refuses reuse")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--cooldown", type=float, default=0.0)
    parser.add_argument("--server-metadata-json", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--label", default="")
    return parser


def run(args: argparse.Namespace) -> int:
    levels = args.client_levels or list(DEFAULT_CLIENT_LEVELS)
    if any(level < 1 for level in levels):
        raise ValueError("--clients must be positive")
    if len(set(levels)) != len(levels):
        raise ValueError("--clients levels must be distinct")
    if args.prompt_offset < 0 or args.max_tokens < 1 or args.timeout <= 0:
        raise ValueError("invalid offset, max-tokens, or timeout")
    prompts = load_prompts(args.prompt_file)
    results = []
    offset = args.prompt_offset
    for index, clients in enumerate(levels):
        if index and args.cooldown > 0:
            time.sleep(args.cooldown)
        print(f"[bench] C={clients} max_tokens={args.max_tokens}", flush=True)
        results.append(run_level(clients, args, prompts, offset))
        offset += clients
    metadata = (
        json.loads(args.server_metadata_json.read_text(encoding="utf-8"))
        if args.server_metadata_json else {}
    )
    report = {
        "schema_version": 2, "label": args.label, "base_url": args.base_url,
        "model": args.model, "max_tokens": args.max_tokens,
        "temperature": args.temperature, "seed": args.seed,
        "ignore_eos": args.ignore_eos, "prompt_offset": args.prompt_offset,
        "client_levels": levels,
        "prompt_file_sha256": hashlib.sha256(args.prompt_file.read_bytes()).hexdigest(),
        "server_metadata": metadata, "levels": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(markdown(report), end="")
    bad = any(level_failed(level, args.ignore_eos) for level in results)
    return 1 if bad else 0


def main() -> int:
    try:
        return run(build_parser().parse_args())
    except Exception as exc:
        print(f"[bench] error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
