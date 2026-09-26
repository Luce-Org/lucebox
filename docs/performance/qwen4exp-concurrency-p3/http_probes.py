#!/usr/bin/env python3
"""HTTP concurrency, capacity, cancellation, context, and goodput probes."""
from __future__ import annotations

import concurrent.futures
import http.client
import json
import os
import statistics
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

BASE = os.environ.get("P3_URL", "http://127.0.0.1:8935").rstrip("/")
HOST = BASE.removeprefix("http://").removeprefix("https://")
HOST, _, PORT = HOST.partition(":")
PORT = int(PORT or (443 if BASE.startswith("https:") else 80))


def _payload(prompt: str, max_tokens: int = 128, *, thinking: bool | None = None) -> dict:
    obj = {
        "model": "qwen4exp",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if thinking is not None:
        obj["chat_template_kwargs"] = {"enable_thinking": thinking}
    return obj


def stream(prompt: str, max_tokens: int = 128, *, barrier=None, close_after_first=False, on_closed=None):
    body = json.dumps(_payload(prompt, max_tokens)).encode()
    req = urllib.request.Request(
        BASE + "/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )
    if barrier:
        barrier.wait()
    t0 = time.perf_counter()
    first = None
    text = ""
    usage = {}
    status = 0
    try:
        response = urllib.request.urlopen(req, timeout=600)
    except urllib.error.HTTPError as exc:
        return {"status": exc.code, "error": exc.read().decode(errors="replace"), "ttft_s": None, "wall_s": time.perf_counter()-t0}
    try:
        status = response.status
        for raw in response:
            line = raw.decode(errors="replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                obj = json.loads(data)
            except json.JSONDecodeError:
                continue
            if obj.get("usage"):
                usage = obj["usage"]
            choices = obj.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                piece = delta.get("content") or delta.get("reasoning_content") or ""
                if piece:
                    if first is None:
                        first = time.perf_counter()
                    text += str(piece)
                    if close_after_first:
                        response.close()
                        if on_closed:
                            on_closed.set()
                        return {"status": status, "text": text, "ttft_s": first-t0, "wall_s": first-t0, "completion_tokens": 1, "disconnected": True}
    except (ConnectionResetError, http.client.IncompleteRead, urllib.error.URLError) as exc:
        if on_closed:
            on_closed.set()
        return {"status": status or 0, "text": text, "error": repr(exc), "ttft_s": (first-t0 if first else None), "wall_s": time.perf_counter()-t0, "disconnected": True}
    finally:
        response.close()
    end = time.perf_counter()
    return {
        "status": status,
        "text": text,
        "ttft_s": (first-t0 if first else None),
        "wall_s": end-t0,
        "completion_tokens": int(usage.get("completion_tokens") or 0),
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
    }


def ask_correct(prompt: str, expected: str, max_tokens=96, barrier=None):
    result = stream(prompt, max_tokens, barrier=barrier)
    result["correct"] = result.get("status") == 200 and expected.lower() in result.get("text", "").lower()
    return result


def run_distinct_waves():
    prompts = [
        ("What is 13 times 17? Reply with the number.", "221"),
        ("What is the capital of France? Reply with the city.", "paris"),
        ("Which planet is largest in our solar system? Reply with the name.", "jupiter"),
        ("What color does red and white paint make? Reply with the color.", "pink"),
    ]
    all_rows = []
    for wave in range(3):
        barrier = threading.Barrier(4)
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(ask_correct, p, e, 96, barrier) for p, e in prompts]
            rows = [f.result() for f in futures]
        ok = all(r.get("correct") for r in rows)
        print(f"[http-distinct] wave={wave+1} pass={ok} rows=" + json.dumps(rows), flush=True)
        all_rows.append({"wave": wave+1, "pass": ok, "rows": rows})
    return all(r["pass"] for r in all_rows), all_rows


def run_fifth_slot():
    # Long answers keep the first four streams active while the fifth arrives.
    prompts = [
        "Write a detailed 350-word explanation of why the sky appears blue.",
        "Write a detailed 350-word explanation of how rain forms.",
        "Write a detailed 350-word explanation of how plants use sunlight.",
        "Write a detailed 350-word explanation of how tides work.",
        "Answer with one word: what is 2 + 2?",
    ]
    barrier = threading.Barrier(5)
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        futures = [pool.submit(stream, p, 512 if i < 4 else 16, barrier=barrier) for i, p in enumerate(prompts)]
        rows = [f.result() for f in futures]
    first4_ok = all(r.get("status") == 200 and r.get("text") for r in rows[:4])
    fifth = rows[4]
    clean_fifth = (fifth.get("status") == 200 and fifth.get("text")) or fifth.get("status") in (400, 409, 429, 503)
    result = {"first_four_ok": bool(first4_ok), "fifth_clean_reject_or_complete": bool(clean_fifth), "rows": rows}
    print("[http-fifth] " + json.dumps(result), flush=True)
    return bool(first4_ok and clean_fifth), result


def run_overflow_isolation():
    huge = "hello " * 42000
    barrier = threading.Barrier(2)
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        bad = pool.submit(stream, huge, 8, barrier=barrier)
        good = pool.submit(ask_correct, "What is the capital of France? Reply with the city.", "paris", 96, barrier)
        overflow, valid = bad.result(), good.result()
    pass_gate = overflow.get("status", 0) in (400, 413, 422) and valid.get("correct", False)
    result = {"overflow_status": overflow.get("status"), "overflow_error": overflow.get("error", "")[-400:], "valid_request_correct": valid.get("correct"), "pass": pass_gate}
    print("[http-overflow] " + json.dumps(result), flush=True)
    return pass_gate, result


def run_disconnect_isolation():
    prompts = [
        ("Write a detailed 350-word explanation of why the sky appears blue.", None),
        ("What is the capital of France? Reply with the city.", "paris"),
        ("Which planet is largest in our solar system? Reply with the name.", "jupiter"),
        ("What color does red and white paint make? Reply with the color.", "pink"),
    ]
    barrier = threading.Barrier(4)
    closed = threading.Event()
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as pool:
        f0 = pool.submit(stream, prompts[0][0], 512, barrier=barrier, close_after_first=True, on_closed=closed)
        others = [pool.submit(ask_correct, p, expected, 96, barrier) for p, expected in prompts[1:]]
        if not closed.wait(timeout=180):
            return False, {"error": "disconnect stream did not receive a token"}
        replacement = pool.submit(ask_correct, "What is 13 times 17? Reply with the number.", "221", 96)
        disconnected = f0.result(timeout=180)
        survivors = [f.result(timeout=600) for f in others]
        replaced = replacement.result(timeout=600)
    result = {"disconnected_request_status": disconnected.get("status"), "survivors_correct": [r.get("correct") for r in survivors], "replacement_correct": replaced.get("correct")}
    pass_gate = all(result["survivors_correct"]) and bool(result["replacement_correct"])
    result["pass"] = pass_gate
    print("[http-disconnect] " + json.dumps(result), flush=True)
    return pass_gate, result


def run_goodput():
    prompt = "Write a comma-separated sequence of positive integers starting at 1. Continue until the response limit; no explanation."

    def wave(n):
        barrier = threading.Barrier(n)
        t0 = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
            rows = [f.result() for f in [pool.submit(stream, prompt, 128, barrier=barrier) for _ in range(n)]]
        elapsed = time.perf_counter() - t0
        firsts = [r["ttft_s"] for r in rows if r.get("ttft_s") is not None]
        ends = [r["wall_s"] for r in rows]
        toks = [r.get("completion_tokens", 0) for r in rows]
        span = max(ends) - min(firsts) if firsts else elapsed
        aggregate = sum(toks) / span if span > 0 else 0.0
        return {"n": n, "wall_s": elapsed, "aggregate_decode_tok_s": aggregate, "total_tokens": sum(toks), "ttft_s": [r.get("ttft_s") for r in rows], "per_request_tok_s": [r.get("completion_tokens", 0)/(r["wall_s"]-r["ttft_s"]) if r.get("ttft_s") is not None and r["wall_s"] > r["ttft_s"] else 0 for r in rows], "completion_tokens": toks, "statuses": [r.get("status") for r in rows]}

    arms = {"n1": [wave(1) for _ in range(3)], "n4": [wave(4) for _ in range(3)]}
    for key, rows in arms.items():
        rates = [r["aggregate_decode_tok_s"] for r in rows]
        print(f"[goodput] {key} median_aggregate_tok_s={statistics.median(rates):.3f} trials=" + json.dumps(rows), flush=True)
    return arms


def main():
    results = {}
    results["distinct_pass"], results["distinct"] = run_distinct_waves()
    results["fifth_pass"], results["fifth"] = run_fifth_slot()
    results["overflow_pass"], results["overflow"] = run_overflow_isolation()
    results["disconnect_pass"], results["disconnect"] = run_disconnect_isolation()
    results["goodput"] = run_goodput()
    out = Path(os.environ.get("P3_HTTP_JSON", "http-probes.json"))
    out.write_text(json.dumps(results, indent=2) + "\n")
    print("[http-probes] summary=" + json.dumps({k: results[k] for k in ("distinct_pass", "fifth_pass", "overflow_pass", "disconnect_pass")}), flush=True)
    return 0 if all(results[k] for k in ("distinct_pass", "fifth_pass", "overflow_pass", "disconnect_pass")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
