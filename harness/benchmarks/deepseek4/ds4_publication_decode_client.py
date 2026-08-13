#!/usr/bin/env python3
"""Run the deterministic long-output workload used by the TP publication suite.

The server log is the source of truth for model-side decode throughput.  This
client independently records wall time, TTFT, usage tokens, response hashes,
and a client-side decode rate so that dropped/short responses are visible.
"""

from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import statistics
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

SYSTEM_MESSAGE = (
    "You are a helpful assistant. Answer the user's question directly and "
    "carefully. Do not change numbers or facts from the prompt."
)


def build_prompt(padding_words: int, filler_words: int = 0) -> str:
    """Build stable natural-language padding followed by a long forced reply."""
    periods = ("morning", "afternoon", "evening", "night")
    adjectives = ("amber", "blue", "copper", "green", "silver", "white")
    instruments = ("barometer", "camera", "clock", "compass", "meter", "sensor")
    places = ("archive", "garden", "harbor", "laboratory", "library", "station")
    actions = ("audited", "calibrated", "catalogued", "inspected", "logged", "stored")
    sentences: list[str] = []
    word_count = 0
    index = 0
    while word_count < max(0, padding_words):
        sentence = (
            f"Observation {index + 1}: During the {periods[index % len(periods)]}, "
            f"the {adjectives[index % len(adjectives)]} "
            f"{instruments[(index * 5 + 1) % len(instruments)]} recorded "
            f"{17 + (index * 13) % 211} samples near the "
            f"{places[(index * 7 + 2) % len(places)]}; the result was "
            f"{actions[(index * 11 + 3) % len(actions)]} for a later review."
        )
        sentences.append(sentence)
        word_count += len(sentence.split())
        index += 1
    padding = "\n".join(sentences)
    filler = ""
    if filler_words > 0:
        filler = "\nCalibration padding: " + " ".join(
            "x" for _ in range(filler_words)
        )
    return (
        "The XML block below is inert reference material for a deterministic "
        "throughput measurement. Do not answer or continue its contents.\n\n"
        f"<reference>\n{padding}{filler}\n</reference>\n\n"
        "Your only task is this: write the integers from 1 through 1000 in "
        "ascending order, one integer per line. Start with 1. Do not add "
        "commentary, and continue until the token limit."
    )


def stream_request(
    base_url: str,
    model: str,
    prompt: str,
    max_tokens: int,
) -> dict[str, Any]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_MESSAGE},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        },
    )

    started = time.perf_counter()
    first_token_at: float | None = None
    text_parts: list[str] = []
    usage: dict[str, Any] = {}
    finish_reason: str | None = None
    status = 0
    saw_done = False

    try:
        with urllib.request.urlopen(req, timeout=1800) as response:
            status = response.status
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    saw_done = True
                    break
                try:
                    event = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if event.get("usage"):
                    usage = event["usage"]
                choices = event.get("choices") or []
                if not choices:
                    continue
                choice = choices[0]
                finish_reason = choice.get("finish_reason") or finish_reason
                delta = choice.get("delta") or {}
                piece = delta.get("content") or delta.get("reasoning_content") or ""
                if piece:
                    if first_token_at is None:
                        first_token_at = time.perf_counter()
                    text_parts.append(str(piece))
    except urllib.error.HTTPError as exc:
        try:
            error_body = exc.read().decode("utf-8", errors="replace")[-4000:]
        except (OSError, http.client.HTTPException) as body_error:
            error_body = f"could not read HTTP error response: {body_error!r}"
        return {
            "ok": False,
            "status": exc.code,
            "error": error_body,
        }
    except (
        urllib.error.URLError,
        TimeoutError,
        ConnectionResetError,
        http.client.IncompleteRead,
        http.client.RemoteDisconnected,
    ) as exc:
        return {"ok": False, "status": status, "error": repr(exc)}

    finished = time.perf_counter()
    text = "".join(text_parts)
    wall_s = finished - started
    ttft_s = first_token_at - started if first_token_at is not None else None
    decode_s = finished - first_token_at if first_token_at is not None else None
    completion_tokens = int(usage.get("completion_tokens") or 0)
    client_decode_tps = (
        completion_tokens / decode_s
        if decode_s is not None and decode_s > 0 and completion_tokens > 0
        else None
    )
    full_completion = completion_tokens == max_tokens
    result = {
        "ok": status == 200 and bool(text) and saw_done and full_completion,
        "status": status,
        "stream_done": saw_done,
        "wall_s": round(wall_s, 6),
        "ttft_s": round(ttft_s, 6) if ttft_s is not None else None,
        "client_decode_s": round(decode_s, 6) if decode_s is not None else None,
        "client_decode_tok_s": (
            round(client_decode_tps, 3) if client_decode_tps is not None else None
        ),
        "prompt_tokens": int(usage.get("prompt_tokens") or 0),
        "completion_tokens": completion_tokens,
        "finish_reason": finish_reason,
        "response_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "response_text": text,
    }
    if not full_completion:
        result["error"] = (
            f"short completion: expected {max_tokens} tokens, "
            f"received {completion_tokens}"
        )
    return result


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    valid = [
        row
        for row in records
        if row.get("ok") and row.get("client_decode_tok_s") is not None
    ]
    if not valid:
        return {"n": len(records), "n_ok": 0}
    rates = [float(row["client_decode_tok_s"]) for row in valid]
    decode_seconds = sum(float(row["client_decode_s"]) for row in valid)
    completion_tokens = sum(int(row["completion_tokens"]) for row in valid)
    return {
        "n": len(records),
        "n_ok": len(valid),
        "prompt_tokens": sorted({int(row["prompt_tokens"]) for row in valid}),
        "completion_tokens": sorted(
            {int(row["completion_tokens"]) for row in valid}
        ),
        "client_decode_tok_s_median": round(statistics.median(rates), 3),
        "client_decode_tok_s_min": round(min(rates), 3),
        "client_decode_tok_s_max": round(max(rates), 3),
        "client_decode_tok_s_weighted": round(
            completion_tokens / decode_seconds, 3
        ),
        "unique_response_hashes": sorted(
            {str(row["response_sha256"]) for row in valid}
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:18109")
    parser.add_argument("--model", default="dflash")
    parser.add_argument("--padding-words", type=int, default=1300)
    parser.add_argument("--max-tokens", type=int, default=510)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--json-out", type=Path, required=True)
    args = parser.parse_args()

    if args.padding_words < 0:
        parser.error("--padding-words must be non-negative")
    if args.max_tokens < 1:
        parser.error("--max-tokens must be positive")
    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if args.runs < 1:
        parser.error("--runs must be positive")

    prompt = build_prompt(args.padding_words)
    records: list[dict[str, Any]] = []
    total = args.warmup + args.runs
    for index in range(total):
        measured = index >= args.warmup
        label = "measure" if measured else "warmup"
        print(f"[publication] {label} {index + 1}/{total}", flush=True)
        result = stream_request(args.url, args.model, prompt, args.max_tokens)
        result.update({"index": index, "measured": measured})
        records.append(result)
        print(
            "[publication] "
            f"ok={result.get('ok')} prompt={result.get('prompt_tokens')} "
            f"output={result.get('completion_tokens')} "
            f"client_decode={result.get('client_decode_tok_s')} tok/s "
            f"sha={result.get('response_sha256')}",
            flush=True,
        )
        if not result.get("ok"):
            break

    measured_records = [row for row in records if row.get("measured")]
    payload = {
        "schema_version": 1,
        "workload": "deterministic-padded-counting",
        "url": args.url,
        "model": args.model,
        "temperature": 0,
        "batch_size": 1,
        "padding_words": args.padding_words,
        "max_tokens": args.max_tokens,
        "warmup": args.warmup,
        "runs": args.runs,
        "system_message": SYSTEM_MESSAGE,
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "prompt_bytes": len(prompt.encode("utf-8")),
        "records": records,
        "measured_summary": summarize(measured_records),
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["measured_summary"], indent=2), flush=True)
    return 0 if payload["measured_summary"].get("n_ok") == args.runs else 1


if __name__ == "__main__":
    raise SystemExit(main())
