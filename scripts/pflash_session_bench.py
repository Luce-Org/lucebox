#!/usr/bin/env python3
"""
PFlash A/B benchmark using multi-turn session fixtures.

Sends the largest multi-turn case that fits within the server's context
window to http://localhost:8080 and reports prefill_s + decode_tps.
Intended for comparing pflash=off vs pflash=on (DFLASH_PREFILL_MODE=compress).

Usage:
    python3 scripts/pflash_session_bench.py
    python3 scripts/pflash_session_bench.py --bucket 65536   # pick specific bucket
    python3 scripts/pflash_session_bench.py --url http://localhost:8080
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

# multi_turn_cases.json is relative to this script's repo root
_REPO_ROOT = Path(__file__).parent.parent
_FIXTURE = _REPO_ROOT / "luce-bench/src/lucebench/fixtures/agent_recorded/multi_turn_cases.json"


def load_cases() -> list[dict]:
    if not _FIXTURE.exists():
        print(f"[error] fixture not found: {_FIXTURE}", file=sys.stderr)
        sys.exit(1)
    data = json.loads(_FIXTURE.read_text())
    return data.get("cases", [])


def pick_case(cases: list[dict], bucket: int | None, max_ctx: int) -> dict | None:
    """Pick the largest case that fits under max_ctx tokens."""
    eligible = [c for c in cases if c["target_bucket_tokens"] <= max_ctx]
    if not eligible:
        return None
    if bucket is not None:
        for c in eligible:
            if c["target_bucket_tokens"] == bucket:
                return c
        print(f"[warn] bucket {bucket} not found; using largest eligible", file=sys.stderr)
    return max(eligible, key=lambda c: c["target_bucket_tokens"])


def get_max_ctx(url: str) -> int:
    try:
        req = urllib.request.Request(f"{url}/props")
        with urllib.request.urlopen(req, timeout=10) as resp:
            props = json.loads(resp.read())
        return int(props.get("max_ctx", 131072))
    except Exception as e:
        print(f"[warn] /props failed ({e}); assuming max_ctx=98304", file=sys.stderr)
        return 98304


def run_case(url: str, case: dict, max_tokens: int = 256) -> dict:
    body = {
        "model": "dflash",
        "messages": case["messages"],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }
    payload = json.dumps(body).encode()
    req = urllib.request.Request(
        f"{url}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            raw = json.loads(resp.read())
    except Exception as e:
        return {"error": str(e), "wall_s": time.perf_counter() - t0}
    wall = time.perf_counter() - t0

    usage = raw.get("usage") or {}
    choice = (raw.get("choices") or [{}])[0]
    msg = (choice.get("message") or {}) if isinstance(choice, dict) else {}
    content = (msg.get("content") or "") + (msg.get("reasoning_content") or "")

    return {
        "wall_s": wall,
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "content_chars": len(content),
        "finish_reason": (choice.get("finish_reason") if isinstance(choice, dict) else None),
        "error": None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--bucket", type=int, default=None,
                    help="Target bucket tokens (8192/16384/32768/65536/102400/131072); default: largest that fits")
    ap.add_argument("--max-tokens", type=int, default=256, help="Max completion tokens (default 256)")
    ap.add_argument("--repeat", type=int, default=1, help="Number of runs")
    ap.add_argument("--list", action="store_true", help="List available cases and exit")
    args = ap.parse_args()

    cases = load_cases()

    if args.list:
        print("Available multi-turn cases:")
        for c in cases:
            print(f"  bucket={c['target_bucket_tokens']:7d}  msgs={len(c.get('messages', []))}  "
                  f"approx_tokens={c.get('context_tokens_approx', '?')}")
        return

    max_ctx = get_max_ctx(args.url)
    print(f"[info] server max_ctx={max_ctx}")

    case = pick_case(cases, args.bucket, max_ctx)
    if case is None:
        print(f"[error] no case fits under max_ctx={max_ctx}", file=sys.stderr)
        sys.exit(1)

    n_msgs = len(case.get("messages", []))
    approx = case.get("context_tokens_approx", "?")
    bucket = case["target_bucket_tokens"]
    print(f"[info] case: bucket={bucket} msgs={n_msgs} approx_tokens={approx}")
    print(f"[info] max_tokens={args.max_tokens} repeat={args.repeat}")

    results = []
    for i in range(args.repeat):
        if args.repeat > 1:
            print(f"[run {i+1}/{args.repeat}]", flush=True)
        result = run_case(args.url, case, max_tokens=args.max_tokens)
        results.append(result)
        if result.get("error"):
            print(f"  error: {result['error']}")
        else:
            wall = result["wall_s"]
            in_tok = result["prompt_tokens"]
            out_tok = result["completion_tokens"]
            tps = out_tok / wall if wall > 0 else 0
            print(f"  wall={wall:.1f}s  in={in_tok}  out={out_tok}  "
                  f"decode={tps:.1f}tok/s  chars={result['content_chars']}  "
                  f"finish={result['finish_reason']}")

    if args.repeat > 1 and results:
        walls = [r["wall_s"] for r in results if not r.get("error")]
        if walls:
            avg = sum(walls) / len(walls)
            print(f"\n[summary] mean_wall={avg:.1f}s over {len(walls)} runs")


if __name__ == "__main__":
    main()
