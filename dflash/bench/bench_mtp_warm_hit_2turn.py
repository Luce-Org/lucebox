#!/usr/bin/env python3
"""R3: 2-turn MTP WARM-hit integration bench.

Sends the same prompt twice to the daemon-backed OpenAI-compatible server and
asserts:
  1. turn2.prefill_s < turn1.prefill_s * gate_ratio (default 0.3) — the WARM
     restore path actually engaged (otherwise turn2 reruns cold prefill).
  2. The first N (default 16) emitted tokens are byte-equal between turns —
     proves the head_kv restore did not silently mangle the chain runner.

Usage:
  python3 dflash/bench/bench_mtp_warm_hit_2turn.py \\
      --server http://127.0.0.1:8000 \\
      --prompt heron-24k.txt \\
      --n-gen 32 --gate-ratio 0.3 --check-prefix 16

Run a daemon-fronting server first with prefix cache + MTP enabled:
  python3 dflash/scripts/server.py --host 127.0.0.1 --port 8000 \\
      --target ${MTP_GGUF} --mtp-gguf ${MTP_GGUF} --mtp-gamma 3 \\
      --bin dflash/build/test_dflash --max-ctx 32768 --fa-window 2048 \\
      --cache-type-k tq3_0 --cache-type-v tq3_0
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path
from typing import Tuple


def post_chat(server: str, prompt: str, n_gen: int, seed: int) -> Tuple[float, list]:
    """POST a single completion, return (server_prefill_s, token_ids)."""
    body = json.dumps({
        "model": "qwen3.5-mtp",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": n_gen,
        "temperature": 0.0,
        "seed": seed,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        url=f"{server.rstrip('/')}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=600) as resp:
        payload = json.loads(resp.read())
    wall = time.monotonic() - t0

    # Server reports prefill_s in usage.dflash if available; fall back to wall.
    usage = payload.get("usage") or {}
    dflash = usage.get("dflash") or {}
    prefill_s = float(dflash.get("prefill_s") or wall)

    # Token ids are the deterministic comparison surface. Fall back to the
    # decoded text if the server doesn't expose ids.
    choice0 = (payload.get("choices") or [{}])[0]
    ids = (choice0.get("dflash") or {}).get("token_ids") or []
    if not ids:
        ids = list((choice0.get("message") or {}).get("content") or "")
    return prefill_s, ids


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", default="http://127.0.0.1:8000")
    ap.add_argument("--prompt", required=True, type=Path,
                    help="Path to plain-text prompt file (will be sent as user content)")
    ap.add_argument("--n-gen", type=int, default=32)
    ap.add_argument("--gate-ratio", type=float, default=0.3,
                    help="turn2.prefill_s must be < turn1 * this to pass")
    ap.add_argument("--check-prefix", type=int, default=16,
                    help="Number of leading tokens that must be byte-equal between turns")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    prompt = args.prompt.read_text()

    print(f"[bench] turn 1 (cold) -> {args.server}", flush=True)
    p1, ids1 = post_chat(args.server, prompt, args.n_gen, args.seed)
    print(f"[bench] turn 1 prefill_s={p1:.3f}s tokens={len(ids1)}", flush=True)

    print(f"[bench] turn 2 (warm) -> {args.server}", flush=True)
    p2, ids2 = post_chat(args.server, prompt, args.n_gen, args.seed)
    print(f"[bench] turn 2 prefill_s={p2:.3f}s tokens={len(ids2)}", flush=True)

    # Gate 1: prefill time
    ratio = p2 / p1 if p1 > 0 else float("inf")
    gate_prefill_ok = ratio < args.gate_ratio
    print(f"[bench] prefill_ratio={ratio:.3f} (gate < {args.gate_ratio}): "
          f"{'PASS' if gate_prefill_ok else 'FAIL'}", flush=True)

    # Gate 2: byte-equal prefix
    n = min(args.check_prefix, len(ids1), len(ids2))
    prefix_match = ids1[:n] == ids2[:n]
    print(f"[bench] first {n} tokens bit-equal: "
          f"{'PASS' if prefix_match else 'FAIL'}", flush=True)
    if not prefix_match:
        print(f"  turn1[:{n}]={ids1[:n]}", flush=True)
        print(f"  turn2[:{n}]={ids2[:n]}", flush=True)

    if gate_prefill_ok and prefix_match:
        print("[bench] R3 GATE: PASS", flush=True)
        return 0
    print("[bench] R3 GATE: FAIL", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
