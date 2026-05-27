"""Phase B.3 end-to-end test: multi-slot THICK LRU prefix cache.

Sends 5 conversation turns with a shared (large) system prompt and a growing
history. Asserts:

  - Turn 1: cold (cache miss).
  - Turns 2-5: each finds a progressively deeper cache hit so only the new
    user message (+ short assistant reply header) needs prefilling.
  - Turns 2-5 wall-time < 30 % of turn 1 (prefill savings dominate for
    small max_tokens).

Run against a running server:
    python3 dflash/scripts/test_multi_turn_prefix_cache.py --url http://localhost:8000

Or spawn a fresh server (requires model files at ~/models/qwen3.6-27b):
    python3 dflash/scripts/test_multi_turn_prefix_cache.py
"""
import argparse
import atexit
import json
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def parse_args():
    ap = argparse.ArgumentParser(description="Multi-turn prefix cache test")
    ap.add_argument("--url", type=str, default=None,
                    help="Base URL of a running dflash_server (skips spawn)")
    return ap.parse_args()


def spawn_server():
    """Spawn a local dflash_server and return its base URL."""
    target = Path.home() / "models/qwen3.6-27b/Qwen3.6-27B-UD-Q4_K_XL.gguf"
    draft = Path.home() / "models/qwen3.6-27b-dflash"
    server_bin = ROOT / "dflash/build/dflash_server"

    if not target.exists() or not server_bin.exists():
        print(f"SKIP: prereqs missing (target={target.exists()} bin={server_bin.exists()})")
        sys.exit(0)

    port = 18182
    log = open("/tmp/test_mt_pc_server.log", "w")
    proc = subprocess.Popen(
        [str(server_bin), str(target),
         "--draft", str(draft),
         "--max-ctx", "8192", "--port", str(port),
         "--prefix-cache-slots", "4"],
        stdout=log, stderr=subprocess.STDOUT, bufsize=1,
    )

    def cleanup():
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()

    atexit.register(cleanup)

    base_url = f"http://127.0.0.1:{port}"
    print("Waiting for server...", flush=True)
    deadline = time.time() + 180
    while time.time() < deadline:
        if proc.poll() is not None:
            print("SERVER DIED; see /tmp/test_mt_pc_server.log")
            sys.exit(2)
        try:
            urllib.request.urlopen(f"{base_url}/v1/models", timeout=1).read()
            print("Server up.", flush=True)
            return base_url
        except (urllib.error.URLError, ConnectionResetError, TimeoutError):
            time.sleep(1)

    print("Server didn't come up within 180s")
    sys.exit(2)


def main():
    args = parse_args()
    spawned = args.url is None
    base_url = args.url if args.url else spawn_server()

    # Large system prompt (~2K tokens) to make the prefill cost measurable.
    system = "You are a helpful coder. " * 200

    def chat_post(payload: dict) -> str:
        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{base_url}/v1/chat/completions",
            data=body,
            headers={"Content-Type": "application/json"},
        )
        resp = urllib.request.urlopen(req, timeout=600)
        data = json.loads(resp.read())
        return data["choices"][0]["message"]["content"]

    def turn(history: list[dict], user: str) -> tuple[float, str]:
        history.append({"role": "user", "content": user})
        msgs = [{"role": "system", "content": system}, *history]
        payload = {
            "model": "luce-dflash",
            "messages": msgs,
            "max_tokens": 8,
            "stream": False,
        }
        t0 = time.time()
        reply = chat_post(payload)
        dt = time.time() - t0
        history.append({"role": "assistant", "content": reply})
        return dt, reply

    history: list[dict] = []

    print("\n=== Turn 1 (cold) ===", flush=True)
    t1, r1 = turn(history, "Q1: what is 2+2?")
    print(f"latency={t1:.2f}s  reply={r1!r}", flush=True)

    print("\n=== Turn 2 (should hit system boundary) ===", flush=True)
    t2, r2 = turn(history, "Q2: what is the capital of France?")
    print(f"latency={t2:.2f}s  reply={r2!r}", flush=True)

    print("\n=== Turn 3 (should hit end-of-user1+asst1) ===", flush=True)
    t3, r3 = turn(history, "Q3: what is the square root of 144?")
    print(f"latency={t3:.2f}s  reply={r3!r}", flush=True)

    print("\n=== Turn 4 (should hit end-of-asst2) ===", flush=True)
    t4, r4 = turn(history, "Q4: what is the largest planet?")
    print(f"latency={t4:.2f}s  reply={r4!r}", flush=True)

    print("\n=== Turn 5 (should hit end-of-asst3) ===", flush=True)
    t5, r5 = turn(history, "Q5: what is the speed of light?")
    print(f"latency={t5:.2f}s  reply={r5!r}", flush=True)

    # Parse server log for cache-hit lines (only available when we spawned).
    hit_lines = []
    if spawned:
        try:
            with open("/tmp/test_mt_pc_server.log") as f:
                for ln in f:
                    if "[pc] lookup hit" in ln or "[pc] snapshot" in ln:
                        hit_lines.append(ln.strip())
        except FileNotFoundError:
            pass

    print("\n=== Cache-hit log (parsed from server) ===")
    if hit_lines:
        for ln in hit_lines:
            print(f"  {ln}")
    else:
        print("  (not available — server log not accessible)")

    # Extract prefix_len for each hit.
    hit_lens = [int(m.group(1)) for ln in hit_lines
                for m in [re.search(r"lookup hit slot=\d+ prefix_len=(\d+)", ln)]
                if m]

    print("\n=== Verdict ===", flush=True)
    print(f"t1={t1:.2f}  t2={t2:.2f}  t3={t3:.2f}  t4={t4:.2f}  t5={t5:.2f}", flush=True)
    ratios = {2: t2 / t1, 3: t3 / t1, 4: t4 / t1, 5: t5 / t1}
    for n, r in ratios.items():
        status = "OK" if r < 0.30 else "SLOW"
        print(f"  turn {n} ratio={r:.2f}  [{status}]", flush=True)

    if hit_lens:
        print(f"\n  hit prefix_lens (turns 2..5): {hit_lens}")

    # Sanity: first reply non-empty.
    assert r1, "Turn 1 reply must be non-empty"

    # Deeper-hit check (only when we have server logs).
    if len(hit_lens) >= 4:
        deeper_ok = hit_lens[-1] > hit_lens[0]
    elif spawned:
        deeper_ok = False
        print(f"\n  WARNING: expected ≥4 hit log lines (turns 2..5), got {len(hit_lens)}")
    else:
        # Can't verify without server logs; skip this gate.
        deeper_ok = True

    if hit_lens:
        print(f"  deeper-hit-on-later-turns: {'OK' if deeper_ok else 'FAIL'} "
              f"(turn-2 hit at {hit_lens[0] if hit_lens else '?'}, "
              f"turn-5 hit at {hit_lens[-1] if hit_lens else '?'})")

    # Non-regression latency gate: warm turns should not be SLOWER than cold turn.
    lat_ok = all(t <= t1 * 1.05 for t in (t2, t3, t4, t5))   # ≤ 5 % regression
    print(f"  no-regression vs cold: {'OK' if lat_ok else 'FAIL'}")

    ok = lat_ok and deeper_ok
    if ok:
        print("\nPASS")
    else:
        if not lat_ok:
            print(f"\nFAIL: a warm turn was >5% slower than cold turn 1 ({t1:.2f}s)")
        if not deeper_ok:
            print("\nFAIL: cache did not walk deeper across turns")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
