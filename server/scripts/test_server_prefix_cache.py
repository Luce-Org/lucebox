"""End-to-end Phase A test: prefix cache integration.

Sends 3 chat completions sharing a 2K-token system prompt, asserts turns 2/3
have noticeably faster prefill than turn 1.

Run against a running server:
    python3 dflash/scripts/test_server_prefix_cache.py --url http://localhost:8000

Or spawn a fresh server (requires model files):
    python3 dflash/scripts/test_server_prefix_cache.py
"""
import argparse
import atexit
import json
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def parse_args():
    ap = argparse.ArgumentParser(description="Prefix cache integration test")
    ap.add_argument("--url", type=str, default=None,
                    help="Base URL of a running dflash_server (skips spawn)")
    return ap.parse_args()


def spawn_server():
    """Spawn a local dflash_server and return its base URL."""
    target = Path.home() / "models/qwen3.6-27b/Qwen3.6-27B-UD-Q4_K_XL.gguf"
    draft = Path.home() / "models/qwen3.6-27b-dflash"
    server_bin = ROOT / "dflash/build/dflash_server"

    if not target.exists() or not server_bin.exists() or not draft.exists():
        print(f"SKIP: prereqs missing (target={target.exists()} "
              f"draft={draft.exists()} bin={server_bin.exists()})")
        sys.exit(0)

    port = 18181
    log = open("/tmp/test_pc_server.log", "w")
    proc = subprocess.Popen(
        [str(server_bin), str(target),
         "--draft", str(draft),
         "--max-ctx", "4096", "--port", str(port),
         "--prefix-cache-slots", "2"],
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
            print("SERVER DIED; see /tmp/test_pc_server.log")
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
    base_url = args.url if args.url else spawn_server()

    # Large system prompt (~2K tokens) to make the prefill cost measurable.
    system = "You are a precise coding assistant. " * 200

    def chat(user_msg, max_tokens=8):
        payload = {
            "model": "luce-dflash",
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user_msg},
            ],
            "max_tokens": max_tokens, "stream": False,
        }
        body = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{base_url}/v1/chat/completions",
            data=body, headers={"Content-Type": "application/json"})
        t0 = time.time()
        resp = urllib.request.urlopen(req, timeout=600)
        data = json.loads(resp.read())
        dt = time.time() - t0
        return dt, data["choices"][0]["message"]["content"]

    # Turn 1: cold (cache miss → snapshot taken at end)
    print("\n=== Turn 1 (cold) ===", flush=True)
    t1, r1 = chat("What is 2+2?")
    print(f"latency={t1:.2f}s  reply={r1!r}")

    # Turn 2: same system prompt → cache HIT, only suffix prefilled
    print("\n=== Turn 2 (warm) ===", flush=True)
    t2, r2 = chat("What is the capital of France?")
    print(f"latency={t2:.2f}s  reply={r2!r}")

    # Turn 3: same system prompt, third user → still warm
    print("\n=== Turn 3 (warm) ===", flush=True)
    t3, r3 = chat("Tell me about Mars.")
    print(f"latency={t3:.2f}s  reply={r3!r}")

    # Verdict
    print("\n=== Verdict ===", flush=True)
    print(f"turn_1: {t1:.2f}s")
    print(f"turn_2: {t2:.2f}s  ratio_2/1={t2/t1:.2f}")
    print(f"turn_3: {t3:.2f}s  ratio_3/1={t3/t1:.2f}")
    # Expect turn 2 and 3 prefill to be much faster (5K system prompt cached).
    # Total wall is prefill + decode; decode is ~constant (small max_tokens).
    # Conservative gate: ratio < 0.85 (turn 2 should be at least 15% faster).
    ok = (t2 / t1) < 0.85 and (t3 / t1) < 0.85
    print("\nPASS" if ok else "FAIL: prefix cache did not visibly speed up subsequent turns")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
