"""End-to-end Phase A test: spin up server.py with --prefix-cache-slots=2,
send 3 chat completions sharing a 2K-token system prompt, assert turns 2/3
have noticeably faster prefill than turn 1.

Prereqs: model files at ~/models/qwen3.6-27b/Qwen3.6-27B-UD-Q4_K_XL.gguf and
~/models/qwen3.6-27b-dflash/model.safetensors. Skipped if missing.

Run: python3 dflash/scripts/test_server_prefix_cache.py
"""
import os, sys, time, json, signal, subprocess, urllib.request, urllib.error
from pathlib import Path

ROOT  = Path(__file__).resolve().parent.parent.parent
TARGET = Path.home() / "models/qwen3.6-27b/Qwen3.6-27B-UD-Q4_K_XL.gguf"
DRAFT  = Path.home() / "models/qwen3.6-27b-dflash"
BIN    = ROOT / "dflash/build/test_dflash"
SERVER_SCRIPT = ROOT / "dflash/scripts/server.py"

if not TARGET.exists() or not BIN.exists() or not DRAFT.exists():
    print(f"SKIP: prereqs missing (target={TARGET.exists()} "
          f"draft={DRAFT.exists()} bin={BIN.exists()})")
    sys.exit(0)

# Start server with prefix cache enabled
SYSTEM = "You are a precise coding assistant. " * 200  # ~2K tokens

PORT = 18181
SERVER_LOG = open("/tmp/test_pc_server.log", "w")
proc = subprocess.Popen(
    [sys.executable, "-u", str(SERVER_SCRIPT),    # -u = unbuffered Python
     "--target", str(TARGET), "--draft", str(DRAFT), "--bin", str(BIN),
     "--max-ctx", "4096", "--port", str(PORT),
     "--prefix-cache-slots", "2"],
    stdout=SERVER_LOG, stderr=subprocess.STDOUT, bufsize=1,
)

def cleanup():
    if proc.poll() is None:
        proc.send_signal(signal.SIGINT)
        try: proc.wait(timeout=10)
        except subprocess.TimeoutExpired: proc.kill()

import atexit
atexit.register(cleanup)

# Wait for server up (poll /v1/models)
print("Waiting for server...", flush=True)
deadline = time.time() + 180
ready = False
while time.time() < deadline:
    if proc.poll() is not None:
        out = proc.stdout.read() if proc.stdout else ""
        print("SERVER DIED:\n" + out)
        sys.exit(2)
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=1).read()
        ready = True; break
    except (urllib.error.URLError, ConnectionResetError, TimeoutError):
        time.sleep(1)

if not ready:
    print("Server didn't come up within 180s")
    sys.exit(2)
print("Server up.", flush=True)


def chat(user_msg, max_tokens=8):
    payload = {
        "model": "luce-dflash",
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user",   "content": user_msg},
        ],
        "max_tokens": max_tokens, "stream": False,
    }
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
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

cleanup()

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
