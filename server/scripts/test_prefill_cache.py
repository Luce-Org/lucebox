"""End-to-end proof for exact prefill/full-prompt cache.

Starts dflash_server with inline prefix cache disabled and prefill cache
enabled, sends the same long chat prompt three times, and asserts:

  - /props.full_cache reports enabled capacity.
  - the first request commits a full-cache entry.
  - requests 2 and 3 hit that entry.
  - warm prefill time is at least 5x faster than cold prefill.

Environment overrides:
  DFLASH_SERVER_BIN
  DFLASH_TARGET
  DFLASH_DRAFT
"""

import atexit
import json
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SERVER_BIN = Path(os.environ.get("DFLASH_SERVER_BIN", ROOT / "build/dflash_server"))
TARGET = Path(os.environ.get("DFLASH_TARGET", Path.home() / "models/Qwen3.6-27B-Q4_K_M.gguf"))
DRAFT = Path(os.environ.get("DFLASH_DRAFT", Path.home() / "models/draft/dflash-draft-3.6-q4_k_m.gguf"))
PORT = int(os.environ.get("DFLASH_PREFILL_CACHE_TEST_PORT", "18185"))
LOG_PATH = Path(os.environ.get("DFLASH_PREFILL_CACHE_TEST_LOG", "/tmp/test_prefill_cache_server.log"))


def require_path(path: Path, label: str, *, executable: bool = False) -> None:
    if not path.exists():
        print(f"SKIP: {label} missing at {path}")
        sys.exit(0)
    if executable and not os.access(path, os.X_OK):
        print(f"ERROR: {label} is not executable at {path}", file=sys.stderr)
        sys.exit(1)


def post_json(path: str, payload: dict, timeout: int = 900) -> dict:
    body = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read())


def get_json(path: str, timeout: int = 5) -> dict:
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}{path}", timeout=timeout) as resp:
        return json.loads(resp.read())


def extract_prefill_s(resp: dict) -> float:
    usage = resp.get("usage") or {}
    timings = usage.get("timings") or {}
    for key in ("prefill_s", "prompt_s", "prefill_seconds"):
        val = timings.get(key)
        if isinstance(val, (int, float)):
            return float(val)
    val = timings.get("prefill_ms")
    if isinstance(val, (int, float)):
        return float(val) / 1000.0
    raise RuntimeError(f"response did not include prefill timing: usage={usage}")


def wait_server(proc: subprocess.Popen, deadline_s: int = 240) -> None:
    deadline = time.time() + deadline_s
    while time.time() < deadline:
        if proc.poll() is not None:
            tail = LOG_PATH.read_text(errors="replace")[-4000:] if LOG_PATH.exists() else ""
            raise RuntimeError(f"server exited early; log tail:\n{tail}")
        try:
            get_json("/v1/models", timeout=1)
            return
        except (urllib.error.URLError, ConnectionResetError, TimeoutError):
            time.sleep(1)
    raise RuntimeError(f"server did not become ready within {deadline_s}s")


def main() -> int:
    require_path(SERVER_BIN, "dflash_server", executable=True)
    require_path(TARGET, "target GGUF")
    require_path(DRAFT, "draft GGUF")

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log_f = LOG_PATH.open("w")
    cmd = [
        str(SERVER_BIN),
        str(TARGET),
        "--draft", str(DRAFT),
        "--max-ctx", "16384",
        "--port", str(PORT),
        "--prefix-cache-slots", "0",
        "--prefill-cache-slots", "2",
        "--ddtree",
        "--ddtree-budget", "16",
        "--cache-type-k", "tq3_0",
        "--cache-type-v", "tq3_0",
        "--fa-window", "0",
    ]
    proc = subprocess.Popen(cmd, stdout=log_f, stderr=subprocess.STDOUT, bufsize=1)

    def cleanup() -> None:
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                proc.kill()
        log_f.close()

    atexit.register(cleanup)
    wait_server(proc)

    props = get_json("/props")
    full_cache = props.get("full_cache", {})
    print(f"startup full_cache={full_cache}", flush=True)
    if not full_cache.get("enabled") or full_cache.get("capacity") != 2:
        raise RuntimeError(f"full cache not active in /props: {full_cache}")

    filler = (
        "The repository contains an inference server, a benchmark harness, "
        "and cache implementations. This sentence is deterministic filler. "
    )
    prompt = (filler * 240) + "\n\nQuestion: Reply with exactly the word cached."
    payload = {
        "model": "dflash",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 8,
        "temperature": 0.0,
        "stream": False,
    }

    prefill = []
    for i in range(3):
        t0 = time.time()
        resp = post_json("/v1/chat/completions", payload)
        wall = time.time() - t0
        pf = extract_prefill_s(resp)
        prefill.append(pf)
        print(f"turn {i + 1}: wall={wall:.3f}s prefill={pf:.3f}s", flush=True)

    after = get_json("/props")
    full_after = after.get("full_cache", {})
    log_text = LOG_PATH.read_text(errors="replace")
    commits = log_text.count("[pc] full-cache committed")
    hits = log_text.count("[pc] full-cache hit")
    print(f"after full_cache={full_after}", flush=True)
    print(f"log commits={commits} hits={hits}", flush=True)

    if full_after.get("in_use", 0) < 1:
        raise RuntimeError(f"full cache did not retain an entry: {full_after}")
    if full_after.get("lifetime_hits", 0) < 2 or hits < 2:
        raise RuntimeError(f"full cache did not hit twice: props={full_after} log_hits={hits}")
    if commits < 1:
        raise RuntimeError("full cache did not log a committed entry")

    cold = prefill[0]
    warm_best = min(prefill[1:])
    speedup = cold / max(warm_best, 0.001)
    suffix = " lower-bound" if warm_best < 0.001 else ""
    print(f"best warm speedup={speedup:.2f}x{suffix}", flush=True)
    if speedup < 5.0:
        raise RuntimeError(f"expected >=5x prefill speedup, got {speedup:.2f}x")

    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
