"""End-to-end proof for exact prefill disk cache.

Starts dflash_server with exact prefill RAM disabled and exact prefill disk
enabled, sends the same long chat prompt three times, and asserts:

  - /props.cache.prefill reports a disk budget.
  - the first request saves an exact prefill snapshot to disk.
  - requests 2 and 3 restore that disk snapshot.
  - warm prefill time is at least 5x faster than cold prefill.

Environment overrides:
  DFLASH_SERVER_BIN
  DFLASH_TARGET
  DFLASH_DRAFT
"""

import atexit
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SERVER_BIN = Path(os.environ.get("DFLASH_SERVER_BIN", ROOT / "build/dflash_server"))
TARGET = Path(os.environ.get("DFLASH_TARGET", Path.home() / "models/Qwen3.6-27B-Q4_K_M.gguf"))
DRAFT = Path(os.environ.get("DFLASH_DRAFT", Path.home() / "models/draft/dflash-draft-3.6-q4_k_m.gguf"))
PORT = int(os.environ.get("DFLASH_PREFILL_DISK_CACHE_TEST_PORT", "18186"))
LOG_PATH = Path(os.environ.get("DFLASH_PREFILL_DISK_CACHE_TEST_LOG", "/tmp/test_prefill_disk_cache_server.log"))


def require_path(path: Path, label: str) -> None:
    if not path.exists():
        print(f"SKIP: {label} missing at {path}")
        sys.exit(0)


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
    require_path(SERVER_BIN, "dflash_server")
    require_path(TARGET, "target GGUF")
    require_path(DRAFT, "draft GGUF")

    cache_dir = Path(tempfile.mkdtemp(prefix="prefill-disk-cache-"))
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    log_f = LOG_PATH.open("w")
    cmd = [
        str(SERVER_BIN),
        str(TARGET),
        "--draft", str(DRAFT),
        "--max-ctx", "16384",
        "--port", str(PORT),
        "--cache-ram", "0",
        "--kv-cache-dir", str(cache_dir),
        "--cache-disk", "1GiB",
        "--cache-prefix-disk", "0",
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
        shutil.rmtree(cache_dir, ignore_errors=True)

    atexit.register(cleanup)
    wait_server(proc)

    props = get_json("/props")
    prefill_pool = ((props.get("cache") or {}).get("prefill") or {})
    print(f"startup prefill_pool={prefill_pool}", flush=True)
    if prefill_pool.get("ram_budget_bytes") != 0:
        raise RuntimeError(f"prefill RAM should be disabled: {prefill_pool}")
    if prefill_pool.get("disk_budget_bytes", 0) <= 0:
        raise RuntimeError(f"prefill disk not active in /props: {prefill_pool}")

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
    prefill_after = ((after.get("cache") or {}).get("prefill") or {})
    log_text = LOG_PATH.read_text(errors="replace")
    saves = log_text.count("[disk-cache] saved")
    hits = log_text.count("[prefill-disk] hit")
    print(f"after prefill_pool={prefill_after}", flush=True)
    print(f"log disk_saves={saves} prefill_disk_hits={hits}", flush=True)

    if prefill_after.get("disk_bytes", 0) <= 0:
        raise RuntimeError(f"prefill disk bytes did not increase: {prefill_after}")
    if saves < 1:
        raise RuntimeError("prefill disk cache did not save a snapshot")
    if hits < 2:
        raise RuntimeError(f"prefill disk cache did not hit twice: hits={hits}")

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
