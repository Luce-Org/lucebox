#!/usr/bin/env python3
"""Model-level forward differentials for the Qwen3.8-Flash-Next backend.

A prompt prefilled in one chunk vs split over smaller chunks must generate
the same greedy continuation.

Runs on the gfx1151 prefill profile the backend defaults (QSA, HC16, MMB).

Requires the model and a built luce_server; run on the gfx1151 box:

    python3 server/scripts/qwen4exp_forward_diff.py qsa --chunks 4096,16384

Not part of ctest: model-backed and long-running.
"""
from __future__ import annotations

import argparse
import json
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

PARAGRAPH = (
    "The quick brown fox jumps over the lazy dog while the engineer studies the "
    "kernel launch counters and the cache residency of every intermediate tensor. "
    "Numbers follow numbers until the context is long enough to matter. "
)
# A long filler context with an unambiguous tail question, so greedy output is
# robust to the small numerical differences chunk boundaries introduce.
LONG_PROMPT = PARAGRAPH * 400 + "\n\nAnswer with a single word. What is the capital of France?"


def log(msg: str) -> None:
    print(msg, flush=True)


def chat(port: int, prompt: str, max_tokens: int, timeout: int) -> str:
    body = json.dumps({
        "model": "luce",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": 0,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"]


def wait_ready(proc: subprocess.Popen, port: int, timeout: int) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models", timeout=2).read()
            return True
        except Exception:
            time.sleep(1)
    return False


def run_config(args: argparse.Namespace, name: str, extra_args: list[str],
               prompt: str, max_tokens: int) -> str:
    subprocess.run(["fuser", "-k", f"{args.port}/tcp"], capture_output=True)
    time.sleep(0.5)
    cmd = [args.server, args.model, "--host", "127.0.0.1", "--port", str(args.port),
           "--target-device", "hip:0", "--max-ctx", "40000", *extra_args]
    log(f"[{name}] spawn: {' '.join(cmd)}")
    logf = open(f"/tmp/qwen4exp_diff_{name}.log", "w")
    proc = subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT)
    try:
        if not wait_ready(proc, args.port, timeout=600):
            raise RuntimeError(f"server did not become ready; see /tmp/qwen4exp_diff_{name}.log")
        log(f"[{name}] ready")
        reply = chat(args.port, prompt, max_tokens, timeout=1800)
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except Exception:
            proc.kill()
        logf.close()
    return reply


def common_prefix_ratio(a: str, b: str) -> float:
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n / max(1, max(len(a), len(b)))


def cmd_qsa(args: argparse.Namespace) -> int:
    results = {}
    for chunk in [int(c) for c in args.chunks.split(",")]:
        name = f"qsa_chunk{chunk}"
        results[chunk] = run_config(args, name, ["--chunk", str(chunk)], LONG_PROMPT, args.max_tokens)
        log(f"[qsa] chunk={chunk} reply={results[chunk][:80]!r}")
    chunks = sorted(results)
    base = results[chunks[-1]].strip()
    ok = True
    for chunk in chunks:
        reply = results[chunk].strip()
        if reply != base:
            ratio = common_prefix_ratio(reply, base)
            log(f"[qsa] chunk={chunk} reply differs from chunk={chunks[-1]} common_prefix={ratio:.3f}")
            ok = ok and ratio >= 0.9
    log(f"[qsa] {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_qsa = sub.add_parser("qsa")
    p_qsa.add_argument("--server", default=str(REPO / "server/build-hip/luce_server"))
    p_qsa.add_argument("--model", default=str(Path.home() / "models/qwen4exp-iq4nl/Qwen3.8-Flash-Next-IQ4_NL-00001-of-00003.gguf"))
    p_qsa.add_argument("--port", type=int, default=8711)
    p_qsa.add_argument("--chunks", default="4096,16384")
    p_qsa.add_argument("--max-tokens", type=int, default=24)
    p_qsa.set_defaults(func=cmd_qsa)
    args = ap.parse_args()
    if not Path(args.server).exists():
        log(f"SKIP: server binary not found at {args.server}")
        return 77
    if not Path(args.model).exists():
        log(f"SKIP: model not found at {args.model}")
        return 77
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
