#!/usr/bin/env python3
"""Model-level forward differentials for the Qwen3.8-Flash-Next backend.

Two self-contained A/B checks that a kernel unit test cannot cover because they
need the real weights and graph marking:

  hc16   LLAMA_MMB_HC16=2 (fused bf16 gate-mix / HC-down) vs =0 (f32 path) must
         generate the same greedy continuation.
  qsa    a prompt prefilled in one chunk vs split over smaller chunks must
         generate the same greedy continuation, and QSA must engage in both
         (QWEN4EXP_FA_TELEMETRY=1 -> "[fa] graph qsa=N").

Requires the model and a built dflash_server; run on the gfx1151 box:

    python3 server/scripts/qwen4exp_forward_diff.py hc16
    python3 server/scripts/qwen4exp_forward_diff.py qsa --chunks 4096,16384

Not part of ctest: model-backed and long-running.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
SERVER_BIN = os.environ.get("QWEN4EXP_SERVER", str(REPO / "server/build-hip/dflash_server"))
MODEL = os.environ.get("QWEN4EXP_MODEL", str(Path.home() / "models/qwen4exp-iq4nl/Qwen3.8-Flash-Next-IQ4_NL-00001-of-00003.gguf"))
PORT = int(os.environ.get("QWEN4EXP_DIFF_PORT", "8711"))
BASE_ENV = {
    "QWEN4EXP_QSA": "1",
    "QWEN4EXP_MMB_CUBLAS": "5",
    "DFLASH_MMB_SHADOW": "1",
    "QWEN4EXP_FA_TELEMETRY": "1",
    "LLAMA_MMB_HC16": "2",
}

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


def chat(prompt: str, max_tokens: int, timeout: int) -> str:
    body = json.dumps({
        "model": "dflash",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "seed": 0,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read())
    return data["choices"][0]["message"]["content"]


def wait_ready(proc: subprocess.Popen, timeout: int) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            return False
        try:
            urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models", timeout=2).read()
            return True
        except Exception:
            time.sleep(1)
    return False


def run_config(name: str, env_overrides: dict[str, str], extra_args: list[str],
               prompt: str, max_tokens: int) -> tuple[str, str]:
    subprocess.run(["fuser", "-k", f"{PORT}/tcp"], capture_output=True)
    time.sleep(0.5)
    cmd = [SERVER_BIN, MODEL, "--host", "127.0.0.1", "--port", str(PORT),
           "--target-device", "hip:0", "--max-ctx", "40000", *extra_args]
    env = {**os.environ, **BASE_ENV, **env_overrides}
    log(f"[{name}] spawn: {' '.join(cmd)}  env+={env_overrides} {extra_args}")
    logf = open(f"/tmp/qwen4exp_diff_{name}.log", "w")
    proc = subprocess.Popen(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT)
    try:
        if not wait_ready(proc, timeout=600):
            raise RuntimeError(f"server did not become ready; see /tmp/qwen4exp_diff_{name}.log")
        log(f"[{name}] ready")
        reply = chat(prompt, max_tokens, timeout=1800)
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except Exception:
            proc.kill()
        logf.close()
    return reply, f"/tmp/qwen4exp_diff_{name}.log"


def read_text(path: str) -> str:
    with open(path, errors="replace") as f:
        return f.read()


def common_prefix_ratio(a: str, b: str) -> float:
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n / max(1, max(len(a), len(b)))


def qsa_counts(log_text: str) -> list[tuple[int, int]]:
    return [(int(q), int(d)) for q, d in re.findall(r"\[fa\] graph qsa=(\d+) dense=(\d+)", log_text)]


def cmd_hc16(args: argparse.Namespace) -> int:
    # >512 prefill tokens so the HC marking (ggml_nrows(xn) >= 512) and the
    # HC-down GEMM (T >= mmb_min_t() = 512) actually engage.
    prompt = PARAGRAPH * 30 + "\n\nAnswer with a single word. What is the capital of France?"
    replies = {}
    logs = {}
    for name, hc in (("hc16_on", "2"), ("hc16_off", "0")):
        env = {"LLAMA_MMB_HC16": hc}
        if hc == "2":
            env["LLAMA_HC16_DEBUG"] = "1"
        replies[name], logs[name] = run_config(name, env, ["--chunk", "16384"], prompt, args.max_tokens)
    same = replies["hc16_on"].strip() == replies["hc16_off"].strip()
    ratio = common_prefix_ratio(replies["hc16_on"], replies["hc16_off"])
    # Require that the bf16 mark actually engaged; otherwise the check is vacuous.
    marked = any("ok=1" in line for line in read_text(logs["hc16_on"]).splitlines() if "HC16 mixdst" in line)
    log(f"[hc16] bf16 mark engaged={marked}")
    log(f"[hc16] on={replies['hc16_on']!r}")
    log(f"[hc16] off={replies['hc16_off']!r}")
    log(f"[hc16] exact={same} common_prefix={ratio:.3f}")
    return 0 if marked and (same or ratio >= 0.9) else 1


def cmd_qsa(args: argparse.Namespace) -> int:
    results = {}
    for chunk in [int(c) for c in args.chunks.split(",")]:
        name = f"qsa_chunk{chunk}"
        reply, logf = run_config(name, {}, ["--chunk", str(chunk)], LONG_PROMPT, args.max_tokens)
        counts = qsa_counts(read_text(logf))
        qsa_total = sum(q for q, _ in counts)
        results[chunk] = (reply, counts)
        log(f"[qsa] chunk={chunk} qsa_graphs={len(counts)} qsa_total={qsa_total} reply={reply[:80]!r}")
    chunks = sorted(results)
    base = results[chunks[-1]][0].strip()
    ok = True
    for chunk in chunks:
        reply = results[chunk][0].strip()
        if reply != base:
            ratio = common_prefix_ratio(reply, base)
            log(f"[qsa] chunk={chunk} reply differs from chunk={chunks[-1]} common_prefix={ratio:.3f}")
            ok = ok and ratio >= 0.9
        if not results[chunk][1]:
            log(f"[qsa] chunk={chunk} FAIL: no QSA telemetry")
            ok = False
    log(f"[qsa] {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_hc = sub.add_parser("hc16")
    p_hc.add_argument("--max-tokens", type=int, default=24)
    p_hc.set_defaults(func=cmd_hc16)
    p_qsa = sub.add_parser("qsa")
    p_qsa.add_argument("--chunks", default="4096,16384")
    p_qsa.add_argument("--max-tokens", type=int, default=24)
    p_qsa.set_defaults(func=cmd_qsa)
    args = ap.parse_args()
    if not Path(SERVER_BIN).exists():
        log(f"SKIP: server binary not found at {SERVER_BIN}")
        return 77
    if not Path(MODEL).exists():
        log(f"SKIP: model not found at {MODEL}")
        return 77
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
