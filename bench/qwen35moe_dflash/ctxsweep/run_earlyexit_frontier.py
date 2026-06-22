#!/usr/bin/env python3
"""
Early-exit frontier benchmark.
Q1: DRAFT_CTX_MAX=8192, N in {0,5,4,3} x {ctx_008192.json, needle_mid_06k.json}
Q2: DRAFT_CTX_MAX=16384, N in {0,5,4,3} x {needle_deep_12k.json, ctx_008192.json}
"""

import fcntl
import json
import os
import re
import signal
import subprocess
import sys
import time
import urllib.request
import urllib.error

BASE_DIR     = "/home/peppi/Dev/lucebox-hub"
SERVER_BIN   = f"{BASE_DIR}/server/build/dflash_server"
TARGET_MODEL = "/home/peppi/models/qwen3.6-35b-a3b/Qwen3.6-35B-A3B-UD-Q3_K_XL.gguf"
DRAFT_MODEL  = "/home/peppi/models/qwen3.6-35b-a3b-dflash-new/qwen3.6-35b-a3b-dflash-new-bf16-reconv.gguf"
CHAT_TMPL    = "/home/peppi/models/qwen3-coder-chat-template.jinja"
BENCH_DIR    = f"{BASE_DIR}/bench/qwen35moe_dflash/ctxsweep"
PORT         = 18081
HOST         = "127.0.0.1"
MAX_CTX      = 40960
FEAT_RING_CAP = 40960
HEALTH_TIMEOUT = 180
GPU_LOCK     = "/tmp/lucebox_gpu.lock"

PROMPTS = {
    "ctx_8192":     f"{BENCH_DIR}/ctx_008192.json",
    "needle_06k":   f"{BENCH_DIR}/needle_mid_06k.json",
    "needle_12k":   f"{BENCH_DIR}/needle_deep_12k.json",
}

# Grid
Q1_N_VALUES   = [0, 5, 4, 3]
Q2_N_VALUES   = [0, 5, 4, 3]
Q1_DRAFT_CTX  = 8192
Q2_DRAFT_CTX  = 16384


def health_check():
    url = f"http://{HOST}:{PORT}/health"
    try:
        with urllib.request.urlopen(url, timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def wait_healthy(timeout=HEALTH_TIMEOUT):
    start = time.time()
    while time.time() - start < timeout:
        if health_check():
            return True
        time.sleep(2)
    return False


def launch_server(draft_ctx, early_exit_n, log_path):
    env = os.environ.copy()
    env["DFLASH_DRAFT_CTX_MAX"]   = str(draft_ctx)
    env["DFLASH_FEAT_RING_CAP"]   = str(FEAT_RING_CAP)
    env["DFLASH_FEATURE_DTYPE"]   = "f16"
    if early_exit_n > 0:
        env["DFLASH_DRAFT_EARLY_EXIT_N"] = str(early_exit_n)
    else:
        env.pop("DFLASH_DRAFT_EARLY_EXIT_N", None)

    # Launch server directly (flock acquired in main via fcntl)
    cmd = [
        SERVER_BIN,
        TARGET_MODEL,
        "--draft", DRAFT_MODEL,
        "--host", HOST,
        "--port", str(PORT),
        "--max-ctx", str(MAX_CTX),
        "--max-tokens", "200",
        "--fa-window", "0",
        "--cache-type-k", "q4_0",
        "--cache-type-v", "q4_0",
        "--chat-template-file", CHAT_TMPL,
        "--model-name", "luce-dflash",
        "--lazy-draft",
    ]

    log_fd = open(log_path, "w")
    proc = subprocess.Popen(cmd, stdout=log_fd, stderr=log_fd, env=env)
    return proc, log_fd


def kill_server(proc, log_fd):
    if proc and proc.poll() is None:
        try:
            os.kill(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    time.sleep(4)
    if log_fd:
        try:
            log_fd.close()
        except Exception:
            pass


def post_prompt(prompt_path, timeout=300):
    with open(prompt_path) as f:
        payload = json.load(f)
    # Ensure temperature=0 for reproducibility
    payload.setdefault("temperature", 0.0)
    data = json.dumps(payload).encode()
    url = f"http://{HOST}:{PORT}/v1/chat/completions"
    req = urllib.request.Request(url, data=data,
                                  headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            body = r.read().decode()
            return json.loads(body), None
    except Exception as e:
        return None, str(e)


def parse_log_for_oom(log_path):
    try:
        with open(log_path) as f:
            content = f.read()
        for p in ["out of memory", "CUDA error", "cudaErrorMemoryAllocation",
                  "CUDA_ERROR_OUT_OF_MEMORY", "alloc failed"]:
            if p.lower() in content.lower():
                return True, p
        return False, None
    except Exception:
        return False, None


def parse_spec_from_log(log_path):
    """Return (accept_pct, avg_commit, decode_tps) from the LAST spec-decode line."""
    try:
        with open(log_path) as f:
            lines = f.readlines()
    except Exception:
        return None, None, None

    accept = avg_commit = decode_tps = None
    for line in lines:
        m = re.search(r'\[spec-decode\].*?(\d+(?:\.\d+)?)%.*?avg_commit=([\d.]+)', line)
        if m:
            accept = float(m.group(1))
            avg_commit = float(m.group(2))
        m2 = re.search(r'\[spec-decode\].*?speed=([\d.]+)\s*tok/s', line)
        if m2:
            decode_tps = float(m2.group(1))
        # Also try [server] chat DONE line for decode speed
        m3 = re.search(r'decode=[\d.]+s\(([\d.]+)tok/s\)', line)
        if m3:
            decode_tps = float(m3.group(1))
    return accept, avg_commit, decode_tps


def check_gate_floored(log_path):
    """Check if spec gate was floored (fell back to AR)."""
    try:
        with open(log_path) as f:
            content = f.read()
        # Check for floor reason line: [spec-gate] floor reason=slow ...
        m = re.search(r'\[spec-gate\]\s+floor\s+reason=(\S+)', content)
        if m:
            return True, m.group(1)
        # Only AR decode, no spec-decode = fully floored
        if "[ar-decode]" in content and "[spec-decode]" not in content:
            return True, "only-AR"
        return False, None
    except Exception:
        return False, None


def check_marker(response):
    if not response:
        return False
    choices = response.get("choices", [])
    if not choices:
        return False
    content = choices[0].get("message", {}).get("content", "")
    return "luce_marker_widget" in content and ("int gamma" in content or "alpha * 31" in content)


def run_cell(draft_ctx, early_exit_n, prompt_name, prompt_path, log_path):
    """Run a single benchmark cell. Returns dict of results."""
    n_label = early_exit_n if early_exit_n > 0 else "0(all6)"
    print(f"\n--- N={n_label} dc={draft_ctx} prompt={prompt_name} ---")
    print(f"  Log: {log_path}")

    proc, log_fd = launch_server(draft_ctx, early_exit_n, log_path)
    print(f"  Server PID: {proc.pid}")

    healthy = wait_healthy()
    if not healthy:
        oom, pat = parse_log_for_oom(log_path)
        status = "LOAD_OOM" if oom else "LOAD_FAIL"
        print(f"  FAIL: {status} ({pat})")
        kill_server(proc, log_fd)
        return {"n": early_exit_n, "draft_ctx": draft_ctx, "prompt": prompt_name,
                "status": status, "accept": None, "avg_commit": None,
                "decode_tps": None, "recalled": None, "gate_floor": None}

    print(f"  Server healthy. Posting prompt...")
    t0 = time.time()
    resp, err = post_prompt(prompt_path)
    elapsed = time.time() - t0
    print(f"  Response in {elapsed:.1f}s")

    oom, oom_pat = parse_log_for_oom(log_path)
    accept, avg_commit, decode_tps = parse_spec_from_log(log_path)
    floored, floor_reason = check_gate_floored(log_path)
    recalled = check_marker(resp)

    if err:
        print(f"  Request error: {err}")
        status = "REQUEST_OOM" if oom else "REQUEST_ERROR"
    else:
        status = "OK"

    # Also pull from response if log parse missed something
    if resp and (accept is None or avg_commit is None):
        usage = resp.get("usage", {})
        if accept is None:
            accept = usage.get("accept_rate")
            if accept is not None:
                accept = accept * 100  # convert 0..1 to pct
        timings = usage.get("timings", {})
        if decode_tps is None:
            decode_tps = timings.get("decode_tokens_per_sec")

    print(f"  accept={accept}% AL={avg_commit} tps={decode_tps} recalled={recalled} status={status}")
    print(f"  OOM={oom} gate_floor={floored}({floor_reason})")

    kill_server(proc, log_fd)

    return {
        "n": early_exit_n,
        "draft_ctx": draft_ctx,
        "prompt": prompt_name,
        "status": status,
        "accept": accept,
        "avg_commit": avg_commit,
        "decode_tps": decode_tps,
        "recalled": recalled,
        "oom": oom,
        "gate_floor": floored,
        "floor_reason": floor_reason,
        "elapsed_s": elapsed,
    }


def fmt_val(v, fmt=".1f"):
    return f"{v:{fmt}}" if v is not None else "---"


def main():
    print("=" * 70)
    print("EARLY-EXIT FRONTIER BENCHMARK")
    print("=" * 70)

    # Check GPU
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10
        )
        print(f"GPU: {result.stdout.strip()}")
        free_mb = int(result.stdout.strip().split(",")[0].strip().split()[0])
        if free_mb < 20000:
            print(f"WARNING: Only {free_mb} MiB free. Proceeding anyway (lazy-draft may help).")
    except Exception as e:
        print(f"nvidia-smi failed: {e}")

    # Acquire GPU flock
    lock_fd = open(GPU_LOCK, "w")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print("GPU lock acquired.")
    except OSError:
        print("GPU lock held by another process. Aborting.")
        sys.exit(1)

    results_q1 = []
    results_q2 = []

    # ---- Q1: draft_ctx=8192 ----
    print("\n" + "=" * 70)
    print("Q1: DRAFT_CTX_MAX=8192")
    print("=" * 70)

    for n in Q1_N_VALUES:
        for prompt_name, prompt_path in [
            ("ctx_8192",   PROMPTS["ctx_8192"]),
            ("needle_06k", PROMPTS["needle_06k"]),
        ]:
            log_path = f"{BENCH_DIR}/earlyexit_q1_N{n}_{prompt_name}.log"
            r = run_cell(Q1_DRAFT_CTX, n, prompt_name, prompt_path, log_path)
            results_q1.append(r)

    # ---- Q2: draft_ctx=16384 ----
    print("\n" + "=" * 70)
    print("Q2: DRAFT_CTX_MAX=16384")
    print("=" * 70)

    for n in Q2_N_VALUES:
        for prompt_name, prompt_path in [
            ("needle_12k", PROMPTS["needle_12k"]),
            ("ctx_8192",   PROMPTS["ctx_8192"]),
        ]:
            log_path = f"{BENCH_DIR}/earlyexit_q2_N{n}_{prompt_name}.log"
            r = run_cell(Q2_DRAFT_CTX, n, prompt_name, prompt_path, log_path)
            results_q2.append(r)

    # ---- Tables ----
    print("\n" + "=" * 70)
    print("Q1 RESULTS TABLE  (DRAFT_CTX_MAX=8192)")
    print("=" * 70)
    print(f"{'N':>6} | {'prompt':>12} | {'accept%':>8} | {'AL':>6} | {'tok/s':>8} | {'recalled':>8} | status")
    print("-" * 72)
    for r in results_q1:
        n_str = str(r['n']) if r['n'] > 0 else "0(all6)"
        accept_s = fmt_val(r['accept'])
        al_s     = fmt_val(r['avg_commit'])
        tps_s    = fmt_val(r['decode_tps'])
        rec_s    = "Y" if r['recalled'] else ("N" if r['recalled'] is False else "---")
        print(f"{n_str:>6} | {r['prompt']:>12} | {accept_s:>8} | {al_s:>6} | {tps_s:>8} | {rec_s:>8} | {r['status']}")

    print("\n" + "=" * 70)
    print("Q2 RESULTS TABLE  (DRAFT_CTX_MAX=16384)")
    print("=" * 70)
    print(f"{'N':>6} | {'dc':>6} | {'prompt':>12} | {'accept%':>8} | {'AL':>6} | {'tok/s':>8} | {'fit':>5} | {'gate_fl':>7} | {'recalled':>8}")
    print("-" * 90)
    for r in results_q2:
        n_str    = str(r['n']) if r['n'] > 0 else "0(all6)"
        accept_s = fmt_val(r['accept'])
        al_s     = fmt_val(r['avg_commit'])
        tps_s    = fmt_val(r['decode_tps'])
        fit_s    = "OOM" if r.get('oom') else ("OK" if r['status'] in ("OK","REQUEST_ERROR") else r['status'])
        fl_s     = "Y" if r.get('gate_floor') else "N"
        rec_s    = "Y" if r['recalled'] else ("N" if r['recalled'] is False else "---")
        print(f"{n_str:>6} | {r['draft_ctx']:>6} | {r['prompt']:>12} | {accept_s:>8} | {al_s:>6} | {tps_s:>8} | {fit_s:>5} | {fl_s:>7} | {rec_s:>8}")

    # Save JSON
    all_results = {"q1": results_q1, "q2": results_q2}
    json_path = f"{BENCH_DIR}/earlyexit_frontier.json"
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults JSON saved: {json_path}")

    # Release GPU flock
    fcntl.flock(lock_fd, fcntl.LOCK_UN)
    lock_fd.close()

    return results_q1, results_q2


if __name__ == "__main__":
    main()
