#!/usr/bin/env python3
"""
Full KV-quant x draft_ctx grid sweep.
For each KV in {f16, q8_0, q4_0, tq3_0} x draft_ctx in {8192, 16384, 24576, 32768}:
  Step A: find max-fitting draft_ctx at ~18K prompt (OOM detection)
  Step B: at each KV's max-fitting draft_ctx, run three needle prompts.

Holds /tmp/lucebox_gpu.lock for the entire run.
Fixed config: Q3_K_XL all-hot, port 18081, max-ctx 40960,
              DFLASH_FEAT_RING_CAP=40960, --fa-window 0, bf16-reconv drafter, --lazy-draft.
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
HEALTH_TIMEOUT = 240  # seconds — large models take time

# Grid dimensions
KV_QUANTS    = ["f16", "q8_0", "q4_0", "tq3_0"]
DRAFT_CTX_VALUES = [8192, 16384, 24576, 32768]

# Needle prompts: (filename, tag, approx_marker_dist_from_end)
NEEDLE_PROMPTS = [
    ("needle_mid_06k.json", "06k", 2593),
    ("needle_mid_08k.json", "08k", 4854),
    ("needle_mid_12k.json", "12k", 7987),
]

PROMPT_18K = f"{BENCH_DIR}/ctx_016384.json"


# ---- helpers ----

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
        time.sleep(3)
    return False

def launch_server(kv_quant, draft_ctx, log_path):
    env = os.environ.copy()
    env["DFLASH_DRAFT_CTX_MAX"]  = str(draft_ctx)
    env["DFLASH_FEAT_RING_CAP"]   = str(FEAT_RING_CAP)

    cmd = [
        SERVER_BIN,
        TARGET_MODEL,
        "--draft", DRAFT_MODEL,
        "--host", HOST,
        "--port", str(PORT),
        "--max-ctx", str(MAX_CTX),
        "--max-tokens", "200",
        "--fa-window", "0",
        "--cache-type-k", kv_quant,
        "--cache-type-v", kv_quant,
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
    time.sleep(5)  # extra settle after GPU frees
    if log_fd:
        try:
            log_fd.close()
        except Exception:
            pass

def post_prompt(prompt_path, timeout=360):
    with open(prompt_path) as f:
        payload = json.load(f)
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

def is_oom(log_path):
    try:
        with open(log_path) as f:
            content = f.read()
        patterns = ["out of memory", "cuda error", "cudaerrormemoryal",
                    "cuda_error_out_of_memory", "alloc failed", "oom"]
        content_lower = content.lower()
        for p in patterns:
            if p in content_lower:
                return True, p
        return False, None
    except Exception:
        return False, None

def parse_log_stats(log_path):
    """Extract last accept%, avg_commit, decode tok/s from server log."""
    try:
        with open(log_path) as f:
            lines = f.readlines()
    except Exception:
        return None, None, None, None

    accept_pct = None
    avg_commit = None
    decode_tps = None
    kv_cache_gib = None

    for line in lines:
        # kv_cache from dynamic placement
        m = re.search(r'kv_cache=([0-9.]+)\s*GiB', line)
        if m:
            kv_cache_gib = float(m.group(1))

        # spec-decode stats
        m = re.search(r'accepted=\d+/\d+\s+\(([0-9.]+)%\)\s+avg_commit=([0-9.]+)', line)
        if m:
            accept_pct = float(m.group(1))
            avg_commit = float(m.group(2))

        # decode tok/s from DONE line
        m = re.search(r'decode=[0-9.]+s\(([0-9.]+)tok/s\)', line)
        if m:
            decode_tps = float(m.group(1))

    return accept_pct, avg_commit, decode_tps, kv_cache_gib

def check_marker_in_response(resp):
    """Check if luce_marker_widget and expected content appear in output."""
    if not resp:
        return False
    choices = resp.get("choices", [])
    if not choices:
        return False
    content = choices[0].get("message", {}).get("content", "")
    return "luce_marker_widget" in content

def get_usage(resp):
    if not resp:
        return None, None
    usage = resp.get("usage", {})
    return usage.get("completion_tokens"), usage.get("prompt_tokens")


# ---- Step A: VRAM ceiling sweep for one KV quant ----

def step_a_vram_sweep(kv_quant):
    """Find max draft_ctx that loads+serves ~18K without OOM. Returns (max_ok, first_oom, per_ctx_data)."""
    print(f"\n=== Step A: VRAM ceiling sweep KV={kv_quant} ===")
    per_ctx = []
    max_ok = None
    first_oom = None

    for draft_ctx in DRAFT_CTX_VALUES:
        tag = f"{kv_quant}_ctx{draft_ctx}"
        log_path = f"{BENCH_DIR}/grid_{tag}.log"
        print(f"  draft_ctx={draft_ctx} -> log: grid_{tag}.log")

        proc, log_fd = launch_server(kv_quant, draft_ctx, log_path)
        print(f"    PID={proc.pid}")

        healthy = wait_healthy()

        # Check for early crash / OOM
        oom, oom_pat = is_oom(log_path)

        if not healthy:
            print(f"    NOT healthy. OOM={oom} ({oom_pat})")
            kill_server(proc, log_fd)
            rec = {"draft_ctx": draft_ctx, "status": "LOAD_OOM" if oom else "LOAD_FAIL",
                   "oom_pattern": oom_pat}
            per_ctx.append(rec)
            if first_oom is None:
                first_oom = draft_ctx
            continue

        if oom:
            print(f"    OOM detected at startup: {oom_pat}")
            kill_server(proc, log_fd)
            rec = {"draft_ctx": draft_ctx, "status": "LOAD_OOM", "oom_pattern": oom_pat}
            per_ctx.append(rec)
            if first_oom is None:
                first_oom = draft_ctx
            continue

        print(f"    Server healthy, posting ~18K prompt...")
        t0 = time.time()
        resp, err = post_prompt(PROMPT_18K)
        elapsed = time.time() - t0

        oom2, oom_pat2 = is_oom(log_path)
        accept, commit, decode_tps, kv_gib = parse_log_stats(log_path)

        if err or oom2:
            print(f"    Request failed: err={err}, OOM={oom2} ({oom_pat2})")
            kill_server(proc, log_fd)
            rec = {"draft_ctx": draft_ctx,
                   "status": "SERVE_OOM" if oom2 else "REQUEST_ERROR",
                   "error": err, "oom_pattern": oom_pat2}
            per_ctx.append(rec)
            if first_oom is None:
                first_oom = draft_ctx
            continue

        comp_toks, prompt_toks = get_usage(resp)
        print(f"    OK: accept={accept}%, commit={commit}, decode={decode_tps} tok/s, elapsed={elapsed:.1f}s")
        print(f"    prompt_tokens={prompt_toks}, kv_cache_gib={kv_gib}")

        choices = resp.get("choices", []) if resp else []
        output = choices[0].get("message", {}).get("content", "") if choices else ""
        print(f"    output[:100]: {output[:100]!r}")

        kill_server(proc, log_fd)
        rec = {
            "draft_ctx": draft_ctx,
            "status": "OK",
            "accept_pct": accept,
            "avg_commit": commit,
            "decode_tps": decode_tps,
            "kv_cache_gib": kv_gib,
            "prompt_tokens": prompt_toks,
            "completion_tokens": comp_toks,
            "elapsed_s": round(elapsed, 1),
        }
        per_ctx.append(rec)
        max_ok = draft_ctx

    print(f"  KV={kv_quant}: max_ok={max_ok}, first_oom={first_oom}")
    return max_ok, first_oom, per_ctx


# ---- Step B: Needle recall at max-fitting draft_ctx ----

def step_b_needle_recall(kv_quant, draft_ctx):
    """Run three needle prompts at the given (KV, draft_ctx)."""
    print(f"\n=== Step B: Needle recall KV={kv_quant} draft_ctx={draft_ctx} ===")

    tag = f"{kv_quant}_needle_ctx{draft_ctx}"
    log_path = f"{BENCH_DIR}/grid_{tag}.log"

    proc, log_fd = launch_server(kv_quant, draft_ctx, log_path)
    print(f"  PID={proc.pid}")

    healthy = wait_healthy()
    if not healthy:
        oom, pat = is_oom(log_path)
        print(f"  Server did NOT become healthy. OOM={oom}")
        kill_server(proc, log_fd)
        return [{"needle": n[1], "marker_dist": n[2], "status": "SERVER_FAIL"}
                for n in NEEDLE_PROMPTS]

    results = []
    for fname, tag_n, dist in NEEDLE_PROMPTS:
        prompt_path = f"{BENCH_DIR}/{fname}"
        print(f"  Needle {tag_n} (marker ~{dist} tok from end)...")
        t0 = time.time()
        resp, err = post_prompt(prompt_path)
        elapsed = time.time() - t0

        if err:
            oom, pat = is_oom(log_path)
            print(f"    Error: {err}, OOM={oom}")
            results.append({"needle": tag_n, "marker_dist": dist,
                            "status": "OOM" if oom else "ERROR", "error": err})
            continue

        accept, commit, decode_tps, _ = parse_log_stats(log_path)
        recalled = check_marker_in_response(resp)
        comp_toks, prompt_toks = get_usage(resp)
        choices = resp.get("choices", []) if resp else []
        output = choices[0].get("message", {}).get("content", "") if choices else ""

        print(f"    accept={accept}%, commit={commit}, decode={decode_tps} tok/s")
        print(f"    recalled={recalled}, elapsed={elapsed:.1f}s")
        print(f"    output[:120]: {output[:120]!r}")

        results.append({
            "needle": tag_n,
            "marker_dist": dist,
            "status": "OK",
            "accept_pct": accept,
            "avg_commit": commit,
            "decode_tps": decode_tps,
            "recalled": recalled,
            "prompt_tokens": prompt_toks,
            "completion_tokens": comp_toks,
            "elapsed_s": round(elapsed, 1),
        })

    kill_server(proc, log_fd)
    return results


# ---- main ----

def main():
    # Hold GPU lock for the entire run
    lock_fd = open("/tmp/lucebox_gpu.lock", "w")
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except IOError:
        print("GPU lock held by another process. Waiting with blocking lock...")
        fcntl.flock(lock_fd, fcntl.LOCK_EX)

    print("GPU lock acquired.")

    grid_results = {}

    try:
        for kv in KV_QUANTS:
            print(f"\n{'='*60}")
            print(f"KV QUANT: {kv}")
            print(f"{'='*60}")

            # Step A
            max_ok, first_oom, per_ctx = step_a_vram_sweep(kv)

            # Step B at max-fitting
            needle_results = []
            if max_ok is not None:
                needle_results = step_b_needle_recall(kv, max_ok)
            else:
                print(f"  No fitting draft_ctx for KV={kv}, skipping needle step.")

            grid_results[kv] = {
                "max_fitting_draft_ctx": max_ok,
                "first_oom_draft_ctx": first_oom,
                "step_a": per_ctx,
                "step_b": needle_results,
            }

        # Save
        out_path = f"{BENCH_DIR}/kv_grid_results.json"
        with open(out_path, "w") as f:
            json.dump(grid_results, f, indent=2)
        print(f"\nResults saved to {out_path}")

    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()
        print("GPU lock released.")

    # Print summary
    print("\n=== GRID SUMMARY ===")
    for kv, data in grid_results.items():
        print(f"\nKV={kv}: max_fitting={data['max_fitting_draft_ctx']}, first_oom={data['first_oom_draft_ctx']}")
        for r in data["step_a"]:
            status = r["status"]
            dc = r["draft_ctx"]
            tps = r.get("decode_tps", "N/A")
            acc = r.get("accept_pct", "N/A")
            kv_gib = r.get("kv_cache_gib", "N/A")
            print(f"  draft_ctx={dc}: {status} | decode={tps} tok/s | accept={acc}% | kv_gib={kv_gib}")
        print("  Needle recall:")
        for nr in data["step_b"]:
            print(f"    {nr.get('needle')}: recalled={nr.get('recalled')}, accept={nr.get('accept_pct')}%, decode={nr.get('decode_tps')} tok/s")

    return grid_results


if __name__ == "__main__":
    results = main()
    print("\n=== FINAL JSON ===")
    print(json.dumps(results, indent=2))
